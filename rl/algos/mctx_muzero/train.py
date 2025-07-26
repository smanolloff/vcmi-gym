import jax
import jax.numpy as jnp
import optax
import muax

# from mctx import muzero_policy, RootFnOutput, RecurrentFnOutput

from .mz_model import MZModel
from .predictor import PredictorWrapper

from rl.world.util.constants_v12 import (
    STATE_SIZE,
    STATE_SIZE_ONE_HEX,
    GLOBAL_ATTR_MAP,
    HEX_ATTR_MAP,
    N_HEX_ACTIONS,
    DIM_OTHER,
)


def get_action_mask_functor():
    # ----- 1) Precompute the flat indices once -----
    _, gstart, glen, *_ = GLOBAL_ATTR_MAP["ACTION_MASK"]
    _, hstart, hlen, *_ = HEX_ATTR_MAP["ACTION_MASK"]

    # index range for the global mask:
    global_idx = jnp.arange(gstart, gstart + glen)

    # index range for each hex; obs[:, DIM_OTHER:] is flattened [B, 165*HEX_D]
    hex_base = DIM_OTHER
    hex_stride = STATE_SIZE_ONE_HEX
    # offsets for each of the 165 hex slots
    hex_offsets = jnp.arange(165) * hex_stride
    # within‑hex offsets for the mask
    inner = jnp.arange(hstart, hstart + hlen)

    hex_idx = (hex_base + hex_offsets[:, None] + inner[None, :]).reshape(-1)

    # full action‑mask indices
    ACTION_MASK_IDX = jnp.concatenate([global_idx, hex_idx], axis=0)  # shape (N_ACTIONS,)

    # ----- 2) Fast get_action_mask -----
    @jax.jit
    def get_action_mask(obs: jnp.ndarray) -> jnp.ndarray:
        # obs: [B, TOTAL_FEATURES]
        # Gather all mask bits in one go:
        return obs[:, ACTION_MASK_IDX]


def get_action_mask_slow(obs):
    map_gmask = GLOBAL_ATTR_MAP["ACTION_MASK"]
    map_hexmask = HEX_ATTR_MAP["ACTION_MASK"]
    global_actmask = obs[:, map_gmask[1]:map_gmask[1]+map_gmask[2]]
    # => (B, 2,)

    obs_hexes = obs[:, DIM_OTHER:].reshape(-1, 165, STATE_SIZE_ONE_HEX)
    hex_actmask = obs_hexes[:, :, map_hexmask[1]:map_hexmask[1]+map_hexmask[2]]
    # => (B, 165, N_HEX_ACTIONS)

    actmask = jnp.concat([global_actmask, hex_actmask.reshape(-1, 165*N_HEX_ACTIONS)], axis=1)
    # => (B, N_ACTIONS)

    return actmask


if __name__ == "__main__":
    rngs = {"params": jax.random.PRNGKey(0)}
    env_kwargs = dict(
        mapname="gym/generated/evaluation/8x512.vmap",
        opponent="BattleAI",
        role="defender",
        swap_sides=0,
        random_heroes=0,
        random_obstacles=0,
        town_chance=0,
        warmachine_chance=0,
        random_terrain_chance=0,
        random_stack_chance=0,
        tight_formation_chance=0,
        reward_step_fixed=-0.01,
        reward_dmg_mult=0.01,
        reward_term_mult=0.01,
        reward_relval_mult=0.01,
    )

    predictorw = PredictorWrapper(
        jit=False,  # very slow, maybe due to differing batch; or due to CPU
        max_transitions=5,
        side=(env_kwargs["role"] == "defender"),
        reward_dmg_mult=env_kwargs["reward_dmg_mult"],
        reward_term_mult=env_kwargs["reward_term_mult"],
    )

    params = predictorw.init(
        jax.random.PRNGKey(0),
        initial_state=jnp.zeros([1, STATE_SIZE]),
        initial_action=jnp.array([1])
    )

    params = predictorw.load(params)

    from functools import partial
    predictor_fwd_no_params = partial(predictorw.apply, params)

    get_action_mask = get_action_mask_functor()

    # @jax.jit
    def repr_fn(obs):
        # here we treat raw obs as the "latent" root
        return obs

    # @jax.jit
    def pred_fn(state):
        # XXX: "latent" state is simply obs in this model

        # produce policy logits & value
        model = MZModel(depth=3)
        logits, value = model(state)
        mask = get_action_mask(state)
        neg_inf = -1e9
        masked_logits = jnp.where(mask, logits, neg_inf)
        return masked_logits, value

    # @jax.jit
    def dyn_fn(state, action):
        next_obs, reward, term, num_t = predictor_fwd_no_params(state, action)
        # next_obs becomes the next latent state in this model
        return next_obs, reward

    # 5) Bundle into a MuZero model
    gradient_transform = muax.model.optimizer(
        init_value=0.02, peak_value=0.02, end_value=0.002,
        warmup_steps=5_000, transition_steps=5_000
    )

    model = muax.MuZero(
        repr_fn,
        pred_fn,
        dyn_fn,
        policy='muzero',
        discount=0.99,
        optimizer=gradient_transform,
        support_size=10
    )

    # 6) Freeze your two pretrained sub‐trees so only the prediction head is trained
    def mask_fn(param_path, _):
        # param_path is a tuple like ('obs', ...), ('rew', ...), or ('pred', ...)
        # we return False (freeze) for 'obs' and 'rew'
        import ipdb; ipdb.set_trace()  # noqa
        return param_path[0] not in ["obs", "rew", "act"]

    opt = optax.masked(gradient_transform, mask_fn)     # freezes obs & reward params :contentReference[oaicite:1]{index=1}

    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
    env = VcmiEnv(**env_kwargs)
    env.reset()

    # TODO: `env` and test_env` must be DIFFERENT
    #       using the same just to test for exceptions
    #       (don't have "proc" vcmienv connectors anymore)
    #test_env = VcmiEnv(**env_kwargs)
    test_env = env

    # 7) Train with muax.fit
    model_path = muax.fit(
        model,
        env=env,
        test_env=test_env,
        max_episodes=100,
        max_training_steps=500,
        tracer=muax.PNStep(5, 0.99, 0.5),
        buffer=muax.TrajectoryReplayBuffer(500),
        k_steps=5,
        sample_per_trajectory=1,
        num_trajectory=16,
        tensorboard_dir='./tb',
        model_save_path='./models',
        save_name='muzero_custom',
        random_seed=0,
        log_all_metrics=True
    )
