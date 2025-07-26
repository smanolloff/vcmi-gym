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

    return get_action_mask

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


def create_venv(env_kwargs, num_envs, sync=True):
    import gymnasium as gym
    import os
    from types import SimpleNamespace
    from functools import partial
    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
    from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper

    # AsyncVectorEnv creates a dummy_env() in the main process just to
    # extract metadata, which causes VCMI init pid error afterwards
    pid = os.getpid()
    dummy_env = SimpleNamespace(
        metadata={'render_modes': ['ansi', 'rgb_array'], 'render_fps': 30},
        render_mode='ansi',
        action_space=VcmiEnv.ACTION_SPACE,
        observation_space=VcmiEnv.OBSERVATION_SPACE["observation"],
        close=lambda: None,
    )

    def env_creator(i):
        if os.getpid() == pid and not sync:
            return dummy_env

        # env = VcmiEnv(**env_kwargs)
        env = gym.make("VCMI-v12", max_episode_steps=100, **env_kwargs)
        env = LegacyObservationSpaceWrapper(env)
        return env

    funcs = [partial(env_creator, i) for i in range(num_envs)]

    # XXX: SyncVectorEnv won't work when both train and eval env are started in main process
    #       => consider always using AsyncVectorEnv:
    #       vec_env = gym.vector.AsyncVectorEnv(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    if num_envs > 1:
        vec_env = gym.vector.AsyncVectorEnv(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        vec_env = gym.vector.SyncVectorEnv(funcs, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    vec_env.reset()

    return vec_env


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
        jit=True,
        max_transitions=5,
        side=(env_kwargs["role"] == "defender"),
        reward_dmg_mult=env_kwargs["reward_dmg_mult"],
        reward_term_mult=env_kwargs["reward_term_mult"],
    )

    params = predictorw.init(
        jax.random.PRNGKey(0),
        initial_state=jnp.zeros([1, STATE_SIZE]),
        initial_action=jnp.array([1], dtype=jnp.int32)
    )

    params = predictorw.load(params)

    from functools import partial
    predictor_fwd_no_params = partial(predictorw.apply, params)

    get_action_mask = get_action_mask_functor()

    # @jax.jit
    def repr_fn(obs):
        print("[main] repr_fn")
        # here we treat raw obs as the "latent" root
        print("[main] repr_fn [return]")
        return obs

    # @jax.jit
    def pred_fn(state):
        print("[main] pred_fn")
        # XXX: "latent" state is simply obs in this model

        # produce policy logits & value
        model = MZModel(depth=3)
        logits, value = model(state)
        mask = get_action_mask(state)
        neg_inf = -1e9
        masked_logits = jnp.where(mask, logits, neg_inf)

        # XXX: muax expects (v, logits), not (logits, v)
        # https://github.com/bwfbowen/muax/blob/4a77962d4adc2a7d63561d3cde31ccafb061a297/muax/nn.py#L37
        print("[main] pred_fn [return]")
        return value, masked_logits

    # @jax.jit
    def dyn_fn(state, action):
        print("[main] dyn_fn")
        # XXX: muax naively assumes state and action are both float32
        #      when calling dyn_fn.init()
        #   https://github.com/bwfbowen/muax/blob/4a77962d4adc2a7d63561d3cde31ccafb061a297/muax/train.py#L137
        #   https://github.com/bwfbowen/muax/blob/4a77962d4adc2a7d63561d3cde31ccafb061a297/muax/model.py#L132
        # Real action dtype is int64, but jax does not handle it natively => use int32
        action = action.astype(jnp.int32)

        next_obs, reward, term, num_t = predictor_fwd_no_params(state, action)

        # XXX: muax expects (r, s), not (s, r)
        # https://github.com/bwfbowen/muax/blob/4a77962d4adc2a7d63561d3cde31ccafb061a297/muax/nn.py#L61
        print("[main] dyn_fn [return]")
        return reward, next_obs

    print("[main] init optimizer")
    # 5) Bundle into a MuZero model
    gradient_transform = muax.model.optimizer(
        init_value=0.02, peak_value=0.02, end_value=0.002,
        warmup_steps=5, transition_steps=5
        # warmup_steps=5_000, transition_steps=5_000
    )

    print("[main] init MuZero")
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

    # XXX: muax naively assumes env.spec.max_episode_steps is available
    #      => use gym.make(..., max_episode_steps=...)
    import gymnasium as gym
    import vcmi_gym
    vcmi_gym.register_envs()
    env = gym.make("VCMI-v12", max_episode_steps=100, **env_kwargs)

    from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper
    env = LegacyObservationSpaceWrapper(env)

    # TODO: `env` and test_env` MUST BE DIFFERENT
    #test_env = VcmiEnv(**env_kwargs)
    test_env = env

    # 7) Train with muax.fit
    print("[main] muax.fit(...)")
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
    print("[main] done")
