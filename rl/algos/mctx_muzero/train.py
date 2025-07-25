import jax
import jax.numpy as jnp
from flax import linen as fnn
from flax.training import checkpoints
import optax
import muax
from muax import nn as muax_nn
from mctx import muzero_policy, RootFnOutput, RecurrentFnOutput

from .predictor import Predictor


# Needed for weights loading
import torch

from rl.world.p10n.jax import p10n
from rl.world.t10n.jax import t10n, reward, load_utils, obs_index
from rl.world.constants_v12 import (
    GLOBAL_ATTR_MAP,
    HEX_ATTR_MAP,
    DIM_OTHER,
    STATE_SIZE_ONE_HEX,
    N_HEX_ACTIONS,
)

from .model import MZModel

OBS_INDEX = obs_index.ObsIndex()


def get_action_mask(obs):
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

    predictor, params = Predictor.create_model(
        jit=False,  # very slow, maybe due to differing batch; or due to CPU
        max_transitions=5,
        side=(env_kwargs["role"] == "defender"),
        reward_dmg_mult=env_kwargs["reward_dmg_mult"],
        reward_term_mult=env_kwargs["reward_term_mult"],
    )

    mz_model = MZModel()
    mz_params = mz_model.init(rngs=rngs, obs=jnp.zeros([1, t10n.DIM_OBS]))

    def representation_fn(obs):
        # here we treat raw obs as the “latent” root
        return obs

    def prediction_fn(latent, obs):
        # produce policy logits & value
        logits, value = mz_model.apply(params["mz"], obs)
        mask = get_action_mask(obs)                      # shape (num_actions,)
        neg_inf = -1e9
        masked_logits = jnp.where(mask, logits, neg_inf)
        return masked_logits, value

    def dynamic_fn(params, latent, action, obs):
        next_obs = obs_model.apply(params["obs"], obs, action)
        reward = rew_model.apply(params["rew"], obs, action)
        # next_obs becomes the next latent state
        return next_obs, reward

    # 5) Bundle into a MuZero model
    gradient_transform = muax.model.optimizer(
        init_value=0.02, peak_value=0.02, end_value=0.002,
        warmup_steps=5_000, transition_steps=5_000
    )

    model = muax.MuZero(
        representation_fn,
        prediction_fn,
        dynamic_fn,
        policy='muzero',
        discount=0.99,
        optimizer=gradient_transform,
        support_size=10
    )

    # 6) Freeze your two pretrained sub‐trees so only the prediction head is trained
    def mask_fn(param_path, _):
        # param_path is a tuple like ('obs', ...), ('rew', ...), or ('pred', ...)
        # we return False (freeze) for 'obs' and 'rew'
        return param_path[0] not in ["obs", "rew", "act"]

















def network_factory(rng, input_shape):
    obs_model = t10n.FlaxTransitionModel(deterministic=True)
    obs_params = obs_model.init(rngs=rng, obs=jnp.zeros([1, t10n.DIM_OBS]), action=jnp.array([0]))

    rew_model = reward.FlaxTransitionModel(deterministic=True)
    rew_params = rew_model.init(rngs=rng, obs=jnp.zeros([1, t10n.DIM_OBS]), action=jnp.array([0]))

    # LOAD
    obs_torch_state = torch.load("hauzybxn-model.pt", weights_only=True, map_location="cpu")
    rew_torch_state = torch.load("aexhrgez-model.pt", weights_only=True, map_location="cpu")
    obs_params = load_utils.load_params_from_torch_state(obs_params, obs_torch_state, head_names=["global", "player", "hex"])
    rew_params = load_utils.load_params_from_torch_state(rew_params, rew_torch_state, head_names=["reward"])

    net = MyMuZeroNetwork(obs_model=obs_model, rew_model=rew_model)
    params = {"obs": obs_params, "rew": rew_params}

    return net, params




class MyMuZeroNetwork(FlaxNetwork):
    obs_model: jnn.Module
    rew_model: jnn.Module

    def representation(self, _params, obs):
        """Map observation to a latent state."""
        # NOTE: no encoder => identity
        return obs

    def dynamics(self, params, latent, action):
        """Given a latent state, predict a reward and the next latent state."""
        # NOTE: no encoder => latent=obs
        obs_pred = self.obs_model.apply(params['obs'], latent, action)
        rew_pred = self.reward_model.apply(params['rew'], latent, action)

        return obs_pred, rew_pred

    def prediction(self, params, latent):
        """Apply an action mask to super()'s predicted action logits."""
        # NOTE: no encoder => latent=obs
        action_mask = get_action_mask(latent)
        logits, value = super().prediction(params, latent)
        logits_masked = jnp.where(action_mask, logits, -1e9)
        return logits_masked, value


# Freeze both pretrained sub-trees in your optimizer so MuZero won’t overwrite them:
def model_params_mask_fn(param_path, _):
    # param_path is a tuple like ('obs', '...') or ('reward','...')
    return param_path[0] not in ("obs", "rew")


if __name__ == "__main__":
    optimizer = optax.masked(optax.adam(1e-3), model_params_mask_fn)
    config = MuZeroConfig(
        network_factory=network_factory,
        optimizer=optimizer,
        action_mask_fn=lambda root: get_action_mask(root.obs),
    )

    trainer = Trainer(config)
    trainer.train(...)
