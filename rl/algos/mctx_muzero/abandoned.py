from mctx import muzero_policy, RootFnOutput, RecurrentFnOutput

# om mctx.networks import FlaxNetwork
import jax.numpy as jnp
import jax.nn as jnn
import optax

# Needed for weights loading
import torch

from rl.world.t10n.jax import t10n, reward, load_utils, obs_index
from rl.world.constants_v12 import (
    GLOBAL_ATTR_MAP,
    HEX_ATTR_MAP,
    DIM_OTHER,
    STATE_SIZE_ONE_HEX,
    N_HEX_ACTIONS,
)

OBS_INDEX = obs_index.ObsIndex()


if __name__ == "__main__":
    # 1. raw obs → (prior_logits, value, embedding)
    def root_fn(params, obs):
        # e.g. policy_value_head.apply returns (logits, value)
        logits, value = policy_value_head.apply(params['pv'], obs)
        embed          = representation_net.apply(params['rep'], obs)
        return RootFnOutput(
          prior_logits=logits,
          value=value,
          embedding=embed,
        )

















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
