from typing import Any, Dict

import gymnasium as gym
import numpy as np
import tensorflow as tf

from ray.rllib.utils import override
from ray.rllib.utils.numpy import one_hot
from ray.rllib.policy.eager_tf_policy import _convert_to_tf
from ray.rllib.core.columns import Columns

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule

from ray.rllib.algorithms.dreamerv3.utils import do_symlog_obs
from ray.rllib.algorithms.dreamerv3.tf.dreamerv3_tf_rl_module import DreamerV3TfRLModule
from ray.rllib.algorithms.dreamerv3.tf.models.actor_network import ActorNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.critic_network import CriticNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.world_model import WorldModel
from ray.rllib.algorithms.dreamerv3.tf.models.components.vector_decoder import VectorDecoder
from ray.rllib.models.tf.tf_distributions import TfCategorical

from .mdreamerv3_encoder import MDreamerV3_Encoder
from .mdreamerv3_dreamer_model import MDreamerV3_DreamerModel
from ..common import common_encoder


class MDreamerV3_RLModule(DreamerV3TfRLModule):
    # Contains modifications to work with VCMI's dict obs space:
    # the idea is that the dreamer model works with regular vector obs
    # (obs and mask are separated in the RL module and passed separately)
    #
    # Also, don't use "catalog" for creating encoder, decoder & dist_cls

    @override(DreamerV3TfRLModule)
    def setup(self):
        TfRLModule.setup(self)
        RLModule.setup(self)

        # Gather model-relevant settings.
        B = 1
        T = self.model_config["batch_length_T"]
        horizon_H = self.model_config["horizon_H"]
        gamma = self.model_config["gamma"]
        symlog_obs = do_symlog_obs(
            self.observation_space,
            self.model_config.get("symlog_obs", "auto"),
        )
        model_size = self.model_config["model_size"]

        if self.model_config["use_float16"]:
            tf.compat.v1.keras.layers.enable_v2_dtype_behavior()
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        self.encoder = MDreamerV3_Encoder(
            self.action_space,
            self.observation_space,
            self.model_config["obs_dims"],
            common_encoder.NetConfig(**self.model_config["network"])
        )

        self.decoder = VectorDecoder(model_size=model_size, observation_space=self.observation_space["observation"])

        # Build the world model (containing encoder and decoder).
        self.world_model = WorldModel(
            model_size=model_size,
            observation_space=self.observation_space["observation"],
            action_space=self.action_space,
            batch_length_T=T,
            encoder=self.encoder,
            decoder=self.decoder,
            symlog_obs=symlog_obs,
        )
        self.actor = ActorNetwork(
            action_space=self.action_space,
            model_size=model_size,
        )
        self.critic = CriticNetwork(
            model_size=model_size,
        )
        # Build the final dreamer model (containing the world model).
        self.dreamer_model = MDreamerV3_DreamerModel(
            model_size=self.model_config["model_size"],
            action_space=self.action_space,
            world_model=self.world_model,
            actor=self.actor,
            critic=self.critic,
            horizon=horizon_H,
            gamma=gamma,
        )

        self.action_dist_cls = TfCategorical

        # Perform a test `call()` to force building the dreamer model's variables.
        if self.framework == "tf2":
            test_obs = np.tile(
                np.expand_dims(self.observation_space["observation"].sample(), (0, 1)),
                reps=(B, T) + (1,) * len(self.observation_space["observation"].shape),
            )
            if isinstance(self.action_space, gym.spaces.Discrete):
                test_actions = np.tile(
                    np.expand_dims(
                        one_hot(
                            self.action_space.sample(),
                            depth=self.action_space.n,
                        ),
                        (0, 1),
                    ),
                    reps=(B, T, 1),
                )
            else:
                test_actions = np.tile(
                    np.expand_dims(self.action_space.sample(), (0, 1)),
                    reps=(B, T, 1),
                )

            results = self.dreamer_model(
                inputs=None,
                observations=_convert_to_tf(test_obs, dtype=tf.float32),
                actions=_convert_to_tf(test_actions, dtype=tf.float32),
                is_first=_convert_to_tf(np.ones((B, T)), dtype=tf.bool),
                start_is_terminated_BxT=_convert_to_tf(
                    np.zeros((B * T,)), dtype=tf.bool
                ),
                gamma=gamma,
            )

            # XXX: masks are used in forward_exploration/inference only,
            #       where there's no "T" dim: observations are (B, ...)
            #
            #       For inference/exploration, the call chain is:
            #           DreamerModel.forward_inference()    # obs is (B, ...)
            #           -> WorldModel.forward_inference ()  # res is (B, ...)
            #
            #       For train, Typically, the call chain is:
            #           DreamerModel.forward_train()    # obs is (B, T, ...)
            #           -> WorldModel.forward_train ()  # res contains (BxT, ...)
            #
            # Since DreamerModel's "test" call() method (called above) involves
            # a forward_train() call, it returns (BxT, ...) results.
            # We are using these results for call_with_masks()
            # => masks must also be BxT...
            test_masks = np.random.randint(0, 2, (B*T, self.action_space.n))

            self.dreamer_model.actor.call_with_masks(
                h=results["world_model_fwd"]["h_states_BxT"],
                z=results["world_model_fwd"]["z_posterior_states_BxT"],
                action_masks=_convert_to_tf(test_masks, dtype=tf.bool),
            )

        # Initialize the critic EMA net:
        self.critic.init_ema()

    @override(DreamerV3TfRLModule)
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Call the Dreamer-Model's forward_inference method and return a dict.
        actions, next_state = self.dreamer_model.forward_inference(
            observations=batch[Columns.OBS]["observation"],
            action_masks=batch[Columns.OBS]["action_mask"],
            previous_states=batch[Columns.STATE_IN],
            is_first=batch["is_first"],
        )
        return {Columns.ACTIONS: actions, Columns.STATE_OUT: next_state}

    @override(DreamerV3TfRLModule)
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Call the Dreamer-Model's forward_exploration method and return a dict.
        actions, next_state = self.dreamer_model.forward_exploration(
            observations=batch[Columns.OBS]["observation"],
            action_masks=batch[Columns.OBS]["action_mask"],
            previous_states=batch[Columns.STATE_IN],
            is_first=batch["is_first"],
        )
        return {Columns.ACTIONS: actions, Columns.STATE_OUT: next_state}

    @override(DreamerV3TfRLModule)
    def _forward_train(self, batch: Dict[str, Any]):
        # Call the Dreamer-Model's forward_train method and return its outputs as-is.
        return self.dreamer_model.forward_train(
            observations=batch[Columns.OBS]["observation"],
            # no masks here (actions already computed)
            actions=batch[Columns.ACTIONS],
            is_first=batch["is_first"],
        )
