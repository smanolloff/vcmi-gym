import tensorflow as tf

from gymnasium.spaces import Box, Discrete
from ray.rllib.algorithms.dreamerv3.tf.models.actor_network import ActorNetwork
from ray.rllib.algorithms.dreamerv3.utils import (
    get_gru_units,
    get_num_z_categoricals,
    get_num_z_classes,
)

# import gymnasium as gym
# from gymnasium.spaces import Box, Discrete
# import numpy as np

# from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
# from ray.rllib.algorithms.dreamerv3.utils import (
#     get_gru_units,
#     get_num_z_categoricals,
#     get_num_z_classes,
# )
# from ray.rllib.utils.framework import try_import_tf, try_import_tfp

# _, tf, _ = try_import_tf()
# tfp = try_import_tfp()

MASK_VALUE = tf.constant(tf.float32.min, dtype=tf.float32)


class MDreamerV3_ActorNetwork(ActorNetwork):
    def __init__(self, *args, **kwargs):
        self.__call_ORIG = self.call
        super().__init__(*args, **kwargs)

        # Trace self.call_with_masks.
        dl_type = tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        self.call_with_masks = tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, get_gru_units(self.model_size)], dtype=dl_type),
                tf.TensorSpec(
                    shape=[
                        None,
                        get_num_z_categoricals(self.model_size),
                        get_num_z_classes(self.model_size),
                    ],
                    dtype=dl_type,
                ),
                tf.TensorSpec(shape=[None, self.action_space.n], dtype=tf.bool),
            ]
        )(self.call_with_masks)

    # Identical to .call, but with action masking (just 2 LOC added)
    def call_with_masks(self, h, z, action_masks):
        """Performs a forward pass through this policy network.

        Args:
            h: The deterministic hidden state of the sequence model. [B, dim(h)].
            z: The stochastic discrete representations of the original
                observation input. [B, num_categoricals, num_classes].
        """
        # Flatten last two dims of z.
        assert len(z.shape) == 3
        z_shape = tf.shape(z)
        z = tf.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        out = tf.concat([h, z], axis=-1)
        out.set_shape(
            [
                None,
                (
                    get_num_z_categoricals(self.model_size)
                    * get_num_z_classes(self.model_size)
                    + get_gru_units(self.model_size)
                ),
            ]
        )
        # Send h-cat-z through MLP.
        action_logits = tf.cast(self.mlp(out), tf.float32)

        if isinstance(self.action_space, Discrete):
            action_probs = tf.nn.softmax(action_logits)

            # Add the unimix weighting (1% uniform) to the probs.
            # See [1]: "Unimix categoricals: We parameterize the categorical
            # distributions for the world model representations and dynamics, as well as
            # for the actor network, as mixtures of 1% uniform and 99% neural network
            # output to ensure a minimal amount of probability mass on every class and
            # thus keep log probabilities and KL divergences well behaved."
            action_probs = 0.99 * action_probs + 0.01 * (1.0 / self.action_space.n)

            # Danijar's code does: distr = [Distr class](logits=tf.log(probs)).
            # Not sure why we don't directly use the already available probs instead.
            action_logits = tf.math.log(action_probs)

            # XXX (simo): apply masking
            action_logits = tf.where(action_masks, action_logits, MASK_VALUE)
            action_probs = tf.nn.softmax(action_logits)

            # Distribution parameters are the log(probs) directly.
            distr_params = action_logits
            distr = self.get_action_dist_object(distr_params)

            action = tf.stop_gradient(distr.sample()) + (
                action_probs - tf.stop_gradient(action_probs)
            )

        elif isinstance(self.action_space, Box):
            # Send h-cat-z through MLP to compute stddev logits for Normal dist
            std_logits = tf.cast(self.std_mlp(out), tf.float32)
            # minstd, maxstd taken from [1] from configs.yaml
            minstd = 0.1
            maxstd = 1.0

            # Distribution parameters are the squashed std_logits and the tanh'd
            # mean logits.
            # squash std_logits from (-inf, inf) to (minstd, maxstd)
            std_logits = (maxstd - minstd) * tf.sigmoid(std_logits + 2.0) + minstd
            mean_logits = tf.tanh(action_logits)

            distr_params = tf.concat([mean_logits, std_logits], axis=-1)
            distr = self.get_action_dist_object(distr_params)

            action = distr.sample()

        return action, distr_params
