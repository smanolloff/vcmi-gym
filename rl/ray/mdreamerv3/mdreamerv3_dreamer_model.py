import tensorflow as tf
from ray.rllib.utils import override
from ray.rllib.algorithms.dreamerv3.tf.models.dreamer_model import DreamerModel

from .mdreamerv3_actor_network import MDreamerV3_ActorNetwork


class MDreamerV3_DreamerModel(DreamerModel):
    # Contains modifications to work with action masks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = MDreamerV3_ActorNetwork(
            model_size=self.actor.model_size,
            action_space=self.actor.action_space
        )

    @tf.function
    def forward_inference(self, observations, action_masks, previous_states, is_first, training=None):
        """Performs a (non-exploring) action computation step given obs and states.

        Note that all input data should not have a time rank (only a batch dimension).

        Args:
            observations: The current environment observation with shape (B, ...).
            previous_states: Dict with keys `a`, `h`, and `z` used as input to the RSSM
                to produce the next h-state, from which then to compute the action
                using the actor network. All values in the dict should have shape
                (B, ...) (no time rank).
            is_first: Batch of is_first flags. These should be True if a new episode
                has been started at the current timestep (meaning `observations` is the
                reset observation from the environment).
        """
        # Perform one step in the world model (starting from `previous_state` and
        # using the observations to yield a current (posterior) state).
        states = self.world_model.forward_inference(
            observations=observations,
            previous_states=previous_states,
            is_first=is_first,
        )
        # Compute action using our actor network and the current states.
        _, distr_params = self.actor.call_with_masks(h=states["h"], z=states["z"], action_masks=action_masks)
        # Use the mode of the distribution (Discrete=argmax, Normal=mean).
        distr = self.actor.get_action_dist_object(distr_params)
        actions = distr.mode()
        return actions, {"h": states["h"], "z": states["z"], "a": actions}

    @tf.function
    def forward_exploration(
        self, observations, action_masks, previous_states, is_first, training=None
    ):
        """Performs an exploratory action computation step given obs and states.

        Note that all input data should not have a time rank (only a batch dimension).

        Args:
            observations: The current environment observation with shape (B, ...).
            previous_states: Dict with keys `a`, `h`, and `z` used as input to the RSSM
                to produce the next h-state, from which then to compute the action
                using the actor network. All values in the dict should have shape
                (B, ...) (no time rank).
            is_first: Batch of is_first flags. These should be True if a new episode
                has been started at the current timestep (meaning `observations` is the
                reset observation from the environment).
        """
        # Perform one step in the world model (starting from `previous_state` and
        # using the observations to yield a current (posterior) state).
        states = self.world_model.forward_inference(
            observations=observations,
            previous_states=previous_states,
            is_first=is_first,
        )
        # Compute action using our actor network and the current states.
        actions, _ = self.actor.call_with_masks(h=states["h"], z=states["z"], action_masks=action_masks)
        return actions, {"h": states["h"], "z": states["z"], "a": actions}
