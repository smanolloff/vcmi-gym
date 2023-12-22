from typing import Any, Dict, Optional, Tuple, Union

import torch as th
import numpy as np

from gymnasium import spaces
from .maskable_quantile_network import MaskableQuantileNetwork
from sb3_contrib.qrdqn.policies import QRDQNPolicy


class MaskableQRDQNPolicy(QRDQNPolicy):
    quantile_net: MaskableQuantileNetwork
    quantile_net_target: MaskableQuantileNetwork

    def make_quantile_net(self) -> MaskableQuantileNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskableQuantileNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True, action_masks: Optional[np.ndarray] = None) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic, action_masks=action_masks)

    def _predict(self, obs: th.Tensor, deterministic: bool = True, action_masks: Optional[np.ndarray] = None) -> th.Tensor:
        return self.quantile_net._predict(obs, deterministic=deterministic, action_masks=action_masks)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, action_masks=action_masks)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_quantiles=self.net_args["n_quantiles"],
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.quantile_net.set_training_mode(mode)
        self.training = mode
