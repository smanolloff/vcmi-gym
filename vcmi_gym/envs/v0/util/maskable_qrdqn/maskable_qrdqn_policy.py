from typing import Any, Dict, Optional

import torch as th
import numpy as np

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
