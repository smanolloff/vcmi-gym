import torch
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from .mppo_encoder import MPPO_Encoder
from . import netutil

MASK_VALUE = torch.tensor(torch.finfo(torch.float32).min, dtype=torch.float32)


class MPPO_RLModule(PPOTorchRLModule):
    @override(PPOTorchRLModule)
    def get_train_action_dist_cls(self):
        return TorchCategorical

    @override(PPOTorchRLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(PPOTorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical

    @override(PPOTorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        print("*** _forward_inference() ***")
        # action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_inference(batch, **kwargs)
        self._mask_action_dist_inputs(outs, batch[Columns.OBS]["action_mask"])
        return outs

    @override(PPOTorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        print("*** _forward_exploration() ***")
        # action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_exploration(batch, **kwargs)
        self._mask_action_dist_inputs(outs, batch[Columns.OBS]["action_mask"])
        return outs

    @override(PPOTorchRLModule)
    def _forward_train(self, batch, **kwargs):
        print("*** _forward_train(): %s" % str(batch[Columns.OBS]["observation"].shape))
        import ipdb; ipdb.set_trace()  # noqa
        # traceback.print_stack()
        # action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_train(batch, **kwargs)
        self._mask_action_dist_inputs(outs, batch[Columns.OBS]["action_mask"])
        return outs

    # @override(PPOTorchRLModule)
    # def compute_values(self, batch):
    #     _, batch = self._preprocess_batch(batch)
    #     return super().compute_values(batch)

    # def _preprocess_batch(self, batch, **kwargs):
    #     action_mask = batch[Columns.OBS].pop("action_mask")
    #     batch[Columns.OBS] = batch[Columns.OBS].pop("observation")
    #     return action_mask, batch

    def _mask_action_dist_inputs(self, outs, mask):
        k = Columns.ACTION_DIST_INPUTS
        # outs[k] = outs[k].where(mask, MASK_VALUE)

        if not hasattr(self, "_cnt"):
            self._cnt = 0

        outs[k] = outs[k].where(torch.zeros_like(mask), MASK_VALUE)

        # outs[k] = outs[k].where(torch.zeros_like(mask), MASK_VALUE)
        # outs[k][0][self._cnt] = 0

        for i in range(outs[k].shape[0]):
            outs[k][i][self._cnt] = 0
            self._cnt += 1

    @override(PPOTorchRLModule)
    def setup(self):
        obs_dims = self.model_config["obs_dims"]
        netcfg = netutil.NetConfig(**self.model_config["network"])

        #
        # IMPORTANT:
        # 1. Do not rename these vars (encoder/pi/vf): they are needed upstream
        # 2. Do not rename "vf_share_layers" key: also needed upstream:
        #   https://github.com/ray-project/ray/blob/ray-2.37.0/rllib/algorithms/ppo/ppo_rl_module.py#L98
        #

        self.encoder = MPPO_Encoder(
            self.model_config["vf_share_layers"],
            self.config.inference_only,
            self.config.action_space,
            self.config.observation_space,
            obs_dims,
            netcfg
        )

        self.pi = netutil.layer_init(netutil.build_layer(netcfg.actor, obs_dims), gain=0.01)
        self.vf = netutil.layer_init(netutil.build_layer(netcfg.critic, obs_dims), gain=1.0)
