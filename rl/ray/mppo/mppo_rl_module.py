import torch

from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from ..common import common_rl_module, common_encoder

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
        outs = super()._forward_inference(batch, **kwargs)
        common_rl_module.mask_action_dist_inputs(outs, batch[Columns.OBS]["action_mask"])
        return outs

    @override(PPOTorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        outs = super()._forward_exploration(batch, **kwargs)
        common_rl_module.mask_action_dist_inputs(outs, batch[Columns.OBS]["action_mask"])
        return outs

    @override(PPOTorchRLModule)
    def _forward_train(self, batch, **kwargs):
        outs = super()._forward_train(batch, **kwargs)
        common_rl_module.mask_action_dist_inputs(outs, batch[Columns.OBS]["action_mask"])
        return outs

    @override(PPOTorchRLModule)
    def compute_values(self, batch, embeddings=None):
        return common_rl_module.compute_values(self, batch, embeddings)

    @override(PPOTorchRLModule)
    def setup(self):
        obs_dims = self.model_config["obs_dims"]
        netcfg = common_encoder.NetConfig(**self.model_config["network"])

        #
        # IMPORTANT:
        # 1. Do not rename these vars (encoder/pi/vf): needed by PPOTorchRLModule
        # 2. Do not rename "vf_share_layers" key: needed by PPORLModule
        #

        self.encoder = common_encoder.Common_Encoder(
            self.model_config["vf_share_layers"],
            self.config.inference_only,
            self.config.action_space,
            self.config.observation_space,
            obs_dims,
            netcfg
        )

        self.pi = common_encoder.layer_init(common_encoder.build_layer(netcfg.actor, obs_dims), gain=0.01)
        self.vf = common_encoder.layer_init(common_encoder.build_layer(netcfg.critic, obs_dims), gain=1.0)

    def jload(self, jagent_file, layer_mapping=None):
        common_rl_module.jload(self, jagent_file, layer_mapping)

    def jsave(self, jagent_file, optimized=False):
        common_rl_module.jsave(self, jagent_file)
