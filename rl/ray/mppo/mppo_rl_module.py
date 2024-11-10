import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from ..common import common_rl_module, common_encoder
from ..common.jit_model import JitModel

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

    #
    # JIT import
    #

    def jload(self, jagent_file, layer_mapping=None):
        common_rl_module.jload(self, jagent_file, layer_mapping)

    #
    # JIT export
    #

    def jsave(self, jagent_file, optimized=False):
        print("Saving JIT agent to %s" % jagent_file)
        shared = self.model_config["vf_share_layers"]
        clean_encoder, clean_pi, clean_vf = self._clean_clone()
        jmodel = torch.jit.script(JitModel(
            encoder_actor=clean_encoder.encoder,
            encoder_critic=None if shared else clean_encoder.critic_encoder,
            actor=self.pi,
            critic=self.vf,
            env_version=self.model_config["env_version"]
        ))

        if optimized:
            jmodel = optimize_for_mobile(jmodel, preserved_methods=["get_version", "predict", "get_value"])
            jmodel._save_for_lite_interpreter(jagent_file)
        else:
            torch.jit.save(jmodel, jagent_file)

    def _clean_clone(self):
        shared = self.model_config["vf_share_layers"]
        obs_dims = self.model_config["obs_dims"]
        netcfg = common_encoder.NetConfig(**self.model_config["network"])

        # 1. Init clean NNs

        clean_encoder = common_encoder.Common_Encoder(
            self.model_config["vf_share_layers"],
            self.config.inference_only,
            self.config.action_space,
            self.config.observation_space,
            obs_dims,
            netcfg
        )

        clean_pi = common_encoder.layer_init(common_encoder.build_layer(netcfg.actor, obs_dims), gain=0.01)
        clean_vf = common_encoder.layer_init(common_encoder.build_layer(netcfg.critic, obs_dims), gain=1.0)

        # 2. Load parameters

        clean_encoder.encoder.load_state_dict(self.encoder.encoder.state_dict(), strict=True)
        clean_pi.load_state_dict(self.pi.state_dict(), strict=True)
        clean_vf.load_state_dict(self.vf.state_dict(), strict=True)

        if not shared:
            clean_encoder.critic_encoder.load_state_dict(self.encoder.critic_encoder.state_dict(), strict=True)

        return clean_encoder, clean_pi, clean_vf
