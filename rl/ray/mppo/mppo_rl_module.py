import torch
import gymnasium as gym

from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from .mppo_encoder import MPPO_Encoder
from . import netutil


class MPPO_RLModule(PPOTorchRLModule):
    # Example from:
    # https://github.com/ray-project/ray/blob/ray-2.37.0/rllib/examples/rl_modules/classes/action_masking_rlm.py#L16

    @override(PPOTorchRLModule)
    def __init__(self, config: RLModuleConfig):
        if not isinstance(config.observation_space, gym.spaces.Dict):
            raise ValueError(
                "This RLModule requires the environment to provide a "
                "`gym.spaces.Dict` observation space of the form: \n"
                " {'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                "  'observation': <real observation space>}"
            )

        self.mask_value = torch.tensor(torch.finfo(torch.float32).min, dtype=torch.float32)

        # XXX: In the Masked RLModule example, `config.observation_space`
        #       is mutated to ensure only the observation is exposed upstream
        #       (i.e. without the mask), then this change is REVERTED in setup().
        #       Turns out this is only needed if I use catalog, which I dont:
        #
        # Here's how RLModule.__init__() looks like:
        #
        # PPOTorchRLModule .init(): (n/a)
        # PPORLModule      .init(): (n/a)
        # TorchRLModule    .init(): (calls nn.Module.init(), RLModule.init(), deletes non-inference attrs)
        # RLModule         .init(): calls self.setup()
        #
        # PPOTorchRLModule .setup(): (n/a)
        # PPORLModule      .setup(): !!! (builds NNs: self.vf, self.pi via self.catalog)
        # TorchRLModule    .setup(): (n/a)
        # RLModule         .setup(): no-op
        #
        # Only self.catalog seems to need observation_space (for building NNs)
        # self.catalog is created by RLModule.get_catalog(), which is not called
        # anywhere (maybe C++ calls it?)
        #
        # The docs advise to avoid catalogs unless necessary:
        #     https://docs.ray.io/en/releases-2.37.0/rllib/rllib-rlmodule.html#extending-existing-rllib-rl-modules
        #
        # I don't use catalog here => no need to fake the observation_space
        #
        # config.observation_space = config.observation_space["observation"]

        super().__init__(config)

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
        # action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_inference(batch, **kwargs)
        self._mask_action_dist_inputs(outs, batch[Columns.OBS]["action_mask"])
        return outs

    @override(PPOTorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        # action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_exploration(batch, **kwargs)
        self._mask_action_dist_inputs(outs, batch[Columns.OBS]["action_mask"])
        return outs

    @override(PPOTorchRLModule)
    def _forward_train(self, batch, **kwargs):
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
        outs[k] = outs[k].where(mask, self.mask_value)

    @override(PPOTorchRLModule)
    def setup(self):
        obs_dims = self.config.model_config_dict["obs_dims"]
        netcfg = netutil.NetConfig(**self.config.model_config_dict["network"])

        #
        # IMPORTANT:
        # 1. Do not rename these vars (encoder/pi/vf): they are needed upstream
        # 2. Do not rename "vf_share_layers" key: also needed upstream:
        #   https://github.com/ray-project/ray/blob/ray-2.37.0/rllib/algorithms/ppo/ppo_rl_module.py#L98
        #

        self.encoder = MPPO_Encoder(
            self.config.model_config_dict["vf_share_layers"],
            self.config.inference_only,
            self.config.action_space,
            self.config.observation_space,
            obs_dims,
            netcfg
        )

        self.pi = netutil.layer_init(netutil.build_layer(netcfg.actor, obs_dims), gain=0.01)
        self.vf = netutil.layer_init(netutil.build_layer(netcfg.critic, obs_dims), gain=1.0)
