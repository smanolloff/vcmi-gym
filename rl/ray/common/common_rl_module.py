import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import CRITIC, ENCODER_OUT

from . import common_encoder
from . import util

MASK_VALUE = torch.tensor(torch.finfo(torch.float32).min, dtype=torch.float32)


def mask_action_dist_inputs(outs, mask):
    k = Columns.ACTION_DIST_INPUTS
    outs[k] = outs[k].where(mask, MASK_VALUE)


# Overrides PPOTorchRLModule to call self.encoder.compute_values()
# instead of self.encoder.critic_encoder.forward()
def compute_values(rl_module, batch, embeddings=None):
    if embeddings is None:
        # Separate vf-encoder.
        if hasattr(rl_module.encoder, "critic_encoder"):
            batch_ = batch
            if rl_module.is_stateful():
                # The recurrent encoders expect a `(state_in, h)`  key in the
                # input dict while the key returned is `(state_in, critic, h)`.
                batch_ = batch.copy()
                batch_[Columns.STATE_IN] = batch[Columns.STATE_IN][CRITIC]
            embeddings = rl_module.encoder.compute_value(batch_)
        # Shared encoder.
        else:
            embeddings = rl_module.encoder(batch)[ENCODER_OUT][CRITIC]

    # Value head.
    vf_out = rl_module.vf(embeddings)
    # Squeeze out last dimension (single node value head).
    return vf_out.squeeze(-1)

#
# JIT import
#

def jload(rl_module, jagent_file, layer_mapping):
    jmodel = torch.jit.load(jagent_file)

    if not layer_mapping:
        # rlmodule <=> jmodel layer
        layer_mapping = {
            "encoder.encoder": "encoder_actor",
            "pi": "actor",
            "vf": "critic",
        }

    for rl_attr, model_attr in layer_mapping.items():
        rlmodule_layer = util.get_nested_attr(rl_module, rl_attr)
        jmodel_layer = util.get_nested_attr(jmodel, model_attr)
        rlmodule_layer.load_state_dict(jmodel_layer.state_dict(), strict=True)

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


class JitModel(torch.nn.Module):
    """ TorchScript version of Model """

    def __init__(
        self,
        encoder_actor: torch.nn.Module,
        encoder_critic: torch.nn.Module,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        env_version: int,
    ):
        super().__init__()
        self.encoder_actor = encoder_actor
        self.encoder_critic = encoder_critic
        self.actor = actor
        self.critic = critic
        self.env_version = env_version

    @torch.jit.export
    def predict(self, obs, mask, deterministic: bool = False) -> int:
        b_obs = obs.unsqueeze(dim=0)
        b_mask = mask.unsqueeze(dim=0)
        latent = self.encoder_actor(b_obs)
        action_logits = self.actor(latent)
        probs = self.categorical_masked(action_logits, b_mask)
        action = torch.argmax(probs, dim=1) if deterministic else self.sample(probs, action_logits)
        return action.int().item()

    @torch.jit.export
    def forward(self, obs) -> torch.Tensor:
        b_obs = obs.unsqueeze(dim=0)
        latent = self.encoder_actor(b_obs)
        return self.actor(latent)

    @torch.jit.export
    def get_value(self, obs) -> float:
        b_obs = obs.unsqueeze(dim=0)
        if self.encoder_critic is None:
            latent = self.encoder_actor(b_obs)
        else:
            latent = self.encoder_critic(b_obs)
        value = self.critic(latent)
        return value.float().item()

    @torch.jit.export
    def get_version(self) -> int:
        return self.env_version

    # Implement SerializableCategoricalMasked as a function
    # (lite interpreter does not support instantiating the class)
    @torch.jit.export
    def categorical_masked(self, logits0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_value = torch.tensor(-((2 - 2**-23) * 2**127), dtype=logits0.dtype)
        logits1 = torch.where(mask, logits0, mask_value)
        logits = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    @torch.jit.export
    def sample(self, probs: torch.Tensor, action_logits: torch.Tensor) -> torch.Tensor:
        num_events = action_logits.size()[-1]
        probs_2d = probs.reshape(-1, num_events)
        samples_2d = torch.multinomial(probs_2d, 1, True).T
        batch_shape = action_logits.size()[:-1]
        return samples_2d.reshape(batch_shape)
