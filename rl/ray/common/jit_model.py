import torch


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
