import torch.optim
from sb3_contrib import MaskablePPO
from .ppo_trainer import PPOTrainer

from .... import VcmiCNN

DEBUG = True


class MPPOTrainer(PPOTrainer):
    def _model_internal_init(self, venv, **learner_kwargs):
        learner_kwargs = dict(
            learner_kwargs,
            policy="CnnPolicy",
            policy_kwargs=dict(
                net_arch=[64, 128, 64],
                features_extractor_class=VcmiCNN,
                features_extractor_kwargs=dict(features_dim=1024),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(weight_decay=0.01)
            ),
        )

        return MaskablePPO(env=venv, **learner_kwargs)

    def _model_internal_load(self, f, venv, **learner_kwargs):
        learner_kwargs = dict(
            learner_kwargs,
            policy="CnnPolicy",
            policy_kwargs=dict(
                net_arch=[64, 128, 64],
                features_extractor_class=VcmiCNN,
                features_extractor_kwargs=dict(features_dim=1024),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(weight_decay=0.01)
            ),
        )

        return MaskablePPO.load(f, env=venv, **learner_kwargs)
