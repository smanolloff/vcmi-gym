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
                net_arch=[],
                features_extractor_class=VcmiCNN,
                features_extractor_kwargs=dict(features_dim=1024)
            ),
        )

        return MaskablePPO(env=venv, **learner_kwargs)

    def _model_internal_load(self, f, venv, **learner_kwargs):
        learner_kwargs = dict(
            learner_kwargs,
            policy="CnnPolicy",
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=VcmiCNN,
                features_extractor_kwargs=dict(features_dim=1024)
            ),
        )

        return MaskablePPO.load(f, env=venv, **learner_kwargs)
