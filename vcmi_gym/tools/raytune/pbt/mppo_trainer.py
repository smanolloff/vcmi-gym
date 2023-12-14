from sb3_contrib import MaskablePPO
from .ppo_trainer import PPOTrainer

from .... import VcmiCNN

DEBUG = False


class MPPOTrainer(PPOTrainer):
    def __model_init(self, venv, **learner_kwargs):
        learner_kwargs = dict(
            learner_kwargs,
            policy="CnnPolicy",
            policy_kwargs=dict(
                features_extractor_class=VcmiCNN,
                features_extractor_kwargs=dict(features_dim=512)
            ),
        )

        return MaskablePPO(env=venv, **learner_kwargs)

    def __model_load(self, f, venv, **learner_kwargs):
        learner_kwargs = dict(
            learner_kwargs,
            policy="CnnPolicy",
            policy_kwargs=dict(
                features_extractor_class=VcmiCNN,
                features_extractor_kwargs=dict(features_dim=512)
            ),
        )

        return MaskablePPO.load(f, env=venv, **learner_kwargs)
