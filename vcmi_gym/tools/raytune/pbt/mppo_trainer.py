from sb3_contrib import MaskablePPO
from .ppo_trainer import PPOTrainer

DEBUG = False


class MPPOTrainer(PPOTrainer):
    def _model_init(self, venv, **learner_kwargs):
        return MaskablePPO(env=venv, **learner_kwargs)

    def _model_load(self, f, venv, **learner_kwargs):
        return MaskablePPO.load(f, env=venv, **learner_kwargs)
