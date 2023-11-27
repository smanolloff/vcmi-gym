from sb3_contrib import RecurrentPPO
from .ppo_trainer import PPOTrainer

DEBUG = False


class RPPOTrainer(PPOTrainer):
    def _model_init(self, venv, **learner_kwargs):
        return RecurrentPPO(env=venv, **learner_kwargs)

    def _model_load(self, f, venv, **learner_kwargs):
        return RecurrentPPO.load(f, env=venv, **learner_kwargs)
