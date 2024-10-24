from ray.rllib.utils.annotations import override
from ray.rllib.algorithms import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from .mppo_rl_module import MPPO_RLModule
# from .mppo_learner import MPPO_Learner


class MPPO_Algorithm(PPO):
    @classmethod
    @override(PPO)
    def get_default_config(cls):
        return MPPO_Config()


class MPPO_Config(PPOConfig):
    def __init__(self):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=MPPO_Algorithm)

        # Activate new API stack
        self.enable_rl_module_and_learner = True
        self.enable_env_runner_and_connector_v2 = True

    # @override(PPOConfig)
    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MPPO_RLModule)

    # @override(PPOConfig)
    # def get_default_learner_class(self):
    #     return MPPO_Learner

    @property
    @override(PPOConfig)
    def _model_config_auto_includes(self):
        return {}
