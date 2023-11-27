import copy
from ..pb2.ppo_trainer import PPOTrainer as PB2_PPOTrainer, deepmerge, debuglog


class PPOTrainer(PB2_PPOTrainer):
    @debuglog
    def setup(self, cfg, initargs):
        new_cfg = deepmerge(copy.deepcopy(initargs["config"]["all_params"]), cfg)
        new_initargs = copy.deepcopy(initargs)
        new_initargs["config"]["hyperparam_bounds"] = initargs["config"]["hyperparam_mutations"]
        super().setup(new_cfg, new_initargs)
