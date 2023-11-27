import copy
from ..pb2.ppo_trainer import PPOTrainer as PB2_PPOTrainer, debuglog
from ..wandb_init import wandb_init


def deepmerge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                a[key] = b[key]
        else:
            raise Exception("Key not found: %s" % key)
    return a


class PPOTrainer(PB2_PPOTrainer):
    def _wandb_init(self, experiment_name, config):
        wandb_init("PBT", self.trial_id, self.trial_name, experiment_name, config)

    @debuglog
    def setup(self, cfg, initargs):
        new_cfg = deepmerge(copy.deepcopy(initargs["config"]["all_params"]), cfg)
        new_initargs = copy.deepcopy(initargs)
        new_initargs["config"]["hyperparam_bounds"] = initargs["config"]["hyperparam_mutations"]
        super().setup(new_cfg, new_initargs)
