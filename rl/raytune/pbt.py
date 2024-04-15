import copy
import ray.tune
from . import common
from .pbt_config import config


def convert_to_param_space(mutations):
    res = {}
    for k, v in mutations.items():
        if isinstance(v, dict):
            res[k] = convert_to_param_space(v)
        elif isinstance(v, list):
            res[k] = ray.tune.choice(v)
        else:
            assert isinstance(v, ray.tune.search.sample.Domain)
    return res


def main(alg, exp_name, resume_path):
    if resume_path:
        tuner = common.resume_tuner(alg, resume_path, config)
    else:
        mutations = config["_raytune"]["hyperparam_mutations"]

        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
        scheduler = ray.tune.schedulers.PopulationBasedTraining(
            time_attr=config["_raytune"]["time_attr"],
            metric="rew_mean",
            mode="max",
            perturbation_interval=config["_raytune"]["perturbation_interval"],
            hyperparam_mutations=mutations,
            quantile_fraction=config["_raytune"]["quantile_fraction"],
            log_config=False,  # used for reconstructing the config schedule
            require_attrs=True,
            synch=config["_raytune"]["synch"],
        )

        # XXX: "initial" values have a lot of caveats, see note in pbt_config.py
        if config["_raytune"]["initial_hyperparams"]:
            initial_cfg = common.common_dict(copy.deepcopy(mutations), config["_raytune"]["initial_hyperparams"])
            searcher = ray.tune.search.BasicVariantGenerator(points_to_evaluate=[initial_cfg])
            param_space = convert_to_param_space(mutations)
            tuner = common.new_tuner(alg, exp_name, config, scheduler, searcher, param_space)
        else:
            tuner = common.new_tuner(alg, exp_name, config, scheduler)

    tuner.fit()
