import copy
import ray.tune
from . import common
from .pbt_config import config


def main(alg, exp_name, resume_path):
    if resume_path:
        tuner = common.resume_tuner(alg, resume_path, config)
    else:
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.pb2.PB2.html#ray-tune-schedulers-pb2-pb2
        scheduler = ray.tune.schedulers.pb2.PB2(
            time_attr=config["_raytune"]["time_attr"],
            metric="rew_mean",
            mode="max",
            perturbation_interval=config["_raytune"]["perturbation_interval"],
            hyperparam_bounds=config["_raytune"]["hyperparam_bounds"],
            quantile_fraction=config["_raytune"]["quantile_fraction"],
            log_config=False,  # used for reconstructing the config schedule
            require_attrs=True,
            synch=config["_raytune"]["synch"],
        )

        initial_cfg = common.common_dict(copy.deepcopy(config["_raytune"]["hyperparam_bounds"]), config)
        searcher = ray.tune.search.BasicVariantGenerator(points_to_evaluate=[initial_cfg])
        tuner = common.new_tuner(alg, exp_name, config, scheduler, searcher)

    tuner.fit()
