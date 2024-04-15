import ray.tune
from . import common
from .pbt_config import config


def main(alg, exp_name, resume_path):
    if resume_path:
        tuner = common.resume_tuner(alg, resume_path, config)
    else:
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
        pbt = ray.tune.schedulers.PopulationBasedTraining(
            time_attr=config["_raytune"]["time_attr"],
            metric="rew_mean",
            mode="max",
            perturbation_interval=config["_raytune"]["perturbation_interval"],
            hyperparam_mutations=config["_raytune"]["hyperparam_mutations"],
            quantile_fraction=config["_raytune"]["quantile_fraction"],
            log_config=False,  # used for reconstructing the config schedule
            require_attrs=True,
            synch=config["_raytune"]["synch"],
        )

        tuner = common.new_tuner(alg, exp_name, config, pbt)

    tuner.fit()
