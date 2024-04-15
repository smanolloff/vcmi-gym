import ray.tune
from . import common
from .pbt_config import config


def main(alg, exp_name, resume_path):
    if resume_path:
        tuner = common.resume_tuner(alg, resume_path, config)
    else:
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.pb2.PB2.html#ray-tune-schedulers-pb2-pb2
        pb2 = ray.tune.schedulers.pb2.PB2(
            time_attr="training_iteration",
            metric="rew_mean",
            mode="max",
            perturbation_interval=config["perturbation_interval"],
            hyperparam_bounds=config["hyperparam_bounds"],
            quantile_fraction=config["quantile_fraction"],
            log_config=False,  # used for reconstructing the config schedule
            require_attrs=True,
            synch=config["synch"],
        )

        tuner = common.new_tuner(alg, exp_name, config, pb2)

    tuner.fit()
