import sys
import os
from pathlib import Path
from datetime import datetime
import copy

# MUST come BEFORE importing ray
os.environ["RAY_DEDUP_LOGS"] = "0"

from ray import train, tune  # noqa: E402
from ray.tune.schedulers.pb2 import PB2  # noqa: E402

from config.raytune.ppo import config  # noqa: E402
from vcmi_gym.tools.raytune.ppo_trainer import PPOTrainer  # noqa: E402


def update_param_space(hyperparam_bounds, param_space):
    for key, value in hyperparam_bounds.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, list):
                    param_space[key][inner_key] = tune.uniform(inner_value[0], inner_value[1])
                else:
                    param_space[key][inner_key] = tune.uniform(inner_value, inner_value)
        else:
            if isinstance(value, list):
                param_space[key] = tune.uniform(value[0], value[1])
            else:
                tune.uniform(value, value)
    return param_space


#
# PB2 Example (from ray docs):
# https://docs.ray.io/en/latest/tune/examples/includes/pb2_example.html#pb2-example
#
def main():
    if len(sys.argv) != 2:
        raise Exception("Expected alg name as the only argument, eg. PPO")

    alg = sys.argv[1]

    if alg not in ["PPO"]:
        raise Exception("Only PPO is supported for raytune currently")

    orig_config = copy.deepcopy(config)

    # NOTE:
    # Apparently, tune also needs those defined as ranges in param_space
    # https://discuss.ray.io/t/pb2-seems-stuck-in-space-margins-and-raises-exceptions-with-lambdas/467/8
    update_param_space(config["hyperparam_bounds"], config["param_space"])

    mapname = Path(config["param_space"]["env_kwargs"]["mapname"]).stem
    assert mapname.isalnum()
    experiment_name = "%s-PPO-%s" % (mapname, datetime.now().strftime("%Y%m%d_%H%M%S"))

    results_dir = config["results_dir"]
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.getcwd(), results_dir)

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.pb2.PB2.html#ray-tune-schedulers-pb2-pb2
    pb2 = PB2(
        # XXX: if synch=True, time_attr must NOT be the default "time_total_s"
        # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/tune/schedulers/pb2.py#L316-L317
        time_attr="training_iteration",
        metric="rew_mean",
        mode="max",
        perturbation_interval=config["perturbation_interval"],
        hyperparam_bounds=config["hyperparam_bounds"],
        quantile_fraction=config["quantile_fraction"],
        log_config=False,  # used for reconstructing the config schedule
        require_attrs=True,
        synch=True,
    )

    checkpoint_config = train.CheckpointConfig(
        num_to_keep=10,
        checkpoint_score_order="max",
        checkpoint_score_attribute="rew_mean",
    )

    # https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray-train-runconfig
    run_config = train.RunConfig(
        name=experiment_name,
        verbose=False,
        failure_config=train.FailureConfig(fail_fast=True),
        checkpoint_config=checkpoint_config,
        stop={"rew_mean": 2000},
        callbacks=[],
        storage_path=results_dir,
    )

    tune_config = tune.TuneConfig(
        trial_name_creator=lambda t: t.trial_id,
        trial_dirname_creator=lambda t: t.trial_id,
        scheduler=pb2,
        reuse_actors=False,  # XXX: False is much safer and ensures no state leaks
        num_samples=config["population_size"]
    )

    trainer_initargs = {"config": orig_config, "experiment_name": experiment_name}

    tuner = tune.Tuner(
        tune.with_parameters(PPOTrainer, initargs=trainer_initargs),
        run_config=run_config,
        tune_config=tune_config,
        param_space=config["param_space"],
    )

    tuner.fit()


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    main()
