import sys
import os
from pathlib import Path
from datetime import datetime
import copy
import importlib

# MUST come BEFORE importing ray
os.environ["RAY_DEDUP_LOGS"] = "0"

# this makes the "storage" arg redundant. By default, TUNE_RESULT_DIR
# is $HOME/ray_results and "storage" just *copies* everything into data...
os.environ["TUNE_RESULT_DIR"] = os.path.join(os.path.dirname(__file__), "data")

# chdir seems redundant. Initially tried to disable wandb's warning
# for requirements.txt, but it persists. However, it's better without
# changing dirs anyway
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

# this is to turn off tensorboard logger (side-effect turns off )
# os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

from ray import train, tune  # noqa: E402
from ray.tune.schedulers.pb2 import PB2  # noqa: E402

from vcmi_gym.tools.raytune.tbx_dummy_callback import TBXDummyCallback  # noqa: E402


def convert_to_param_space(hyperparam_bounds):
    for key, value in hyperparam_bounds.items():
        if isinstance(value, dict):
            convert_to_param_space(value)
        else:
            assert isinstance(value, list) and len(value) == 2
            vmin, vmax = value
            hyperparam_bounds[key] = tune.uniform(vmin, vmax)


#
# PB2 Example (from ray docs):
# https://docs.ray.io/en/latest/tune/examples/includes/pb2_example.html#pb2-example
#
def main():
    if len(sys.argv) != 2:
        raise Exception("Expected alg name as the only argument, eg. PPO")

    alg = sys.argv[1]

    if alg not in ["PPO", "RPPO"]:
        raise Exception("Only PPO, RPPO is supported for raytune currently")

    config_mod = importlib.import_module("config.raytune.pb2.%s" % alg.lower())
    trainer_mod = importlib.import_module("vcmi_gym.tools.raytune.pb2.%s_trainer" % alg.lower())

    config = config_mod.config
    trainer_cls = getattr(trainer_mod, "%sTrainer" % alg)

    orig_config = copy.deepcopy(config)

    mapname = Path(config["all_params"]["env_kwargs"]["mapname"]).stem
    assert mapname.isalnum()
    experiment_name = "%s-%s-%s" % (
        mapname,
        alg,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

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

        # TODO: this could resolve the issue with min/max param swings
        # custom_explore_fn
    )

    checkpoint_config = train.CheckpointConfig(
        num_to_keep=10,
        # XXX: can't use score as it may delete the *latest* checkpoint
        #      and then fail when attempting to load it after perturb...
        # checkpoint_score_order="max",
        # checkpoint_score_attribute="rew_mean",
    )

    # https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray-train-runconfig
    run_config = train.RunConfig(
        name=experiment_name,
        verbose=False,
        failure_config=train.FailureConfig(fail_fast=True),
        checkpoint_config=checkpoint_config,
        stop={"rew_mean": 2000},
        callbacks=[TBXDummyCallback()],
        # storage_path=results_dir,  # redundant, using TUNE_RESULT_DIR instead
    )

    tune_config = tune.TuneConfig(
        trial_name_creator=lambda t: t.trial_id,
        trial_dirname_creator=lambda t: t.trial_id,
        scheduler=pb2,
        reuse_actors=False,  # XXX: False is much safer and ensures no state leaks
        num_samples=config["population_size"],
    )

    trainer_initargs = {
        "config": orig_config,
        "experiment_name": experiment_name,
        "root_dir": os.getcwd(),
    }

    tuner = tune.Tuner(
        tune.with_parameters(trainer_cls, initargs=trainer_initargs),
        run_config=run_config,
        tune_config=tune_config,
        param_space=convert_to_param_space(copy.deepcopy(config["hyperparam_bounds"])),
    )

    tuner.fit()


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    main()
