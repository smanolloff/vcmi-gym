import sys
import os
import re
from datetime import datetime
import copy
import importlib

# MUST come BEFORE importing ray
os.environ["RAY_DEDUP_LOGS"] = "0"

# this makes the "storage" arg redundant. By default, TUNE_RESULT_DIR
# is $HOME/ray_results and "storage" just *copies* everything into data
os.environ["TUNE_RESULT_DIR"] = os.path.join(os.path.dirname(__file__), "..", "data")

# chdir seems redundant. Initially tried to disable wandb's warning
# for requirements.txt, but it persists. However, it's better without
# changing dirs anyway
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

# this is to turn off tensorboard logger (side-effect turns off )
# os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

from ray import train, tune  # noqa: E402
from ray.tune.schedulers import PopulationBasedTraining  # noqa: E402
from .common import TBXDummyCallback  # noqa: E402

from .pbt_config import config  # noqa: E402


def main():
    assert len(sys.argv) == 3, "usage: python rl.raytune.pbt mppo <name>"
    alg = sys.argv[1]
    desc = sys.argv[2]
    assert alg in ["mppo"], "Only mppo is supported for PBT"
    assert re.match(r"^[0-9A-Za-z_-].+$", desc)
    experiment_name = "PBT-%s-%s" % (desc, datetime.now().strftime("%Y%m%d_%H%M%S"))
    trainable_mod = importlib.import_module("rl.raytune.trainable_%s" % alg)
    trainable_cls = getattr(trainable_mod, "Trainable%s" % alg.upper())
    orig_config = copy.deepcopy(config)

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="rew_mean",
        mode="max",
        perturbation_interval=config["perturbation_interval"],
        hyperparam_mutations=config["hyperparam_mutations"],
        quantile_fraction=config["quantile_fraction"],
        log_config=False,  # used for reconstructing the config schedule
        require_attrs=True,
        synch=True,
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
        failure_config=train.FailureConfig(max_failures=-1),
        checkpoint_config=checkpoint_config,
        stop={"rew_mean": config["target_ep_rew_mean"]},
        callbacks=[TBXDummyCallback()],
        # storage_path=results_dir,  # redundant, using TUNE_RESULT_DIR instead
    )

    tune_config = tune.TuneConfig(
        trial_name_creator=lambda t: t.trial_id,
        trial_dirname_creator=lambda t: t.trial_id,
        scheduler=pbt,
        reuse_actors=False,  # XXX: False is much safer and ensures no state leaks
        num_samples=config["population_size"],
    )

    initargs = {
        "config": orig_config,
        "experiment_name": experiment_name,
    }

    tuner = tune.Tuner(
        tune.with_parameters(trainable_cls, initargs=initargs),
        run_config=run_config,
        tune_config=tune_config,
        # NOT working - params are always randomly sampled for the 1st run?
        # param_space=initial_params
    )

    tuner.fit()


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    print("PID: %d" % os.getpid())
    main()
