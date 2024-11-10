# These env vars must be set *before* importing ray/wandb modules
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["WANDB_SILENT"] = "true"

import ast
import argparse
import ray.tune
import vcmi_gym
import datetime
import pygit2
import json
import wandb
import tempfile
import copy
import multiprocessing

from ray.rllib.utils import deep_update

from . import MPPO_Algorithm, MPPO_Config
from ..common.common_config import ENV_TUNE_ID, ENV_VERSION_CHECK_KEY
from ..common import util


# Silence here in order to supress messages from local runners
# Silence once more in algo init to supress messages from remote runners
util.silence_log_noise()


def override_config_value(cfg, path, value):
    keys = path.split('.')
    d = cfg

    # Traverse the dict to find the position of the final key
    for key in keys[:-1]:
        # A typo when "updating" a key leads to an invalid key being saved
        # into the wandb run config which is hard to recover from
        assert key in d, "Key '%s' not found in current config"
        if key not in d:
            d[key] = {}
        d = d[key]

    old_value = d.get(keys[-1])
    new_value = ast.literal_eval(value)
    action = "No change for" if old_value == new_value else "Overwrite"
    print("%s %s: %s -> %s" % (action, path, old_value, value))
    d[keys[-1]] = new_value
    return old_value, new_value


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


def make_env_creator(env_gym_id):
    env_cls = util.get_env_cls(env_gym_id)

    # Even if there are remote runners, ray still needs the local runners
    # It doesn't really use them as envs, though => use dummy ones
    class DummyEnv(env_cls):
        def __init__(self, *args, **kwargs):
            self.action_space = env_cls.ACTION_SPACE
            self.observation_space = env_cls.OBSERVATION_SPACE
            pass

        def step(self, *args, **kwargs):
            raise Exception("step() called on DummyEnv")

        def reset(self, *args, **kwargs):
            raise Exception("reset() called on DummyEnv")

        def render(self, *args, **kwargs):
            raise Exception("render() called on DummyEnv")

        def close(self, *args, **kwargs):
            pass

    def env_creator(cfg):
        if cfg.num_workers > 0 and cfg.worker_index == 0:
            return DummyEnv()
        else:
            print(f"Env kwargs: {json.dumps(cfg)}")
            env_kwargs = copy.deepcopy(cfg)
            assert env_cls.ENV_VERSION == env_kwargs.pop(ENV_VERSION_CHECK_KEY)
            return env_cls(**env_kwargs)

    return env_creator


def load_ray_checkpoint(path):
    # Same as "load_ray_checkout"
    with open(util.to_abspath(f"{path}/mppo_config.json"), "r") as f:
        old_algo_config = json.load(f)

    # No sense in loading from file after loading from checkpoint
    old_algo_config["_master_config"]["user"]["model_load_file"] = ""
    return old_algo_config


def load_wandb_artifact(artifact):
    tempdir = tempfile.TemporaryDirectory()
    artifact.download(tempdir)
    old_algo_config = load_ray_checkpoint(str(tempdir))
    return old_algo_config, tempdir


def new_tuner(opts):
    # All cases below will result in a new W&B Run
    # The config used depends on the init method
    old_algo_config = None
    master_config = None
    checkpoint_load_dir = None

    assert not opts.env_gym_id, "env_gym_id is needed only when resuming an experiment"

    match opts.init_method:
        # Same as "load_wandb_artifact" but for run's latest artifact
        case "load_wandb_run":
            print("Loading W&B run %s ..." % opts.init_argument)
            assert len(opts.init_argument.split("/")) == 3, "wandb run format: s-manolloff/<project>/<run_id>"
            run = wandb.Api().run(opts.init_argument)
            artifact = next(a for a in reversed(run.logged_artifacts()) if a.type == "model")
            old_algo_config, checkpoint_load_dir = load_wandb_artifact(artifact)

        case "load_wandb_artifact":
            print("Loading W&B artifact %s ..." % opts.init_argument)
            artifact = wandb.Api().artifact(opts.init_argument)
            old_algo_config, checkpoint_load_dir = load_wandb_artifact(artifact)

        case "load_ray_checkpoint":
            print("Loading ray checkpoint %s ..." % opts.init_argument)
            checkpoint_load_dir = opts.init_argument
            old_algo_config = load_ray_checkpoint(checkpoint_load_dir)

        case "load_json_config":
            print("Loading JSON config %s ..." % opts.init_argument)
            with open(util.to_abspath(opts.init_argument), "r") as f:
                old_algo_config = json.load(f)

        case "load_python_config":
            import importlib
            print("Loading python module %s ..." % opts.init_argument)
            mod = importlib.import_module(opts.init_argument, package=__package__)
            master_config = mod.load()

        case _:
            raise Exception(f"Unknown init_argument: {opts.init_argument}")

    if old_algo_config:
        # Extract master_config from old_algo_config dict
        assert not master_config
        master_config = old_algo_config["_master_config"]
    else:
        assert master_config

    master_overrides = {}

    # config_overrides is a list of "path.to.key=value"
    for co in (opts.master_overrides or []):
        name, value = co.split("=")
        oldvalue, newvalue = override_config_value(master_config, name, value)
        if oldvalue != newvalue:
            master_overrides[name] = [oldvalue, newvalue]

    git = pygit2.Repository(util.vcmigym_root_path)
    now = datetime.datetime.now()

    init_info = dict(
        checkpoint_load_dir=str(checkpoint_load_dir or ""),
        experiment_name=opts.experiment_name.format(datetime=now.strftime("%Y%m%d_%H%M%S")),
        git_head=str(git.head.target),
        git_is_dirty=any(git.diff().patch),
        master_overrides=master_overrides,
        init_argument=opts.init_argument,
        init_method=opts.init_method,
        timestamp=now.astimezone().isoformat(),
        wandb_project=opts.wandb_project,  # "" disables wandb
    )

    util.deepmerge(master_config["user"], init_info, in_place=True, allow_new=False, update_existing=True)

    env_gym_id = master_config["user"]["env_gym_id"]
    ray.tune.registry.register_env(ENV_TUNE_ID, make_env_creator(env_gym_id))

    # Extract hyperparam_values from old_algo_config dict:
    # NOTE: Hyperparams match the AlgorithmConfig *variables*
    #       (not the master_config keys)
    if old_algo_config:
        master_config["user"]["hyperparam_values"] = util.common_dict(
            master_config["user"]["hyperparam_mutations"],
            old_algo_config,
            strict=True
        )

    algo_config = MPPO_Config().master_config(master_config)

    if (
        algo_config.evaluation_num_env_runners == 0
        and algo_config.num_env_runners == 0
        and algo_config.env_config["conntype"] == "thread"
        and algo_config.evaluation_config["env_config"]["conntype"] == "thread"
    ):
        raise Exception("Both 'train' and 'eval' runners are local -- at least one must have conntype='proc'")

    checkpoint_config = ray.train.CheckpointConfig(num_to_keep=3)
    run_config = ray.train.RunConfig(
        name=algo_config.user_config.experiment_name,
        verbose=False,
        failure_config=ray.train.FailureConfig(max_failures=-1),
        checkpoint_config=checkpoint_config,
        # stop={"rew_mean": pbt_config["_raytune"]["target_ep_rew_mean"]},
        # callbacks=[TBXDummyCallback()],
        storage_path=str(util.data_path)
    )

    # Hyperparams are used twice here:
    # 1. In PBT's `hyperparam_mutations` (as lists of values) - used
    #    for perturbations.
    # 2. In Tuner's `param_space` (as tune.choice objects) - used
    #    for initial sampling

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    scheduler = ray.tune.schedulers.PopulationBasedTraining(
        metric=algo_config.user_config.metric,
        mode="max",
        time_attr="training_iteration",
        perturbation_interval=1,
        hyperparam_mutations=algo_config.user_config.hyperparam_mutations,
        quantile_fraction=algo_config.user_config.quantile_fraction,
        log_config=True,  # used for reconstructing the PBT schedule
        require_attrs=True,
        synch=True,
    )

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html#ray.tune.TuneConfig
    tune_config = ray.tune.TuneConfig(
        trial_name_creator=lambda t: t.trial_id,
        trial_dirname_creator=lambda t: t.trial_id,
        scheduler=scheduler,
        reuse_actors=False,  # XXX: False is much safer and ensures no state leaks
        num_samples=algo_config.user_config.population_size,
        search_alg=None
    )

    for k, v in convert_to_param_space(algo_config.user_config.hyperparam_mutations).items():
        assert hasattr(algo_config, k)
        if isinstance(v, dict):
            deep_update(getattr(algo_config, k), v)
        else:
            setattr(algo_config, k, v)

    tuner = ray.tune.Tuner(
        trainable="MPPO",
        run_config=run_config,
        tune_config=tune_config,
        param_space=algo_config,
    )

    return tuner, checkpoint_load_dir


# XXX: PBT restore is broken in 2.38.0, I have fixed it locally
#      PR: https://github.com/ray-project/ray/pull/48616
def resume_tuner(opts):
    # Resuming will resume the W&B runs as well
    restore_path = util.to_abspath(opts.init_argument)
    print("*** RESUMING EXPERIMENT FROM %s ***" % restore_path)

    assert not opts.master_overrides, "overrides are not allowed when resuming an experiment"

    # THe env ID is stored in the config, but we don't have access to it here
    assert opts.env_gym_id, "env ID is required when resuming an experiment"
    ray.tune.registry.register_env(ENV_TUNE_ID, make_env_creator(opts.env_gym_id))

    # Trial ID will be the same => wandb run will be the same
    # Iteration will be >0 => any initial config
    # (e.g. old_wandb_run, model_load_file) will be ignored.
    # Tuner docs are clear that the restored config must be *identical*
    # => we can't change anything here
    assert not opts.master_overrides, "Overrides not allowed when resuming"
    return ray.tune.Tuner.restore(path=str(restore_path), trainable="MPPO")


if __name__ == "__main__":
    vcmi_gym.register_envs()
    ray.tune.registry.register_trainable("MPPO", MPPO_Algorithm)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment-name", type=str, metavar="<name>", default="MPPO-{datetime}", help="Experiment name (if starting a new experiment)")
    parser.add_argument("-m", "--init-method", type=str, metavar="<method>", default="load_python_config", help="Init method")
    parser.add_argument("-a", "--init-argument", type=str, metavar="<argument>", default=".pbt_config", help="Init argument")
    parser.add_argument("-o", "--master-overrides", type=str, metavar="<cfgpath>=<value>", action='append', help='Master config overrides')
    parser.add_argument("-w", "--wandb-project", type=str, metavar="<project>", default="newray", help="W&B project ('' to disable)")
    parser.add_argument("-e", "--env-gym-id", type=str, metavar="<env>", help="gym env id (when resuming an experiment only)")
    parser.add_argument("--num-cpus", type=int, metavar="<cpus>", help="CPU count for this node (default: auto)")

    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = """

Init methods:
  load_python_config*   Load master_config from a PY module (must have a load() function)
  load_json_config      Load master_config from a JSON file (must have a "_master_config" root key)
  load_ray_checkpoint   Load master_config and models from a ray checkpoint
  load_wandb_artifact   Load a ray checkpoint from a W&B artifact
  load_wandb_run        Load a ray checkpoint from a W&B run's latest artifact
  resume_experiment     Seamlessly resume an experiment from a path (no changes allowed)

Example:
  python -m rl.ray.mppo.main ...

  ... -n "MPPO-test-{datetime}"
  ... -m load_wandb_run -a s-manolloff/newray/35091_00000
  ... -m load_wandb_artifact -a s-manolloff/newray/model.pt:v52
  ... -m resume_experiment -a data/MPPO-20241106_163258 -e VCMI-v4

"""

    opts = parser.parse_args()

    if opts.init_method == "resume_experiment":
        tuner = resume_tuner(opts)
        checkpoint_load_dir = None
    else:
        tuner, checkpoint_load_dir = new_tuner(opts)

    try:
        ncpus = opts.num_cpus or multiprocessing.cpu_count()
        ray.init(resources={"train_cpu": ncpus, "eval_cpu": ncpus})

        tuner.fit()
    finally:
        if isinstance(checkpoint_load_dir, tempfile.TemporaryDirectory):
            checkpoint_load_dir.cleanup()
