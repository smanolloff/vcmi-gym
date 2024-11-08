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
import dataclasses
import pathlib
import multiprocessing

from ray.rllib.utils import deep_update

from . import MPPO_Algorithm, MPPO_Config, util

# Silence here in order to supress messages from local runners
# Silence once more in algo init to supress messages from remote runners
util.silence_log_noise()


def calculate_fragment_duration_s(cfg, train_env_cfg, eval_env_cfg):
    if "train_batch_size_per_learner" in cfg["hyperparam_mutations"]:
        batch_sizes = cfg["hyperparam_mutations"]["train_batch_size_per_learner"]
    else:
        batch_sizes = [cfg["train_batch_size_per_learner"]]

    opponent = train_env_cfg["kwargs"]["opponent"]
    max_fragment_length = max(batch_sizes) / max(1, train_env_cfg["runners"])
    step_duration_s = cfg["env"]["step_duration_s"][opponent]
    fragment_duration_s = max_fragment_length * step_duration_s
    print(f"Estimated time for collecting samples: {fragment_duration_s:.1f}s")

    # Maximum allowed time for sample collection (hard-coded)
    max_fragment_duration_s = 30

    if fragment_duration_s > max_fragment_duration_s:
        raise Exception(
            "Estimated fragment_duration_s is too big: %.1f (based on %s's step_duration_s=%s).\n"
            "To fix this, either:\n"
            "\t* Increase train env runners (current: %d)\n"
            "\t* Decrease train_batch_size_per_learner (current: %s)\n"
            "\t* Increase max_fragment_duration_s (current: %d, hard-coded)" % (
                fragment_duration_s, opponent, step_duration_s,
                train_env_cfg["runners"], batch_sizes,
                max_fragment_duration_s
            )
        )

    return fragment_duration_s


def build_master_config(cfg):
    env_cls = getattr(vcmi_gym, cfg["env"]["cls"])
    train_env_cfg = util.deepmerge(cfg["env"]["common"], cfg["env"]["train"])
    eval_env_cfg = util.deepmerge(cfg["env"]["common"], cfg["env"]["eval"])
    fragment_duration_s = calculate_fragment_duration_s(cfg, train_env_cfg, eval_env_cfg)

    # Estimation is for Mac M1, ensure enough headroom for slower CPUs
    fragment_duration_headroom = 10
    sample_timeout_s = fragment_duration_s * fragment_duration_headroom
    print(f"Using sample_timeout_s={sample_timeout_s:.1f}")


def update_config_value(cfg, path, value):
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


def make_env_creator(env_id):
    env_cls = util.get_env_cls(env_id)

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
            return env_cls(**cfg)

    return env_creator


def new_tuner(opts):
    # All cases below will result in a new W&B Run
    # The config used depends on the init method
    assert opts.experiment_name is not None
    master_config = None
    hyperparam_values = None
    checkpoint_load_dir = None

    def load_ray_checkpoint(path):
        # Same as "load_ray_checkout"
        master_config = json.loads(f"{checkpoint_dir}/master_config.json")
        hyperparam_values = util.common_dict(
            a=master_config["user"]["hyperparam_mutations"],
            b=master_config,
            strict=True
        )

        # No sense in loading from file after loading from checkpoint
        # => set `model_load_file` to None
        master_config["user"]["init_info"]["model_load_file"] = None

        return master_config, hyperparam_values

    def load_wandb_artifact(artifact):
        tempdir = tempfile.TemporaryDirectory()
        artifact.download(tempdir)
        master_config, hyperparam_values = load_ray_checkpoint(tempdir.name)
        return master_config, hyperparam_values, tempdir

    match opts.init_method:
        # Same as "load_wandb_artifact" but for run's latest artifact
        case "load_wandb_run":
            assert len(opts.init_argument.split("/")) == 3, "wandb run format: s-manolloff/<project>/<run_id>"
            run = wandb.Api().run(opts.init_argument)
            artifact = next(a for a in reversed(run.logged_artifacts()) if a.type == "model")

            (
                master_config,
                hyperparam_values,
                checkpoint_load_dir
            ) = load_wandb_artifact(artifact)

        case "load_wandb_artifact":
            artifact = wandb.Api().artifact(opts.init_argument)

            (
                master_config,
                hyperparam_values,
                checkpoint_load_dir
            ) = load_wandb_artifact(artifact)

        case "load_ray_checkpoint":
            # must have a .name attribute
            checkpoint_load_dir = pathlib.Path(opts.init_argument)
            master_config, hyperparam_values = load_ray_checkpoint(checkpoint_load_dir.name)

        case "load_json_config":
            master_config = json.loads(opts.init_argument)

        case "load_python_config":
            import importlib
            modpath = opts.init_argument or ".pbt_config"
            mod = importlib.import_module(modpath, package=__package__)
            master_config = mod.load()

        case _:
            raise Exception(f"Unknown init source: {opts.init_argument}")

    # should be present at this point
    assert master_config

    master_overrides = {}

    # config_overrides is a list of "path.to.key=value"
    for co in (opts.master_overrides or []):
        name, value = co.split("=")
        oldvalue, newvalue = update_config_value(master_config, name, value)
        if oldvalue != newvalue:
            master_overrides[name] = [oldvalue, newvalue]

    git = pygit2.Repository(util.vcmigym_root_path)
    now = datetime.datetime.now()

    init_info = MPPO_Config.InitInfo(
        experiment_name=opts.experiment_name.format(datetime=now.strftime("%Y%m%d_%H%M%S")),
        timestamp=now.astimezone().isoformat(),
        git_head=str(git.head.target),
        git_is_dirty=any(git.diff().patch),
        init_method=opts.init_method,
        init_argument=opts.init_argument,
        wandb_project=opts.wandb_project,
        checkpoint_load_dir=checkpoint_load_dir.name if checkpoint_load_dir else None,
        hyperparam_values=hyperparam_values,
        master_overrides=master_overrides,
    )

    # This info is used during setup and is also recorded in history
    master_config["user"]["init_info"] = init_info
    master_config["user"]["init_history"].append(init_info)

    # Validation: dict -> UserConfig -> dict
    master_config["user"] = dataclasses.asdict(MPPO_Config.UserConfig(**master_config["user"]))

    env_id = master_config["environment"]["env"]
    ray.tune.registry.register_env(env_id, make_env_creator(env_id))
    ray.tune.registry.register_trainable("MPPO", MPPO_Algorithm)

    algo_config = MPPO_Config()
    algo_config.master_config(master_config)

    if (
        algo_config.evaluation_num_env_runners == 0
        and algo_config.num_env_runners == 0
        and algo_config.env_config["conntype"] == "thread"
        and algo_config.evaluation_config["env_config"]["conntype"] == "thread"
    ):
        raise Exception("Both 'train' and 'eval' runners are local -- at least one must have conntype='proc'")

    checkpoint_config = ray.train.CheckpointConfig(num_to_keep=3)
    run_config = ray.train.RunConfig(
        name=opts.experiment_name,
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
        metric=pbt_config["metric"],
        mode="max",
        time_attr="training_iteration",
        perturbation_interval=1,
        hyperparam_mutations=algo_config.user_config.hyperparam_mutations,
        quantile_fraction=pbt_config["quantile_fraction"],
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
        num_samples=pbt_config["population_size"],
        search_alg=None
    )

    #
    # XXX: no use in limiting cluster or worker resources
    #      Calculating the appropriate population_size is enough
    #      (it is essentially the limit for number of spawned workers)
    #      => don't impose any additional limits here to avoid confusion
    #      GPU must be non-0 to be available at all => set to 0.01 if cuda is available
    #
    # ray.init() by default uses num_cpus=os.cpu_count(), num_gpus=torch.cuda.device_count()
    # However, if GPU is 0 then CUDA is always unavailable in the workers => set to 0.01
    #
    # resources = ray.tune.PlacementGroupFactory([{
    #     "CPU": 0.01,
    #     "GPU": 0.01 if torch.cuda.is_available() else 0
    # }])

    # trainable = ray.tune.with_parameters(trainable_cls, initargs=initargs)
    # trainable = ray.tune.with_resources(trainable, resources=resources)

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


# XXX: PBT restore is broken in 2.38.0, I locally fixed it in pip
#      PR: https://github.com/ray-project/ray/pull/48616
def resume_tuner(opts):
    # Resuming will resume the W&B runs as well
    assert opts.experiment_name is None
    restore_path = util.to_abspath(opts.init_argument)
    print("*** RESUMING EXPERIMENT FROM %s ***" % restore_path)

    # Trial ID will be the same => wandb run will be the same
    # Iteration will be >0 => any initial config
    # (e.g. old_wandb_run, model_load_file) will be ignored.
    # We can't change anything here:
    # Tuner docs are clear that the restored config must be *identical*
    assert not opts.master_overrides, "Overrides not allowed when resuming"
    return ray.tune.Tuner.restore(path=str(restore_path), trainable="MPPO")


if __name__ == "__main__":
    vcmi_gym.register_envs()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment-name", metavar="<name>", default="MPPO-{datetime}", help="Experiment name (if starting a new experiment)")
    parser.add_argument("-m", "--init-method", metavar="<method>", default="default", help="Init method")
    parser.add_argument("-a", "--init-argument", metavar="<argument>", help="Init argument")
    parser.add_argument("-o", "--master-overrides", metavar="<cfgpath>=<value>", action='append', help='Master config overrides')
    parser.add_argument("-w", "--wandb-project", metavar="<project>", default="newray", help="W&B project ('' disables W&B)")
    parser.add_argument("--num-cpus", metavar="<cpus>", help="CPU count for this node (default: auto)")
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = """

Init methods:
  load_python_config*   Load master_config from a PY module's load() function (default: .pbt_config)
  load_json_config      Load master_config from a JSON file
  load_ray_checkpoint   Load master_config and models from a ray checkpoint
  load_wandb_artifact   Load a ray checkpoint from a W&B artifact
  load_wandb_run        Load a ray checkpoint from a W&B run's latest artifact
  resume_experiment     Seamlessly resume an experiment from a path (no changes allowed)

Example:
  python -m rl.ray.mppo.main ...

  ... -n "MPPO-test-{datetime}"
  ... -m load_wandb_run -a s-manolloff/newray/35091_00000 -n "MPPO-test-{datetime}"
  ... -m load_wandb_artifact -a s-manolloff/newray/model.pt:v52
  ... -m resume_experiment -a data/MPPO-20241106_163258

"""

    opts = parser.parse_args()

    if opts.init_method == "resume_experiment":
        tuner = resume_tuner(opts)
        checkpoint_dir = None
    else:
        tuner, checkpoint_dir = new_tuner(opts)

    try:
        ncpus = opts.num_cpus or multiprocessing.cpu_count()

        # To add remote machines to the ray cluster:
        # $ ray start --address=<HEAD_NODE_IP>:6379 --node-ip-address=<WORKER_NODE_IP>

        ray.init(
            # address="HEAD_NODE_IP:6379",
            resources={"train_cpu": ncpus, "eval_cpu": ncpus}
        )

        tuner.fit()
    finally:
        if isinstance(checkpoint_dir, tempfile.TemporaryDirectory):
            checkpoint_dir.cleanup()
