import sys
import os
import tempfile
import copy
from datetime import datetime
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import wandb

# MUST come BEFORE importing ray
os.environ["RAY_DEDUP_LOGS"] = "0"

from ray import train, tune  # noqa: E402
from ray.tune.schedulers.pb2 import PB2  # noqa: E402
# from ray.air.integrations.wandb import WandbLoggerCallback  # noqa: E402
# from ray.air.integrations.wandb import setup_wandb  # noqa: E402
from config.raytune.ppo import config  # noqa: E402

from vcmi_gym.tools.main import run  # noqa: E402


class RaytuneCallback(BaseCallback):
    def __init__(self, perturbation_interval, reduction_factor, hyperparam_bounds):
        super().__init__()

        self.reduction_factor = reduction_factor
        self.perturbation_interval = perturbation_interval  # already reduced
        self.n_rollouts = 0
        self.n_rollouts_reduced = 0
        self.n_perturbations = 0
        self.leaf_params = self._get_leaf_keys(hyperparam_bounds)
        self.metric_aggregations = {
            "rew_mean": np.zeros(reduction_factor, dtype=np.float32),
            "success_rate": np.zeros(reduction_factor, dtype=np.float32),
            "net_value_mean": np.zeros(reduction_factor, dtype=np.float32),
            "n_errors_mean": np.zeros(reduction_factor, dtype=np.float32),
        }
        # print("RAyTUNE CALLBACK INIT")

    def _on_step(self):
        pass

    def _on_training_start(self):
        params = {}
        env = self.training_env.envs[0].unwrapped

        for name in self.leaf_params:
            if hasattr(self.model, name):
                params[f"config/{name}"] = getattr(self.model, name)
            elif hasattr(env, name):
                params[f"config/{name}"] = getattr(env, name)
            else:
                raise Exception("Could not find value for %s" % name)

        wandb.log(params)

    def _on_rollout_end(self):
        self.n_rollouts += 1

        # See notes in train_sb3 regarding Monitor wrapper.
        # We use ep_info["r"], here to calc rew_mean, code copied from:
        # https://github.com/DLR-RM/stable-baselines3/blob/v2.2.1/stable_baselines3/common/on_policy_algorithm.py#L292
        metrics = {
            "rew_mean": safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
        }

        rem = self.n_rollouts % self.reduction_factor

        for (k, v) in metrics.items():
            self.metric_aggregations[k][rem-1] = v

        if rem != 0:
            return

        self.n_rollouts_reduced += 1

        # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/tune/examples/pbt_function.py#L81-L86
        # Based on these metrics, run may stop (see run_config)
        log = {k: np.mean(self.metric_aggregations[k]) for k in metrics.keys()}
        report = {"rew_mean": log["rew_mean"]}

        # print("aggregations[rew_mean]: %s" % self.metric_aggregations["rew_mean"])
        # print("report: %s" % report)

        if self.n_rollouts_reduced % self.perturbation_interval == 0:
            self.n_perturbations += 1
            log["n_perturbations"] = self.n_perturbations
            with tempfile.TemporaryDirectory() as tempdir:
                f = os.path.join(tempdir, "model.zip")
                # print("Model checkpoint: %s" % f)
                self.model.save(f)
                train.report(report, checkpoint=train.Checkpoint.from_directory(tempdir))

        else:
            train.report(report)

    def _get_leaf_keys(self, data):
        leaf_keys = []
        for key, value in data.items():
            if isinstance(value, list):
                leaf_keys.append(key)
            else:
                leaf_keys.extend(self._get_leaf_keys(value))
        return leaf_keys


def get_leaf_paths(data, parent_keys=[]):
    paths = []
    for key, value in data.items():
        current_keys = parent_keys + [key]
        if isinstance(value, dict):
            paths.extend(get_leaf_paths(value, current_keys))
        else:
            paths.append(current_keys)
    return paths


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


# A ray callback for logging ray-specific metrics
# NOT WORKING: this object living in the main PID and thread
# But the trials are other PIDs and threads => wandb is not inited here...
# class MyWandbLoggerCallback(tune.logger.LoggerCallback):

#
# PB2 Example (from ray docs):
# https://docs.ray.io/en/latest/tune/examples/includes/pb2_example.html#pb2-example
#
def main():
    cwd = os.getcwd()

    if len(sys.argv) != 2:
        raise Exception("Expected alg name as the only argument, eg. PPO")

    alg = sys.argv[1]

    if alg not in ["PPO"]:
        raise Exception("Only PPO is supported for raytune currently")

    # NOTE:
    # Apparently, tune also needs those defined as ranges in param_space
    # https://discuss.ray.io/t/pb2-seems-stuck-in-space-margins-and-raises-exceptions-with-lambdas/467/8

    update_param_space(config["hyperparam_bounds"], config["param_space"])

    # Increase for shorter population iteration cycles
    # total_timesteps = config["param_space"]["total_timesteps"]
    # learner_n_steps = config["param_space"]["learner_kwargs"]["n_steps"]
    # total_rollouts = total_timesteps / learner_n_steps
    # perturbation_interval = int(1.0/config["desired_perturbations"] * total_rollouts)

    perturbation_interval = config["perturbation_interval"]
    reduction_factor = config["reduction_factor"]
    config["param_space"]["log_interval"] = reduction_factor

    # Reporting will be reduced by the same factor
    # => no *real* change in perturbation interval
    perturbation_interval //= reduction_factor

    results_dir = config["results_dir"]
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(cwd, results_dir)

    # print("perturbation_interval: %d" % perturbation_interval)

    def train_function(cfg):
        # print("[%s] TRAINING_FUNC START" % time.time())
        cfg = copy.deepcopy(cfg)

        train_ctx = train.get_context()
        trial_id = train_ctx.get_trial_id()
        trial_name = train_ctx.get_trial_name()
        experiment_name = train_ctx.get_experiment_name()

        # Hard-coding out_dir with the experiment name in order
        # to match ray's directory
        out_dir = os.path.join(results_dir, experiment_name, trial_id)

        if wandb.run:
            # print("[%s] FIRST FINISH WANDB" % time.time())
            wandb.finish(quiet=True)
            # print("[%s] WANDB FINISED. wandb.run? %s" % (time.time(), wandb.run))
        # else:
            # Only patch once => only if no prev runs
            # NOTE: this screws up the wandb workspace
            #       better to NOT patch it, which results in a warning:
            #       "WARNING Found log directory outside of given root_logdir..."
            #       However, that is harmless, and the resulting wandb workspace
            #       is much better
            # wandb.tensorboard.patch(root_logdir=out_root_dir)

        # print("[%s] INITWANDB: PID: %s, trial_id: %s" % (time.time(), os.getpid(), trial_id))
        # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/air/integrations/wandb.py#L601-L607
        wandb_run = wandb.init(
            id=trial_id,
            name="PB2_%s" % trial_name.split("_")[-1],
            resume="allow",
            reinit=True,
            allow_val_change=True,
            # To disable System/ stats:
            # settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
            group=experiment_name,
            project=config["wandb_project"],
            config=config,
            # NOTE: this takes a lot of time, better to have detailed graphs
            #       tb-only (local) and log only most important info to wandb
            # sync_tensorboard=True,
            sync_tensorboard=False,
        )
        # print("[%s] DONE WITH INITWANDB" % time.time())

        seed = int(np.random.default_rng().integers(2**31))
        cfg = dict(cfg, run_id=wandb_run.id, seed=seed, out_dir_template=out_dir)
        cb = RaytuneCallback(perturbation_interval, reduction_factor, config["hyperparam_bounds"])
        extras = {
            "wandb_run": wandb_run,
            "train_sb3.callback": cb,
            "checkpoint": train.get_checkpoint(),
            "overwrite_env": True,
        }

        # this function never returns. Not even a try/catch works.
        run("train_ppo", cfg, extras)

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.pb2.PB2.html#ray-tune-schedulers-pb2-pb2
    pb2 = PB2(
        # NOTE: we report ray metrics only on each rollout
        # => `training_iteration` is incremented on each rollout
        # Perturbation_interval will perturb on each Nth `time_attr`:
        time_attr="training_iteration",
        metric="rew_mean",
        mode="max",
        perturbation_interval=perturbation_interval,
        hyperparam_bounds=config["hyperparam_bounds"],
        quantile_fraction=config["quantile_fraction"],
        log_config=False,  # used for reconstructing the config schedule
        require_attrs=True,
        synch=True,
    )

    # https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray-train-runconfig
    run_config = train.RunConfig(
        name="PPO-PB2-%s" % datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        verbose=False,
        failure_config=train.FailureConfig(fail_fast=True),
        checkpoint_config=train.CheckpointConfig(num_to_keep=50),
        stop={"rew_mean": 500},
        callbacks=[],
        local_dir=results_dir,
    )

    tune_config = tune.TuneConfig(
        trial_name_creator=lambda t: t.trial_id,
        # trial_name_creator=trial_name_creator,
        scheduler=pb2,
        num_samples=config["population_size"]
    )

    tuner = tune.Tuner(
        train_function,
        run_config=run_config,
        tune_config=tune_config,
        param_space=config["param_space"],
    )

    tuner.fit()


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    main()
