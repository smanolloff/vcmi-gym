import sys
import os
import tempfile
import numpy as np
import ray
from ray import train, tune
from datetime import datetime
from ray.tune.schedulers.pb2 import PB2
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from stable_baselines3.common.callbacks import BaseCallback

from vcmi_gym.tools.main import run
from config.raytune.ppo import config

class RaytuneCallback(BaseCallback):
    def __init__(self, perturbation_interval):
        super().__init__()

        self.perturbation_interval = perturbation_interval
        self.n_rollouts = 0
        self.ep_rew_avg = 0.0

    def _on_step(self):
        env = self.training_env.envs[0].unwrapped
        if env.terminated:
            self.n_terminations += 1
            self.ep_rew_avg += (env.reward_total - self.ep_rew_avg) / self.n_terminations

    def _on_rollout_end(self):
        self.n_rollouts += 1

        # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/tune/examples/pbt_function.py#L81-L86
        # Based on these metrics, run may stop (see run_config)
        metrics = {"ep_rew_avg": self.ep_rew_avg}

        if self.n_rollouts % self.perturbation_interval == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                f = os.path.join(tempdir, "model.zip")
                print("Model checkpoint: %s" % f)
                self.model.save(f)
                train.report(metrics, checkpoint=train.Checkpoint.from_directory(tempdir))

        else:
            train.report(metrics)


#
# PB2 Example (from ray docs):
#
# https://docs.ray.io/en/latest/tune/examples/includes/pb2_example.html#pb2-example
#
def main():
    if len(sys.argv) != 2:
        raise Exception("Expected alg name as the only argument, eg. PPO")

    alg = sys.argv[1]

    if alg not in ["PPO"]:
        raise Exception("Only PPO is supported for raytune currently")

    # Increase for shorter population iteration cycles
    total_reports = 100
    total_timesteps = config["param_space"]["total_timesteps"]
    learner_n_steps = config["param_space"]["learner_kwargs"]["n_steps"]
    total_rollouts = total_timesteps / learner_n_steps
    perturbation_interval = int(1.0/config["desired_perturbations"] * total_rollouts)

    print("perturbation_interval: %d" % perturbation_interval)

    def train_function(cfg):
        extras = {
            "train_sb3.callback": RaytuneCallback(perturbation_interval),
            "checkpoint": train.get_checkpoint()
        }

        print(cfg)
        run("train_ppo", cfg, extras)

    @ray.remote(num_cpus=2, num_gpus=0)
    class MyActor:
        def __init__(self):
            self.value = 0
            print("w0000t")

        def train_function(cfg):
            print("w0000OOOOOOOOOOOOt")
            extras = {
                "train_sb3.callback": RaytuneCallback(perturbation_interval),
                "checkpoint": train.get_checkpoint()
            }

            run("train_ppo", cfg, extras)

    class MyTrainable(tune.Trainable):
        def setup(self, cfg):
            # Initialize Ray actors
            # self.actor = MyActor.remote()
            self.cfg = cfg

        def step(self):
            # ray.get(self.actor.perform_action.remote(self.cfg))
            extras = {
                "train_sb3.callback": RaytuneCallback(perturbation_interval),
                "checkpoint": train.get_checkpoint()
            }
            run("train_ppo", self.cfg, extras)
            return {"ep_rew_avg": 1}

        def save_checkpoint(self, checkpoint_dir):
            print("ADSADDSADSADASDASADS")
            breakpoint()
            # Save model/state if needed
            pass

        def load_checkpoint(self, checkpoint_dir):
            print("QEWQQRQWERWQQWER")
            breakpoint()
            # Restore model/state if needed
            pass

    # https://docs.ray.io/en/latest/tune/api/doc/tune.schedulers.pb2.PB2.html#ray-tune-schedulers-pb2-pb2
    pb2 = PB2(
        time_attr="args.criteria",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=perturbation_interval,
        hyperparam_bounds=config["hyperparam_bounds"],
        quantile_fraction=config["quantile_fraction"],
        log_config=False,  # used for reconstructing the config schedule
        require_attrs=True,
        # synch=True,
    )

    # https://docs.ray.io/en/latest/train/api/doc/train.RunConfig.html#ray-train-runconfig
    run_config = train.RunConfig(
        name="PB2-" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        verbose=False,
        failure_config=train.FailureConfig(fail_fast=True),
        checkpoint_config=train.CheckpointConfig(num_to_keep=50),
        stop={"ep_rew_avg": 500},
        callbacks=[WandbLoggerCallback(log_config=True, project=config["wandb_project"])]
    )

    def trial_name_creator(trial):
        return f"{alg}-{trial.trial_id}"

    tune_config = tune.TuneConfig(
        # metric="ep_rew_avg",
        # mode="max",
        # search_alg=None,
        # num_samples=1,
        # time_budget_s=None,
        # reuse_actors=None,  # defaults to true for function trainables
        max_concurrent_trials=1,
        trial_name_creator=trial_name_creator,
        scheduler=pb2,
        num_samples=4
    )

    tuner = tune.Tuner(
        train_function,
        # MyTrainable(logger_creator=N),
        run_config=run_config,
        tune_config=tune_config,
        param_space=config["param_space"]
    )

    trainable = MyTrainable()
    resource_group = tune.PlacementGroupFactory([{"CPU": 2, "GPU": 1}])
    tune.with_resources(MyTrainable, resource_group)
    ray.init(address='192.168.0.193:6379')
    tuner.fit()


if __name__ == "__main__":
    main()
