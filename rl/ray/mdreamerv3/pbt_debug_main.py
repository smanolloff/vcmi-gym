import ray.tune
import vcmi_gym
import datetime
import ray
import json
import copy

from ..common.common_config import ENV_TUNE_ID, ENV_VERSION_CHECK_KEY
from ..common import common_main, util
from . import pbt_config, MDreamerV3_Config, MDreamerV3_Algorithm


def main():
    vcmi_gym.register_envs()
    now = datetime.datetime.now()
    master_config = pbt_config.load()

    debug_cfg = dict(
        # TRAIN runners
        environment={
            "env_config": {
                "mapname": "gym/A1.vmap",
                # "opponent": "StupidAI",
                # "user_timeout": 3600,
                # "vcmi_timeout": 3600,
                # "boot_timeout": 3600,
                "conntype": "proc",  # "thread" causes issues if workers are local
            },
        },
        env_runners={
            "num_env_runners": 0,
        },
        # EVAL runners
        evaluation={
            "evaluation_config": {
                "env_config": {
                    # "mapname": "gym/A1.vmap",
                    # "opponent": "StupidAI",
                    # "user_timeout": 3600,
                    # "vcmi_timeout": 3600,
                    # "boot_timeout": 3600,
                    "conntype": "proc",  # "thread" causes issues if workers are local
                },
            },
            "evaluation_duration": 50,
            "evaluation_interval": 100,
            "evaluation_num_env_runners": 0,
        },
        rl_module={},
        training={
            # "model_size": "XS",
            "training_ratio": 256,
            "gc_frequency_train_steps": 100,
            "batch_size_B": 16,
            "batch_length_T": 32,
            "horizon_H": 15,
            "gae_lambda": 0.95,
            "entropy_scale": 3e-4,
            "return_normalization_decay": 0.99,
            "train_critic": True,
            "train_actor": True,
            "intrinsic_rewards_scale": 0.1,
            "world_model_lr": 1e-4,
            "actor_lr": 3e-5,
            "critic_lr": 3e-5,
            "world_model_grad_clip_by_global_norm": 1000.0,
            "critic_grad_clip_by_global_norm": 100.0,
            "actor_grad_clip_by_global_norm": 100.0,
            "symlog_obs": True,
            "use_float16": False,
        },
        user={
            "env_runner_keepalive_interval_s": 60,
            "wandb_log_interval_s": 0,
            "model_load_file": "",

            # these dicts need to be replaced (not merged)
            # "hyperparam_mutations": {"vf_clip_param": [0.1, 0.2, 0.3]},
            # "hyperparam_values": {},
            "metric": "train/ep_rew_mean",
            "population_size": 1,
            "quantile_fraction": 0.3,
            "training_step_duration_s": 30,

            # Typically added via cmdline args:
            "checkpoint_load_dir": "",
            # "experiment_name": f"test-{now.strftime('%Y%m%d_%H%M%S')}",
            "experiment_name": f"MDreamerV3-{now.strftime('%Y%m%d_%H%M%S')}",
            "git_head": "git_head",
            "git_is_dirty": False,
            "init_method": "init_method",
            "init_argument": "init_argument",
            "master_overrides": {},
            "timestamp": "2000-01-01T00:00:00",
            "wandb_project": "newray",
        }
    )

    # Simulate cmd-line overrides "path.to.key=value"
    OVERRIDES = [
        # "rl_module.model_config.vf_share_layers=False"
    ]

    master_config = util.deepmerge(master_config, debug_cfg, allow_new=False, update_existing=True)
    master_config["user"]["hyperparam_mutations"] = {}
    master_config["user"]["hyperparam_values"] = {}

    override_summary = common_main.apply_and_summarize_overrides(master_config, OVERRIDES),
    print("*** OVERRIDE SUMMARY: %s" % override_summary)

    # DreamerV3 uses "special" EnvRunner which does not pass an EnvCtx as cfg
    # but rather a the plain cfg dict (i.e. no .num_workers or .worker_index)
    env_gym_id = master_config["user"]["env_gym_id"]
    env_cls = util.get_env_cls(env_gym_id)

    def env_creator(cfg):
        print(f"Env kwargs: {json.dumps(cfg)}")
        env_kwargs = copy.deepcopy(cfg)
        assert env_cls.ENV_VERSION == env_kwargs.pop(ENV_VERSION_CHECK_KEY)
        return env_cls(**env_kwargs)

    ray.tune.registry.register_env(ENV_TUNE_ID, env_creator)
    ray.tune.registry.register_trainable("MDreamerV3", MDreamerV3_Algorithm)

    ray.init(
        # address="HEAD_NODE_IP:6379",
        resources={"train_cpu": 8, "eval_cpu": 8}
    )

    algo_config = MDreamerV3_Config()
    algo_config.master_config(master_config)
    algo = algo_config.build()

    while True:
        algo.train()

    print("Done")


if __name__ == "__main__":
    main()
