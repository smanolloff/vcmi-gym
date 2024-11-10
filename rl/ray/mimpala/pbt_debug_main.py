import ray.tune
import vcmi_gym
import datetime
import ray
from wandb.util import json_dumps_safer

from . import pbt_config, MIMPALA_Config, MIMPALA_Algorithm
from ..common.common_config import ENV_TUNE_ID
from ..common import common_main, util


if __name__ == "__main__":
    vcmi_gym.register_envs()
    now = datetime.datetime.now()
    master_config = pbt_config.load()

    debug_cfg = dict(
        # TRAIN runners
        environment={
            "env_config": {
                "mapname": "gym/A1.vmap",
                "opponent": "StupidAI",
                "user_timeout": 3600,
                "vcmi_timeout": 3600,
                "boot_timeout": 3600,
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
                    "mapname": "gym/A1.vmap",
                    "opponent": "StupidAI",
                    "user_timeout": 3600,
                    "vcmi_timeout": 3600,
                    "boot_timeout": 3600,
                    "conntype": "proc",  # "thread" causes issues if workers are local
                },
            },
            "evaluation_duration": 10,
            "evaluation_num_env_runners": 0,
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
            "experiment_name": f"test-{now.strftime('%Y%m%d_%H%M%S')}",
            "git_head": "git_head",
            "git_is_dirty": False,
            "init_method": "init_method",
            "init_argument": "init_argument",
            "master_overrides": {},
            "timestamp": "2000-01-01T00:00:00",
            "wandb_project": "",
        }
    )

    master_config = util.deepmerge(master_config, debug_cfg, allow_new=False, update_existing=True)
    master_config["user"]["hyperparam_mutations"] = {}
    master_config["user"]["hyperparam_values"] = {}

    env_gym_id = master_config["user"]["env_gym_id"]
    ray.tune.registry.register_env(ENV_TUNE_ID, common_main.make_env_creator(env_gym_id))
    ray.tune.registry.register_trainable("MPPO", MIMPALA_Algorithm)

    ray.init(
        # address="HEAD_NODE_IP:6379",
        resources={"train_cpu": 8, "eval_cpu": 8}
    )

    algo_config = MIMPALA_Config()
    algo_config.master_config(master_config)
    algo = algo_config.build()

    for i in range(2):
        res = algo.train()
        print("Result from training step %d:\n%s" % (i, json_dumps_safer(res)))
        print("Will sleep 10s now...")
        import time
        time.sleep(10)

    print("Done")
