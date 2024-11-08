import ray.tune
import vcmi_gym
import datetime
import ray
from wandb.util import json_dumps_safer

from .pbt_main import make_env_creator
from . import MPPO_Config, MPPO_Algorithm, util
from . import pbt_config

if __name__ == "__main__":
    vcmi_gym.register_envs()
    now = datetime.datetime.now()
    master_config = pbt_config.load()

    debug_overrides = dict(
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
            "evaluation_num_env_runners": 1,
        },
        user={
            "env_runner_keepalive_interval_s": 60,
            "experiment_name": f"test-{now.strftime('%Y%m%d_%H%M%S')}",
            "wandb_project": None,
            "wandb_log_interval_s": 0,

            # These will be replaced (not merged)
            # "hyperparam_mutations": {"vf_clip_param": [0.1, 0.2, 0.3]},
            # "hyperparam_values": {},

            "population_size": 1,
            "quantile_fraction": 0.3,
            "training_step_duration_s": 30,
        }
    )

    # Typically added via cmdline args
    new_keys = dict(
        checkpoint_load_dir=None,
        git_head="git_head",
        git_is_dirty=False,
        master_overrides=None,
        model_load_file=None,
        init_method="init_method",
        init_argument="init_argument",
        timestamp="2000-01-01T00:00:00",
    )

    master_config = util.deepmerge(master_config, debug_overrides, allow_new=False, update_existing=True)

    user_config = master_config["user"]
    user_config["hyperparam_mutations"] = {}
    user_config["hyperparam_values"] = {}
    util.deepmerge(user_config, new_keys, in_place=True, allow_new=True, update_existing=False)

    env_id = master_config["environment"]["env"]
    ray.tune.registry.register_env(env_id, make_env_creator(env_id))
    ray.tune.registry.register_trainable("MPPO", MPPO_Algorithm)

    ray.init(
        # address="HEAD_NODE_IP:6379",
        resources={"train_cpu": 8, "eval_cpu": 8}
    )

    algo_config = MPPO_Config()
    algo_config.master_config(master_config)
    algo = algo_config.build()

    for i in range(2):
        res = algo.train()
        print("Result from training step %d:\n%s" % (i, json_dumps_safer(res)))
        print("Will sleep 10s now...")
        import time
        time.sleep(10)

    print("Done")
