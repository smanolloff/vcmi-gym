import ray.tune
import vcmi_gym
from wandb.util import json_dumps_safer

from .pbt_config import pbt_config
from .pbt_main import build_master_config, env_creator #make_env_creator
from . import MPPO_Config, MPPO_Callback, MPPO_Algorithm


env_cls = getattr(vcmi_gym, pbt_config["env"]["cls"])

algo_config = MPPO_Config()
algo_config.callbacks(MPPO_Callback)  # this cannot go in master_config

pbt_config = {
    # "wandb_project": "newray",
    "wandb_project": None,
    "experiment_name": "resumetest-1-{datetime}",

    # Relative paths are interpreted w.r.t. VCMI_GYM root dir
    "load_options": {
        # Resume entire tune experiment
        "experiment": {
            "path": "data/newray-20241105_000738",
        },

        # Load policy & optimizer from a checkpoint
        "learner_group": {
            "path": "data/newray-20241105_000738/35091_00000/checkpoint_000030/learner_group",
        },

        # # Load model from a JIT file
        # "model": {
        #     "path": "data/newray-20241105_000738/35091_00000/checkpoint_000027/jit-model.pt",
        #     # rlmodule <=> model layer
        #     "layer_mapping": {
        #         "encoder.encoder": "encoder_actor",
        #         "pi": "actor",
        #         "vf": "critic",
        #     }
        # }
    },

    "metric": "train/ep_rew_mean",
    "population_size": 2,
    "quantile_fraction": 0.5,
    "wandb_log_interval_s": 5,
    "training_step_duration_s": 15,
    "evaluation_episodes": 10,
    "env_runner_keepalive_interval_s": 5,  # must be smaller than "user_timeout"

    "hyperparam_mutations": {
        "vf_clip_param": [0.1, 0.2, 0.3],
        "env_config": {  # affects training env only
            "term_reward_mult": [0, 5]
        }
    },

    "clip_param": 0.3,
    "entropy_coeff": 0.0,
    "gamma": 0.8,
    "grad_clip": 5,
    "grad_clip_by": "global_norm",  # global_norm = nn.utils.clip_grad_norm_(model.parameters)
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "lambda_": 1.0,
    "lr": 0.001,
    "minibatch_size": 20,
    "num_epochs": 1,
    "train_batch_size_per_learner": 100,
    "shuffle_batch_per_epoch": True,
    "use_critic": True,
    "use_gae": True,
    "use_kl_loss": True,
    "vf_clip_param": 10.0,
    "vf_loss_coeff": 1.0,

    "vf_share_layers": True,
    "network": {
        "attention": None,
        "features_extractor1_misc": [
            {"t": "LazyLinear", "out_features": 4},
            {"t": "LeakyReLU"},
        ],
        "features_extractor1_stacks": [
            {"t": "LazyLinear", "out_features": 8},
            {"t": "LeakyReLU"},
            {"t": "Flatten"},
        ],
        "features_extractor1_hexes": [
            {"t": "LazyLinear", "out_features": 8},
            {"t": "LeakyReLU"},
            {"t": "Flatten"},
        ],
        "features_extractor2": [
            {"t": "Linear", "in_features": 1484, "out_features": 512},
            {"t": "LeakyReLU"},
        ],
        "actor": {"t": "Linear", "in_features": 512, "out_features": 2312},
        "critic": {"t": "Linear", "in_features": 512, "out_features": 1}
    },
    "env": {
        "cls": "VcmiEnv_v4",
        "cpu_demand": {
            # As measured on Mac M1.
            # XXX: will probably differ on other CPUs. Might need a matrix.
            "BattleAI": 1,
            "StupidAI": 0.6
        },
        "episode_duration_s": {
            # Needed for properly setting "evaluation_sample_timeout_s"
            # Measured on Mac M1 vs. MMAI_USER (will be longer vs. a model)
            # XXX: will probably differ on other CPUs. Might need a matrix.
            "BattleAI": 1.5,
            "StupidAI": 0.04
        },
        "step_duration_s": {
            # Needed for properly setting "rollout_fragment_length"
            # Measured on Mac M1 vs. MMAI_USER (will be longer vs. a model)
            # XXX: will probably differ on other CPUs. Might need a matrix.
            "BattleAI": 0.08,
            "StupidAI": 0.002
        },
        # Either eval or train env MUST BE proc
        # (ray always creates a local env for both)
        "eval": {
            "runners": 1,  # 0=use main process
            "kwargs": {
                "opponent": "StupidAI",
                "conntype": "thread",
            },
        },
        "train": {
            "runners": 0,  # 0=use main process
            "kwargs": {
                "opponent": "StupidAI",
                "conntype": "thread",
            },
        },
        "common": {
            "kwargs": {
                "mapname": "gym/generated/4096/4096-mixstack-100K-01.vmap",
                # "vcmienv_loglevel": "DEBUG",
                "reward_dmg_factor": 5,
                "step_reward_fixed": 0,
                "step_reward_frac": -0.001,
                "step_reward_mult": 1,
                "term_reward_mult": 0,
                "reward_clip_tanh_army_frac": 1,
                "reward_army_value_ref": 500,
                "reward_dynamic_scaling": False,
                "random_terrain_chance": 100,
                "town_chance": 10,
                "warmachine_chance": 40,
                "tight_formation_chance": 0,
                "battlefield_pattern": "",
                "random_heroes": 1,
                "random_obstacles": 1,
                "mana_min": 0,
                "mana_max": 0,
                "swap_sides": 0,
                "user_timeout": 3600,
                "vcmi_timeout": 3600,
                "boot_timeout": 3600,
                "sparse_info": True,
            },
        },
    },
}


if __name__ == "__main__":
    algo_config = MPPO_Config()
    algo_config.callbacks(MPPO_Callback)  # this cannot go in master_config

    init_info = MPPO_Config.InitInfo(
        experiment_name="aaa",
        timestamp="2000-01-01T00:00:00",
        git_head="git-head",
        git_is_dirty=False,
        init_method="init_method",
        init_argument="init_argument",
        wandb_project=None,
        checkpoint_load_dir=None,
        hyperparam_values={},
        master_overrides={},
    )

    master_config = build_master_config(pbt_config)

    # This info is used during setup and is also recorded in history
    master_config["user"]["init_info"] = init_info
    master_config["user"]["init_history"].append(init_info)

    env_cls = getattr(vcmi_gym, master_config["user"]["env_cls"])
    # ray.tune.registry.register_env("VCMI", make_env_creator(env_cls))
    ray.tune.registry.register_env("VCMI", env_creator)
    ray.tune.registry.register_trainable("MPPO", MPPO_Algorithm)

    algo_config.master_config(master_config)

    if (
        algo_config.evaluation_num_env_runners == 0
        and algo_config.num_env_runners == 0
        and algo_config.env_config["conntype"] == "thread"
        and algo_config.evaluation_config["env_config"]["conntype"] == "thread"
    ):
        raise Exception("Both 'train' and 'eval' runners are local -- at least one must have conntype='proc'")

    # ray.tune.registry.register_env("VCMI", make_env_creator(env_cls))
    algo = algo_config.build()

    for i in range(2):
        res = algo.train()
        print("Result from training step %d:\n%s" % (i, json_dumps_safer(res)))
        print("Will sleep 10s now...")
        import time
        time.sleep(10)

    print("Done")
