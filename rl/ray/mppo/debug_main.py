import ray.tune
import vcmi_gym
from wandb.util import json_dumps_safer
from .pbt_config import pbt_config
from . import MPPO_Config, MPPO_Callback
from .pbt_main import build_master_config

env_cls = getattr(vcmi_gym, pbt_config["env"]["cls"])

algo_config = MPPO_Config()
algo_config.callbacks(MPPO_Callback)  # this cannot go in master_config

pbt_config = {
    "metric": "eval/is_success",
    "population_size": 2,
    "quantile_fraction": 0.5,
    "wandb_log_interval_s": 5,
    "training_step_duration_s": 10,
    "evaluation_episodes": 100,

    "hyperparam_mutations": {},

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
    "train_batch_size_per_learner": 10000,
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
            "runners": 8,  # 0=use main process
            "kwargs": {
                "conntype": "thread",
                "opponent": "StupidAI",
            },
        },
        "train": {
            "runners": 8,  # 0=use main process
            "kwargs": {
                "conntype": "thread",
                "opponent": "StupidAI",
            },
        },
        "common": {
            "kwargs": {
                "mapname": "gym/generated/4096/4096-mixstack-100K-01.vmap",
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
    "experiment_name": "newray-test",  # overwritten via cmd-line args
    "wandb_project": "vcmi-gym",
}


# Even if there are remote runners, ray still creates 2 local runners
# It does not use them for anything except inferring action/observation spaces
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


if __name__ == "__main__":
    algo_config = MPPO_Config()
    algo_config.callbacks(MPPO_Callback)  # this cannot go in master_config
    algo_config.master_config(build_master_config(pbt_config))

    # ray.tune.registry.register_env("VCMI", lambda cfg: (print("NEW ENV WITH INDEX: %s" % cfg.worker_index), env_cls(**cfg)))
    def env_creator(cfg):
        if cfg.num_workers > 0 and cfg.worker_index == 0:
            return DummyEnv()
        else:
            return env_cls(**cfg)

    ray.tune.registry.register_env("VCMI", env_creator)
    algo = algo_config.build()

    for i in range(2):
        res = algo.train()
        print("Result from training step %d:\n%s" % (i, json_dumps_safer(res)))
        print("Will sleep 10s now...")
        import time
        time.sleep(10)

    print("Done")
