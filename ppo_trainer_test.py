import os
import copy
import re
import statistics
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import ray.tune
from sb3_contrib import MaskablePPO
import ray.train
# import wandb
import torch.optim
from datetime import datetime

# from ..wandb_init import wandb_init
# from .... import InfoDict
import vcmi_gym
from vcmi_gym import InfoDict
# from vcmi_gym.tools.raytune.sb3_callback import SB3Callback
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.utils import safe_mean


class WandbDummy():
    def log(self, any, commit=False):
        print("**** WANDB LOG (%s): %s" % (commit, any))


def deepmerge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                a[key] = b[key]
        else:
            raise Exception("Key not found: %s" % key)
    return a


# Replacing wandb to see what's logged
wandb = WandbDummy()


class SB3CallbackDummy(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_episodes = 0
        self.rollouts = 0
        self.ep_rew_mean = 0

        # wdb tables are logged only once, at end of the iteration
        # (ie. once every config["rollouts_per_iteration"])
        self.wdb_tables = {}
        self.uncommitted_log = False

    def _on_step(self):
        self.rollout_episodes += self.locals["dones"].sum()
        return True

    def _on_rollout_start(self):
        if self.uncommitted_log:
            # Must log *something* to commit the rest
            # Since timesteps are preserved between checkpoint loads,
            # mark this as a trial-related metric
            wandb.log({"trial/num_timesteps": self.num_timesteps})
            self.uncommitted_log = False

    def _on_rollout_end(self):
        self.rollouts += 1
        wdb_log = {"rollout/n_episodes": self.rollout_episodes}
        self.rollout_episodes = 0

        if self.rollouts % self.locals["log_interval"] != 0:
            return

        for k in InfoDict.SCALAR_VALUES:
            v = safe_mean([ep_info[k] for ep_info in self.model.ep_info_buffer])
            self.model.logger.record(f"{k}", v)
            wdb_log[k] = v

        # From here on it's W&B stuff only

        wdb_log["rollout/count"] = self.rollouts

        # Also add sb3's Monitor info keys: "r" (reward) and "l" (length)
        # (these are already recorded to TB by sb3, but not in W&B)
        self.ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        wdb_log["rollout/ep_rew_mean"] = self.ep_rew_mean

        v = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        wdb_log["rollout/ep_len_mean"] = v

        for (k, columns) in InfoDict.D1_ARRAY_VALUES.items():
            action_types_vec_2d = [ep_info[k] for ep_info in self.model.ep_info_buffer]
            ary = np.mean(action_types_vec_2d, axis=0)

            # In SB3's logger, Tensor objects are logged as a Histogram
            # https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/logger.py#L412
            # NOT logging this to TB, it's not visualized well there
            # tb_data = torch.as_tensor(ary)
            # self.model.logger.record(f"user/{k}", tb_data)

            # In W&B, we need to unpivot into a name/count table
            # NOTE: reserved column names: "id", "name", "_step" and "color"
            wk = f"table/{k}"
            rotated = [list(row) for row in zip(columns, ary)]
            if wk not in self.wdb_tables:
                self.wdb_tables[wk] = wandb.Table(columns=["key", "value"])

            wb_table = self.wdb_tables[wk]
            for row in rotated:
                wb_table.add_data(*row)

        # for k in InfoDict.D2_ARRAY_VALUES:
        #     action_types_vec_3d = [ep_info[k] for ep_info in self.model.ep_info_buffer]
        #     ary_2d = np.mean(action_types_vec_3d, axis=0)

        #     wk = f"table/{k}"
        #     if wk not in self.wdb_tables:
        #         # Also log the "rollout" so that inter-process logs (which are different _step)
        #         # can be aggregated if needed
        #         self.wdb_tables[wk] = wandb.Table(columns=["x", "y", "value"])

        #     wb_table = self.wdb_tables[wk]

        #     for (y, row) in enumerate(ary_2d):
        #         for (x, cell) in enumerate(row):
        #             wb_table.add_data(x, y, cell)

        # Commit will be done either:
        #   a) on_rollout_start()
        #   b) by the trainer (if this is the last rollout)
        wandb.log(wdb_log, commit=False)
        self.uncommitted_log = True


class PB2_PPOTrainerTest():
    # logfile = None

    def __init__(self):
        self.iteration = 69
        self.trial_id = "kurtrial_00001"
        self.trial_name = "kurtrial_00001"

    def log(self, msg):
        # if not cls.logfile:
        #     cls.logfile = tempfile.NamedTemporaryFile(mode="w")
        #     print("-- %s [%s] Logfile: %s\n" % (time.time(), os.getpid(), cls.logfile.name))

        # cls.logfile.write("%s\n" % msg)
        # cls.logfile.flush()
        # print(msg)
        print("-- %s [%s I=%d] %s" % (datetime.now().isoformat(), self.trial_name, self.iteration, msg))

    @staticmethod
    def default_resource_request(_config):
        return ray.tune.execution.placement_groups.PlacementGroupFactory([{"CPU": 1}])

    # XXX: in case of SIGTERM/SIGINT, ray does not wait for cleanup to finish
    #      during regular perturbation, it waits though
    def cleanup(self):
        self.model.env.close()
        # wandb.finish(quiet=True)

    def setup(self, cfg, initargs):
        self.rollouts_per_iteration = initargs["config"]["rollouts_per_iteration"]
        self.rollouts_per_role = initargs["config"]["rollouts_per_role"]
        self.maps_per_iteration = initargs["config"]["maps_per_iteration"]
        self.logs_per_iteration = initargs["config"]["logs_per_iteration"]
        self.hyperparam_bounds = initargs["config"]["hyperparam_bounds"]
        self.initial_checkpoint = initargs["config"]["initial_checkpoint"]
        self.experiment_name = initargs["experiment_name"]
        self.all_params = copy.deepcopy(initargs["config"]["all_params"])

        # self._wandb_init(self.experiment_name, initargs["config"])

        self.sb3_callback = SB3CallbackDummy()
        self.leaf_keys = self._get_leaf_hyperparam_keys(self.hyperparam_bounds)
        self.reset_config(cfg)

        assert self.rollouts_per_iteration % self.logs_per_iteration == 0
        self.log_interval = self.rollouts_per_iteration // self.logs_per_iteration

        # Ensure both roles get equal amount of rollouts
        assert self.rollouts_per_iteration % self.rollouts_per_role == 0
        assert (self.rollouts_per_iteration // self.rollouts_per_role) % 2 == 0

        # Ensure there's equal amount of logs for each role
        assert self.rollouts_per_role % self.log_interval == 0

        assert self.rollouts_per_iteration % self.maps_per_iteration == 0
        self.rollouts_per_map = self.rollouts_per_iteration // self.maps_per_iteration

        assert self.rollouts_per_map % self.rollouts_per_role == 0
        assert (self.rollouts_per_map // self.rollouts_per_role) % 2 == 0

        # The model and env should be inited here
        # However, self.iteration is wrong (always 0) in setup()
        # => do lazy init in step()
        self.model = None
        self.initial_side = "attacker"

    def reset_config(self, cfg):
        cfg = deepmerge(copy.deepcopy(self.all_params), cfg)
        self.cfg = self._fix_floats(copy.deepcopy(cfg), self.hyperparam_bounds)
        steps_per_rollout = cfg["learner_kwargs"]["n_steps"]
        self.total_timesteps_per_iteration = steps_per_rollout * self.rollouts_per_iteration
        self.total_timesteps_per_role = steps_per_rollout * self.rollouts_per_role
        return True

    def save_checkpoint(self, checkpoint_dir):
        f = os.path.join(checkpoint_dir, "model.zip")
        self.model.save(f)
        # self.log("(save): self.experiment_name: '%s', f: '%s'" % (self.experiment_name, f))
        # self.log("Saved %s" % f.split(self.experiment_name)[1])
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        f = os.path.join(checkpoint_dir, "model.zip")
        self.model = self._model_checkpoint_load(f)

    def step(self):
        if not self.model:
            # lazy init model - see note in setup()
            self.model = self._model_init()

        wlog = self._get_perturbed_config()
        wlog["trial/iteration"] = self.iteration

        # Commit will be done by the callback's first log
        wandb.log(wlog, commit=False)

        old_rollouts = self.sb3_callback.rollouts
        rollouts_this_iteration = 0
        side = self.initial_side
        ep_rew_means = []
        while rollouts_this_iteration < self.rollouts_per_iteration:
            if len(ep_rew_means) > 0:
                side = "attacker" if len(ep_rew_means) % 2 == 0 else "defender"
                self.model.env.close()
                self.model.env = self._venv_init(side, rollouts_this_iteration)
                self.model.env.reset()

            self.model.learn(
                total_timesteps=self.total_timesteps_per_role,
                log_interval=self.log_interval,
                reset_num_timesteps=(self.iteration == 0 and rollouts_this_iteration == 0),
                progress_bar=False,
                callback=self.sb3_callback
            )

            rollouts_this_iteration += self.rollouts_per_role
            ep_rew_means.append(self.sb3_callback.ep_rew_mean)

        diff_rollouts = self.sb3_callback.rollouts - old_rollouts
        iter_rollouts = self.rollouts_per_iteration

        assert diff_rollouts == iter_rollouts, f"expected {iter_rollouts}, got: {diff_rollouts}"

        # TODO: add net_value to result (to be saved as checkpoint metadata)
        report = {"rew_mean": statistics.mean(ep_rew_means)}
        return report

    def log_result(self, result):
        assert self.sb3_callback.uncommitted_log
        assert self.sb3_callback.wdb_tables
        wdb_log = dict(self.sb3_callback.wdb_tables)
        wdb_log["trial/time_this_iter_s"] = result["time_this_iter_s"]
        wandb.log(wdb_log)

    #
    # private
    #

    # def _wandb_init(self, experiment_name, config):
    #     wandb_init(self.trial_id, self.trial_name, experiment_name, config)

    def _venv_init(self, role, rollouts_this_iteration=0):
        env_kwargs = dict(self.cfg["env_kwargs"], attacker="StupidAI", defender="StupidAI")
        env_kwargs[role] = "MMAI_USER"
        mpool = self.all_params["map_pool"]

        offset = self.all_params["map_pool_offset_idx"]
        offset += self.maps_per_iteration * self.iteration

        mid = (offset + rollouts_this_iteration // self.rollouts_per_map) % len(mpool)
        env_kwargs["mapname"] = "ai/generated/%s" % (mpool[mid])
        mapnum = int(re.match(r".+?([0-9]+)\.vmap", env_kwargs["mapname"]).group(1))

        env_kwargs["actions_log_file"] = f"/tmp/{self.trial_id}-actions.log"

        wandb.log(
            # 0=attacker, 1=defender
            {"mapnum": mapnum, "role": ["attacker", "defender"].index(role)},
            commit=False
        )

        self.log("Env kwargs: %s" % env_kwargs)

        return make_vec_env(
            "VCMI-v0",
            n_envs=1,
            env_kwargs=env_kwargs,
            monitor_kwargs={"info_keywords": InfoDict.ALL_KEYS},
        )

    # If "optimizer_class" and/or "optimizer_kwargs" are given
    # directly in "policy_kwars", SB3 does not allow to load a model
    # with a another setup unless explicitly specific in `custom_objects`
    # => add them to "custom_objects" kwarg at load
    # If both are blank, "custom_objects" will be {} and whatever was saved
    # will be loaded.
    def _learner_kwargs_for_init(self):
        policy_kwargs = {"net_arch": self.cfg["net_arch"]}

        # Any custom features extractor is assumed to be a VcmiCNN-type policy
        if self.cfg["features_extractor"]:
            policy_kwargs["features_extractor_class"] = getattr(vcmi_gym, self.cfg["features_extractor"]["class_name"])
            policy_kwargs["features_extractor_kwargs"] = self.cfg["features_extractor"]["kwargs"]

        if self.cfg["optimizer"]:
            policy_kwargs["optimizer_class"] = getattr(torch.optim, self.cfg["optimizer"]["class_name"])
            policy_kwargs["optimizer_kwargs"] = self.cfg["optimizer"]["kwargs"]

        return dict(
            self.cfg["learner_kwargs"],
            policy="MlpPolicy",
            policy_kwargs=policy_kwargs
        )

    def _learner_kwargs_for_load(self):
        # PPO.load() compares the policy_kwargs at the top-level
        # with the ones stored in the .zip file and will blow up on mismatch.
        #
        # * if the network arch itself is different => error
        #   => net can NOT change
        # * if the optimizer class is different => danger (state load error?)
        #   => optimizer class can NOT change
        # * if the optimizer kwargs are different => nothing happens
        #   (SB3 loads entire optimizer state later, overwriting kwargs)
        #   => optimizer kwargs have NO effect
        #
        # => use only learner_kwargs, and explicitly update optimizer later
        return dict(self.cfg["learner_kwargs"])

    def _model_internal_load(self, f, venv, **learner_kwargs):
        return PPO.load(f, env=venv, **learner_kwargs)

    def _model_internal_init(self, venv, **learner_kwargs):
        return PPO(env=venv, **learner_kwargs)

    def _model_init(self):
        # at init, we are the origin
        origin = int(self.trial_id.split("_")[1])
        # => int("00002") => 2

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)
        model = None

        if self.initial_checkpoint:
            self.log("Load %s (initial)" % self.initial_checkpoint)
            model = self._model_load(self.initial_checkpoint)
        else:
            venv = self._venv_init(self.initial_side)
            model = self._model_internal_init(venv, **self._learner_kwargs_for_init())

        return model

    def _model_load(self, f):
        venv = self._venv_init(self.initial_side)
        model = self._model_internal_load(f, venv, **self._learner_kwargs_for_load())

        # Need to explicitly update optimizer settings after load
        optimizer = self.cfg.get("optimizer", {})

        if optimizer:
            klass = model.policy.optimizer.__class__.__name__
            # XXX: SB3 loads optimizer state from ZIP
            # There may be issues if the class is different
            assert optimizer["class_name"] == klass, "updating optimizer class is not supported"

            for (k, v) in optimizer.get("kwargs", {}).items():
                model.policy.optimizer.param_groups[0][k] = v

        return model

    def _model_checkpoint_load(self, f):
        # Checkpoint tracking: log the trial ID of the checkpoint we are restoring now
        relpath = re.match(fr".+/{self.experiment_name}/(.+)", f).group(1)
        # => "6e59d_00004/checkpoint_000038/model.zip"

        origin = int(re.match(r".+?_(\d+)/.+", relpath).group(1))
        # => int("00004") => 4

        wandb.log({"trial/checkpoint_origin": origin}, commit=False)
        self.log("Load %s (origin: %d)" % (relpath, origin))

        return self._model_load(f)

    def _get_leaf_hyperparam_keys(self, data):
        leaf_keys = []

        for key, value in data.items():
            if isinstance(value, dict):
                leaf_keys.extend(self._get_leaf_hyperparam_keys(value))
            else:
                leaf_keys.append(key)

        return leaf_keys

    # NOTE: This could be extracted from self.cfg
    #       However, extracting those values from the objects is more accurate
    #       and has proved useful in the past
    #
    # TODO: `clip_range` is returned as a schedule fn and logged as a string
    #       Need to figure out how to convert it to float
    def _get_perturbed_config(self):
        params = {}
        env = self.model.env.envs[0].unwrapped

        for name in self.leaf_keys:
            if name == "clip_range":
                # clip_range is stored as a constant_fn callable
                params[f"config/{name}"] = self.model.clip_range(1)
            elif hasattr(self.model, name):
                params[f"config/{name}"] = getattr(self.model, name)
            elif hasattr(env, name):
                params[f"config/{name}"] = getattr(env, name)
            elif name in self.model.policy.optimizer.param_groups[0]:
                value = self.model.policy.optimizer.param_groups[0][name]
                params[f"config/{name}"] = value
            else:
                raise Exception("Could not find value for %s" % name)

        return params

    # Needed as Tune passes `float32` objects,
    # but SB3 expects regular `float` objects (or int for some params)
    # Also needed for PBT as values can go out of bounds:
    # https://github.com/ray-project/ray/issues/5035
    def _fix_floats(self, cfg, hyperparam_bounds):
        for key, value in hyperparam_bounds.items():
            if isinstance(value, dict):
                assert key in cfg and isinstance(cfg[key], dict)
                self._fix_floats(cfg[key], value)
            else:
                if key in ["n_epochs", "n_steps", "reward_clip_mod"]:
                    cfg[key] = int(cfg[key])
                else:
                    cfg[key] = float(cfg[key])

                # XXX: PBT only
                if hasattr(value, "lower"):
                    cfg[key] = min(cfg[key], value.upper)
                    cfg[key] = max(cfg[key], value.lower)

        return cfg


class PBT_PPOTrainerTest(PB2_PPOTrainerTest):
    def setup(self, cfg, initargs):
        new_cfg = deepmerge(copy.deepcopy(initargs["config"]["all_params"]), cfg)
        new_initargs = copy.deepcopy(initargs)
        new_initargs["config"]["hyperparam_bounds"] = initargs["config"]["hyperparam_mutations"]
        super().setup(new_cfg, new_initargs)


class PBT_MPPOTrainerTest(PBT_PPOTrainerTest):
    def _model_internal_init(self, venv, **learner_kwargs):
        print("MaskablePPO(%s)" % learner_kwargs)
        return MaskablePPO(env=venv, **learner_kwargs)

    def _model_internal_load(self, f, venv, **learner_kwargs):
        return MaskablePPO.load(f, env=venv, **learner_kwargs)


if __name__ == "__main__":
    setup_args = ({'learner_kwargs': {}, 'optimizer': {'kwargs': {'weight_decay': 0.1}}},)
    setup_kwargs = {
        'initargs': {
            'config': {
                'wandb_project': 'vcmi',
                'results_dir': 'data',
                'population_size': 6,
                'target_ep_rew_mean': 300000,
                'initial_checkpoint': '/Users/simo/Projects/vcmi-gym/data/GEN-PBT-MPPO-20231227_001755/9f59b_00003/checkpoint_000001/model.zip',
                'perturbation_interval': 1,
                'rollouts_per_iteration': 2000,
                'logs_per_iteration': 200,
                'rollouts_per_role': 100,
                'maps_per_iteration': 5,
                'hyperparam_mutations': {
                    'learner_kwargs': {},
                    'optimizer': {
                        'kwargs': {'weight_decay': [0, 0.01, 0.1]}
                    }
                },
                'quantile_fraction': 0.25,
                'all_params': {
                    'learner_kwargs': {
                        'stats_window_size': 100,
                        'learning_rate': 0.00126,
                        'n_steps': 512,
                        'batch_size': 64,
                        'n_epochs': 10,
                        'gamma': 0.8425,
                        'gae_lambda': 0.8,
                        'clip_range': 0.4,
                        'normalize_advantage': True,
                        'ent_coef': 0.007,
                        'vf_coef': 0.6,
                        'max_grad_norm': 2.5
                    },
                    'optimizer': {'class_name': 'AdamW', 'kwargs': {'eps': 1e-05, 'weight_decay': 0.1}},
                    'net_arch': [],
                    'features_extractor': {
                        'class_name': 'VcmiNN',
                        'kwargs': {
                            'output_dim': 1024,
                            'activation': 'ReLU',
                            'layers': [
                                {'t': 'Conv2d', 'out_channels': 32, 'kernel_size': (1, 15), 'stride': (1, 15), 'padding': 0},
                                {'t': 'Conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                {'t': 'Conv2d', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'padding': 2}
                            ]
                        }
                    },
                    'env_kwargs': {
                        'max_steps': 1000,
                        'vcmi_loglevel_global': 'error',
                        'vcmi_loglevel_ai': 'error',
                        'vcmienv_loglevel': 'WARN',
                        'consecutive_error_reward_factor': -1,
                        'sparse_info': True
                    },
                    'map_pool_offset_idx': 0,
                    'map_pool': ['A01.vmap', 'A02.vmap', 'A03.vmap', 'A04.vmap', 'A05.vmap', 'A06.vmap', 'A07.vmap', 'A08.vmap', 'A09.vmap', 'A10.vmap']
                }
            },
            'experiment_name': 'GEN-PBT-MPPO-20231227_212512',
            'root_dir': '/Users/simo/Projects/vcmi-gym'
        }
    }

    reset_config_args = ({
        'learner_kwargs': {
            'stats_window_size': 100,
            'learning_rate': 0.00126,
            'n_steps': 512,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.8425,
            'gae_lambda': 0.8,
            'clip_range': 0.4,
            'normalize_advantage': True,
            'ent_coef': 0.007,
            'vf_coef': 0.6,
            'max_grad_norm': 2.5
        },
        'optimizer': {
            'class_name': 'AdamW', 'kwargs': {'eps': 1e-05, 'weight_decay': 0.1}
        },
        'net_arch': [],
        'features_extractor': {
            'class_name': 'VcmiNN',
            'kwargs': {
                'output_dim': 1024,
                'activation': 'ReLU',
                'layers': [
                    {'t': 'Conv2d', 'out_channels': 32, 'kernel_size': (1, 15), 'stride': (1, 15), 'padding': 0},
                    {'t': 'Conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                    {'t': 'Conv2d', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'padding': 2}
                ]
            }
        },
        'env_kwargs': {
            'max_steps': 1000,
            'vcmi_loglevel_global': 'error',
            'vcmi_loglevel_ai': 'error',
            'vcmienv_loglevel': 'WARN',
            'consecutive_error_reward_factor': -1,
            'sparse_info': True
        },
        'map_pool_offset_idx': 0,
        'map_pool': [
            'A01.vmap', 'A02.vmap', 'A03.vmap', 'A04.vmap', 'A05.vmap',
            'A06.vmap', 'A07.vmap', 'A08.vmap', 'A09.vmap', 'A10.vmap'
        ]
    },)
    reset_config_kwargs = {}

    t = PBT_MPPOTrainerTest()
    t.setup(*setup_args, **setup_kwargs)
    t.reset_config(*reset_config_args, **reset_config_kwargs)
    t.step()
