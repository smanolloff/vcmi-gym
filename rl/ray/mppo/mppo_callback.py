import re
import wandb
from datetime import datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_LEN_MEAN,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    FAULT_TOLERANCE_STATS,
    LEARNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
)

from ray.rllib.core.learner.learner import (
    DEFAULT_OPTIMIZER,
    ENTROPY_KEY,
    POLICY_LOSS_KEY,
    VF_LOSS_KEY,
)

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    LEARNER_RESULTS_CURR_KL_COEFF_KEY,
)

from ray.rllib.policy.sample_batch import (
    DEFAULT_POLICY_ID
)


class MPPO_Callback(DefaultCallbacks):
    #
    # on_algorithm_init arguments:
    #   * algorithm
    #   * metrics_logger:  ray.rllib.utils.metrics.metrics_logger.MetricsLogger
    @override(DefaultCallbacks)
    def on_algorithm_init(self, algorithm, **kwargs):
        self._wandb_init(algorithm)
        self._wandb_log_hyperparams(algorithm)

    #
    # on_episode_end arguments:
    #   * env:             gym.SyncVectorEnv
    #   * env_index:       int
    #   * env_runner:      ray.rllib.env.env_runner.EnvRunner (or a SingleAgentEnvRunner)
    #   * episode:         ray.rllib.evaluation.Episode
    #   * metrics_logger:  ray.rllib.utils.metrics.metrics_logger.MetricsLogger
    #   * rl_module:       ray.rllib.core.rl_module.rl_module
    #
    @override(DefaultCallbacks)
    def on_episode_end(self, metrics_logger, env, episode, env_runner, **kwargs):
        info = episode.get_infos()[-1]
        # print("episode with len=%d at %s!" % (episode.agent_steps(), datetime.isoformat(datetime.now())))

        # TODO: different metrics based on info
        #       (requires changes in VCMI and VcmiEnv)
        #   eval/all/open/...
        #   eval/all/fort/...
        #   eval/all/bank/...
        #   eval/pool/5k/open/...
        #   eval/pool/5k/fort/...
        #   eval/pool/5k/bank/...

        # NOTE: Metrics are automatically namespaced, e.g. under ["evaluation"]["env_runners"]
        metrics = {
            "user/net_value": info["net_value"],
            "user/is_success": info["is_success"]
        }

        # The window is for the central metric collector stream, not per-worker
        window = env_runner.config.metrics_num_episodes_for_smoothing
        metrics_logger.log_dict(metrics, reduce="mean", window=window)

    @override(DefaultCallbacks)
    def on_train_result(self, algorithm, metrics_logger, result, **kwargs):
        # XXX: `result` is the return value of MPPO_Algorithm.training_step()
        # XXX: Do NOT use metrics_logger.peek() (some metrics are reset there)

        eg = result[EVALUATION_RESULTS]
        e = result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS]
        ft = result[FAULT_TOLERANCE_STATS]

        to_log = {
            "eval/global/num_episodes": eg[NUM_EPISODES_LIFETIME],
            "eval/global/num_timesteps": eg[NUM_ENV_STEPS_SAMPLED_LIFETIME],
            "eval/net_value": e["user/net_value"],
            "eval/is_success": e["user/is_success"],
            "eval/ep_len_mean": e[EPISODE_LEN_MEAN],
            "eval/ep_rew_mean": e[EPISODE_RETURN_MEAN],
            "eval/num_episodes": e[NUM_EPISODES],
            "eval/num_timesteps": e[NUM_ENV_STEPS_SAMPLED],
            "remote/eval_healthy_workers": eg["num_healthy_workers"],
            "remote/eval_worker_inflight_reqs": eg["num_in_flight_async_reqs"],
            "remote/eval_worker_restarts": eg["num_remote_worker_restarts"],
            "remote/train_healthy_workers": ft["num_healthy_workers"],
            "remote/train_worker_inflight_reqs": ft["num_in_flight_async_reqs"],
            "remote/train_worker_restarts": ft["num_remote_worker_restarts"],
        }

        # Add tune metrics to the result (must be top-level)
        result["eval/net_value"] = to_log["eval/net_value"]
        result["eval/is_success"] = to_log["eval/is_success"]

        self.on_train_subresult(algorithm, result, commit=False)
        self.wandb_log(to_log, commit=True)

    # Custom callback added called by MPPO_Algorithm
    def on_train_subresult(self, algorithm, result, commit=True):
        l = result[LEARNER_RESULTS][DEFAULT_POLICY_ID]
        t = result[ENV_RUNNER_RESULTS]
        tg = result

        to_log = {
            "learn/entropy": l[ENTROPY_KEY],
            "learn/explained_var": l[LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY],
            "learn/kl_loss": l[LEARNER_RESULTS_KL_KEY],
            "learn/policy_loss": l[POLICY_LOSS_KEY],
            "learn/vf_loss": l[VF_LOSS_KEY],
            "learn/vf_loss_unclipped": l[LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY],
            "train/net_value": t["user/net_value"],
            "train/is_success": t["user/is_success"],
            "train/ep_len_mean": t[EPISODE_LEN_MEAN],
            "train/ep_rew_mean": t[EPISODE_RETURN_MEAN],
            "train/num_episodes": t[NUM_EPISODES],
            "train/num_timesteps": t[NUM_ENV_STEPS_SAMPLED],
            "train/global/num_episodes": tg[NUM_EPISODES_LIFETIME],
            "train/global/num_timesteps": tg[NUM_ENV_STEPS_SAMPLED_LIFETIME],
        }

        # Not present otherwise
        if algorithm.config.use_kl_loss:
            to_log["learn/kl_coef"] = l[LEARNER_RESULTS_CURR_KL_COEFF_KEY]

        if algorithm.config.log_gradients:
            to_log["learn/grad_"] = l[f"gradients_{DEFAULT_OPTIMIZER}_global_norm"]

        result["train/net_value"] = to_log["train/net_value"]
        result["train/is_success"] = to_log["train/is_success"]
        result["train/ep_rew_mean"] = to_log["train/ep_rew_mean"]

        self.wandb_log(to_log, commit=commit)

    #
    # private
    #

    def _wandb_init(self, algo):
        old_run_id = algo.config.user["wandb_old_run_id"]

        if old_run_id:
            assert re.match(r"^[0-9a-z]+_[0-9]+$", old_run_id), f"bad id to resume: {old_run_id}"
            run_id = "%s_%s" % (old_run_id.split("_")[0], algo.trial_id.split("_")[1])
            print("Will resume run as id %s (Trial ID is %s)" % (run_id, algo.trial_id))
        else:
            run_id = algo.trial_id
            print("Will start new run %s" % run_id)

        run_name = algo.trial_name
        if algo.trial_name != "default":
            run_name = "T%d" % int(algo.trial_name.split("_")[-1])

        algo.ns.run_id = run_id
        algo.ns.run_name = run_name

        if algo.config.user["wandb_project"]:
            wandb.init(
                project=algo.config.user["wandb_project"],
                group=algo.config.user["experiment_name"],
                id=algo.ns.run_id,
                name=algo.ns.run_name,
                resume="allow",
                reinit=True,
                allow_val_change=True,
                settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                config=algo.ns.master_config,
                sync_tensorboard=False,
            )

            self._wandb_add_watch()

            # TODO: check if "code saving" works as expected:
            #       https://docs.wandb.ai/guides/track/log/
            # self._wandb_log_code()  # superseded by wandb_log_git
            # self._wandb_log_git()  # superseded by "code saving" profile setting

            # For wandb.log, commit=True by default
            # for wandb_log, commit=False by default
            def wandb_log(*args, **kwargs):
                wandb.log(*args, **dict({"commit": False}, **kwargs))
        else:
            def wandb_log(*args, **kwargs):
                print("*** WANDB LOG AT %s: %s %s" % (datetime.isoformat(datetime.now()), args, kwargs))

        self.wandb_log = wandb_log

    def _wandb_add_watch(self, algo):
        # XXX: wandb.watch() caused issues during serialization in oldray scripts
        #      (it pollutes the model with non-serializable callbacks)
        #      ray's checkpointing may break due to this as well...
        assert algo.learner_group.is_local
        algo.learner_group.foreach_learner(lambda l: wandb.watch(
            l.module[DEFAULT_POLICY_ID].encoder.encoder,
            log="all",
            log_graph=True,
            log_freq=1000
        ))

    # XXX: There's already a wandb "code saving" profile setting
    #      It saves only requirements.txt and the git metadata which is enough
    #      See https://docs.wandb.ai/guides/track/log/
    # def _wandb_log_code(self):
    #     # https://docs.wandb.ai/ref/python/run#log_code
    #     # XXX: "path" is relative to `ray_root`
    #     this_file = pathlib.Path(__file__)
    #     ray_root = this_file.parent.parent.absolute()
    #     # TODO: log requirements.txt as well
    #     wandb.run.log_code(
    #         root=ray_root,
    #         include_fn=lambda path: path.endswith(".py"),
    #     )

    # XXX: There's already a wandb "code saving" profile setting
    #      See https://docs.wandb.ai/guides/track/log/
    # def _wandb_log_git(self):
    #     repo = pygit2.Repository(".")
    #     commit = str(repo.head.target)
    #     diff = repo.diff()  # By default, diffs unstaged changes
    #     if diff.stats.files_changed > 0:
    #         print("Patch:\n%s" % diff.patch)
    #     else:
    #         print("No diff")

    def _wandb_log_hyperparams(self, algo):
        for k, v in algo.config.user["hyperparam_mutations"].items():
            if k == "train":
                for k1, v1 in v:
                    assert hasattr(algo.config, k1), f"hasattr(algo.config, {k1})"
                    assert "/" not in k1
                    self.wandb_log(f"train/{k1}", getattr(algo.config, k1))
            if k == "env":
                for k1, v1 in v:
                    assert hasattr(algo.config.env_config, k1), f"hasattr(algo.config.env_config, {k1})"
                    assert "/" not in k1
                    self.wandb_log(f"env/{k1}", getattr(algo.config.env_config, k1))
