from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    EPISODE_LEN_MEAN,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    FAULT_TOLERANCE_STATS,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
)

from ray.rllib.utils.metrics.stats import Stats

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
    #
    @override(DefaultCallbacks)
    def on_algorithm_init(self, algorithm, **kwargs):
        pass

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
        #      + additional metrics added by Algorithm (eval, FT, etc.)
        # XXX: Do NOT use metrics_logger.peek() as some metrics are reset to 0

        eg = result[EVALUATION_RESULTS]
        e = result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS]
        ft = result[FAULT_TOLERANCE_STATS]

        to_log = {
            "eval/global/num_episodes": int(eg[NUM_EPISODES_LIFETIME]),
            "eval/global/num_timesteps": int(eg[NUM_ENV_STEPS_SAMPLED_LIFETIME]),

            "remote/eval_healthy_workers": int(eg["num_healthy_workers"]),
            "remote/eval_worker_inflight_reqs": int(eg["num_in_flight_async_reqs"]),
            "remote/eval_worker_restarts": int(eg["num_remote_worker_restarts"]),
            "remote/train_healthy_workers": float(ft["num_healthy_workers"]),
            "remote/train_worker_inflight_reqs": float(ft["num_in_flight_async_reqs"]),
            "remote/train_worker_restarts": float(ft["num_remote_worker_restarts"]),
        }

        # NOTE: eval metrics must be manually RESET
        eval_metrics = {
            "eval/net_value": "user/net_value",
            "eval/is_success": "user/is_success",
            "eval/ep_len_mean": EPISODE_LEN_MEAN,
            "eval/ep_rew_mean": EPISODE_RETURN_MEAN,
            "eval/num_episodes": NUM_EPISODES,
            "eval/num_timesteps": NUM_ENV_STEPS_SAMPLED,
        }

        estats = metrics_logger.stats[EVALUATION_RESULTS][ENV_RUNNER_RESULTS]
        for logname, statname in eval_metrics.items():
            to_log[logname] = e[statname]
            estats[statname] = Stats.similar_to(estats[statname])

        # Add tune metrics to the result (must be top-level)
        result["eval/net_value"] = to_log["eval/net_value"]
        result["eval/is_success"] = to_log["eval/is_success"]

        self.on_train_subresult(algorithm, result, commit=False)
        algorithm.wandb_log(to_log, commit=True)

    # Custom callback added called by MPPO_Algorithm
    def on_train_subresult(self, algorithm, result, commit=True):
        l = result[LEARNER_RESULTS][DEFAULT_POLICY_ID]
        t = result[ENV_RUNNER_RESULTS]
        tg = result

        # XXX: explicit type casts to convert Stat objects to numeric values
        to_log = {
            "learn/entropy": float(l[ENTROPY_KEY]),
            "learn/explained_var": float(l[LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY]),
            "learn/kl_loss": float(l[LEARNER_RESULTS_KL_KEY]),
            "learn/policy_loss": float(l[POLICY_LOSS_KEY]),
            "learn/vf_loss": float(l[VF_LOSS_KEY]),
            "learn/vf_loss_unclipped": float(l[LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY]),
            "learn/update_time_mean": float(result[TIMERS][LEARNER_UPDATE_TIMER]),
            "train/net_value": float(t["user/net_value"]),
            "train/is_success": float(t["user/is_success"]),
            "train/ep_len_mean": float(t[EPISODE_LEN_MEAN]),
            "train/ep_rew_mean": float(t[EPISODE_RETURN_MEAN]),
            "train/num_episodes": int(t[NUM_EPISODES]),
            "train/num_timesteps": int(t[NUM_ENV_STEPS_SAMPLED]),
            "train/global/num_episodes": int(tg[NUM_EPISODES_LIFETIME]),
            "train/global/num_timesteps": int(tg[NUM_ENV_STEPS_SAMPLED_LIFETIME]),
            "train/batch_sampling_time_mean": float(result[TIMERS][ENV_RUNNER_SAMPLING_TIMER]),
            "train/sync_weights_time_mean": float(result[TIMERS][SYNCH_WORKER_WEIGHTS_TIMER]),
        }

        # Not present otherwise
        if algorithm.config.use_kl_loss:
            to_log["learn/kl_coef"] = float(l[LEARNER_RESULTS_CURR_KL_COEFF_KEY])

        if algorithm.config.log_gradients:
            to_log["learn/grad_"] = float(l[f"gradients_{DEFAULT_OPTIMIZER}_global_norm"])

        result["train/net_value"] = float(to_log["train/net_value"])
        result["train/is_success"] = float(to_log["train/is_success"])
        result["train/ep_rew_mean"] = float(to_log["train/ep_rew_mean"])

        algorithm.wandb_log(to_log, commit=commit)
