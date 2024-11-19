from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override

from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    EPISODE_LEN_MEAN,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    FAULT_TOLERANCE_STATS,
    LEARNER_RESULTS,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
)

from ray.rllib.utils.metrics.stats import Stats

from ray.rllib.policy.sample_batch import (
    DEFAULT_POLICY_ID
)


class MDreamerV3_Callback(DefaultCallbacks):
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
        ft = result[FAULT_TOLERANCE_STATS]
        l = result[LEARNER_RESULTS][DEFAULT_POLICY_ID]
        t = result[ENV_RUNNER_RESULTS]
        tg = result

        # off-policy algos do not have train/ep_rew_mean
        # train/ep_rew_mean, train/net_value, etc. are needed as tune metrics
        # assert int(t["num_episodes"]) > 0, "at least 1 episode is needed for metrics (?)"

        to_log = {
            # XXX: DreamerV3 has MANY learn/ keys that could be logged
            "learn/total_loss": float(l["total_loss"]),
            "learn/global/num_timesteps": int(result[LEARNER_RESULTS][ALL_MODULES][NUM_ENV_STEPS_TRAINED]),

            # No much use in logging these -- there's too little env steps taken
            "train/ep_len_mean": float(t[EPISODE_LEN_MEAN]),
            "train/ep_rew_mean": float(t[EPISODE_RETURN_MEAN]),
            "train/net_value": float(t["user/net_value"]),
            "train/is_success": float(t["user/is_success"]),
            "train/num_episodes": int(t[NUM_EPISODES]),
            "train/num_timesteps": int(t[NUM_ENV_STEPS_SAMPLED]),

            "train/global/num_episodes": int(tg[NUM_EPISODES_LIFETIME]),
            "train/global/num_timesteps": int(tg[NUM_ENV_STEPS_SAMPLED_LIFETIME]),
            "train/sync_weights_time_mean": float(result[TIMERS][SYNCH_WORKER_WEIGHTS_TIMER]),
            "train/sample_time_mean": float(result[TIMERS]["sample"]),

            "remote/train_healthy_workers": float(ft["num_healthy_workers"]),
            "remote/train_worker_inflight_reqs": float(ft["num_in_flight_async_reqs"]),
            "remote/train_worker_restarts": float(ft["num_remote_worker_restarts"]),
        }

        if EVALUATION_RESULTS in result:
            eg = result[EVALUATION_RESULTS]
            e = result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS]

            to_log["eval/global/num_episodes"] = int(eg[NUM_EPISODES_LIFETIME])
            to_log["eval/global/num_timesteps"] = int(eg[NUM_ENV_STEPS_SAMPLED_LIFETIME])
            to_log["remote/eval_healthy_workers"] = int(eg["num_healthy_workers"])
            to_log["remote/eval_worker_inflight_reqs"] = int(eg["num_in_flight_async_reqs"])
            to_log["remote/eval_worker_restarts"] = int(eg["num_remote_worker_restarts"])

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

        algorithm.wandb_log(to_log, commit=True)
