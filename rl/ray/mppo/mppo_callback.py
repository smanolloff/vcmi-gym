import wandb
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
    NUM_ENV_STEPS_TRAINED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
)

from ray.rllib.core.learner.learner import (
    ENTROPY_KEY,
    POLICY_LOSS_KEY,
    VF_LOSS_KEY,
)

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_CURR_KL_COEFF_KEY,
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)

from ray.rllib.policy.sample_batch import (
    DEFAULT_POLICY_ID
)

class MPPO_Callback(DefaultCallbacks):
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

    #
    # on_train_result arguments:
    #   * algorithm
    #   * metrics_logger
    #   * result:           result from Algorithm.train() call
    #
    @override(DefaultCallbacks)
    def on_train_result(self, algorithm, metrics_logger, result, **kwargs):
        # XXX: Do NOT use metrics_logger.peek() here,
        #      as some metrics are already reset to 0
        #      It can be used for user-metrics only (not auto-reset)

        l = result[LEARNER_RESULTS][DEFAULT_POLICY_ID]
        ft = result[FAULT_TOLERANCE_STATS]
        e = result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS]
        eg = result[EVALUATION_RESULTS]
        # Train results
        t = result[ENV_RUNNER_RESULTS]
        tg = result

        # These are not aggregated => manually log for aggregation
        # (use user/ prefix to avoid name collisions)
        user_metrics = {
            "mean": {
                "learn/entropy": l[ENTROPY_KEY],
                "learn/explained_var": l[LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY],
                "learn/kl_loss": l[LEARNER_RESULTS_KL_KEY],
                "learn/policy_loss": l[POLICY_LOSS_KEY],
                "learn/vf_loss": l[VF_LOSS_KEY],
                "learn/vf_loss_unclipped": l[LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY],
                "remote/train_healthy_workers": ft["num_healthy_workers"],
                "remote/eval_healthy_workers": eg["num_healthy_workers"],
                "remote/train_worker_inflight_reqs": ft["num_in_flight_async_reqs"],
                "remote/eval_worker_inflight_reqs": eg["num_in_flight_async_reqs"],
            },
            "sum": {
                "remote/train_worker_restarts": ft["num_remote_worker_restarts"],
                "remote/eval_worker_restarts": eg["num_remote_worker_restarts"],
            }
        }

        for k, v in user_metrics["mean"].items():
            metrics_logger.log_value(f"user/{k}", v, reduce="mean", window=algorithm.ns.log_interval)

        for k, v in user_metrics["sum"].items():
            metrics_logger.log_value(f"user/{k}", v, reduce="sum", clear_on_reduce=True)

        if algorithm.iteration % algorithm.ns.log_interval > 0 or algorithm.iteration == 0:
            return

        to_log = {
            "train/net_value": t["user/net_value"],
            "train/is_success": t["user/is_success"],
            "train/ep_len_mean": t[EPISODE_LEN_MEAN],
            "train/ep_rew_mean": t[EPISODE_RETURN_MEAN],
            "train/num_episodes": t[NUM_EPISODES],
            "train/num_timesteps": t[NUM_ENV_STEPS_SAMPLED],
            "train/global/num_episodes": tg[NUM_EPISODES_LIFETIME],
            "train/global/num_timesteps": tg[NUM_ENV_STEPS_SAMPLED_LIFETIME],
            "train/global/num_rollouts": tg[NUM_ENV_STEPS_TRAINED_LIFETIME],

            "eval/net_value": e["user/net_value"],
            "eval/is_success": e["user/is_success"],
            "eval/ep_len_mean": e[EPISODE_LEN_MEAN],
            "eval/ep_rew_mean": e[EPISODE_RETURN_MEAN],
            "eval/num_episodes": e[NUM_EPISODES],
            "eval/num_timesteps": e[NUM_ENV_STEPS_SAMPLED],
            "eval/global/num_episodes": eg[NUM_EPISODES_LIFETIME],
            "eval/global/num_timesteps": eg[NUM_ENV_STEPS_SAMPLED_LIFETIME],
        }

        for k, v in user_metrics["mean"].items():
            to_log[k] = metrics_logger.peek(f"user/{k}")

        for k, v in user_metrics["sum"].items():
            to_log[k] = metrics_logger.peek(f"user/{k}")

        if wandb.run:
            wandb.log(to_log)
        else:
            print(log)
