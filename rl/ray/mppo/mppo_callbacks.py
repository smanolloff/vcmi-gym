from ray.rllib.algorithms.callbacks import DefaultCallbacks


class MPPO_Callbacks(DefaultCallbacks):
    #
    # Typical callback arguments:
    #   * env:             gym.SyncVectorEnv
    #   * env_index:       int
    #   * env_runner:      ray.rllib.env.env_runner.EnvRunner (or a SingleAgentEnvRunner)
    #   * episode:         ray.rllib.evaluation.Episode
    #   * metrics_logger:  ray.rllib.utils.metrics.metrics_logger.MetricsLogger
    #   * rl_module:       ray.rllib.core.rl_module.rl_module
    #

    def on_episode_end(self, env, episode, metrics_logger, **kwargs):
        last_info = episode.get_infos()[-1]

        # XXX: episode has no `custom_metrics` attribute
        #      (ray examples are wrong)
        # episode.custom_metrics["net_value"] = last_info["net_value"]
        # episode.custom_metrics["is_success"] = last_info["is_success"]
        # print(" **** net_value: %s ***" % last_info["net_value"])
        # metrics_logger.log_value("net_value", last_info["net_value"])
        # metrics_logger.log_value("is_success", last_info["is_success"])
