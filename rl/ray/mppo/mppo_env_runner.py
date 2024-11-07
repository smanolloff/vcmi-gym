from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.utils.annotations import override


class MPPO_EnvRunner(SingleAgentEnvRunner):
    @override(SingleAgentEnvRunner)
    def __init__(self, *args, **kwargs):
        import ipdb; ipdb.set_trace()  # noqa
        super().__init__(*args, **kwargs)


class MPPO_EvalEnvRunner(SingleAgentEnvRunner):
    @override(SingleAgentEnvRunner)
    def __init__(self, *args, **kwargs):
        import ipdb; ipdb.set_trace()  # noqa
        super().__init__(*args, **kwargs)
