import datetime
from ray.rllib.utils.annotations import override
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner


class BaseEnv(SingleAgentEnvRunner):
    @override(SingleAgentEnvRunner)
    def ping(self) -> str:
        # Unwrap to bypass OrderEnforcing
        # Also, render() should be thread-safe (connector uses a lock)
        return str(bool(self.env.envs[0].unwrapped.render()))

    def sample(self, *args, **kwargs):
        print("SAMPLE CALLED AT %s: %s %s" % (datetime.datetime.now().isoformat(), args, kwargs))
        res = super().sample(*args, **kwargs)
        print("SAMPLE RETURNED %s" % datetime.datetime.now().isoformat())
        return res


# XXX: Can't set custom actor prefixes (tried __repr__, .actor_name(), etc.)
# Ray just uses the class name as a prefix and that can't be changed.
# Still, the class name gives at least some context in the logs
# => use separate classes for eval and train


class TrainEnv(BaseEnv):
    pass


class EvalEnv(BaseEnv):
    pass
