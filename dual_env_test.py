import threading
import random
import vcmi_gym
import time
from vcmi_gym.envs.v0.dual_env import DualEnvController, DualEnvClient


class DummyEnv():
    def step(self, *args, **kwargs):
        print("*** DUMMY ENV: step(%s, %s)" % (args, kwargs))

    def reset(self, *args, **kwargs):
        print("*** DUMMY ENV: reset(%s, %s)" % (args, kwargs))


def loop_functor(controller, name):
    def loop_func():
        env = DualEnvClient(controller, name)
        obs, rew, term, trunc, info = None, None, True, None, None

        while True:
            if term or trunc:
                obs, info = env.reset()
                rew, term, trunc = None, None, None
            else:
                if random.randint(0, 10) < 5:
                    obs, info = env.reset()
                else:
                    indices = [i for i, v in enumerate(env.action_masks()) if v]
                    action = random.choice(indices)
                    obs, rew, term, trunc, info = env.step(action)

    return loop_func


if __name__ == "__main__":
    env = vcmi_gym.VcmiEnv(
        mapname="ai/generated/A05.vmap",
        vcmienv_loglevel="DEBUG",
        attacker="MMAI_USER",
        defender="MMAI_USER",
        vcmi_loglevel_global="error",  # vcmi loglevel
        vcmi_loglevel_ai="error",  # vcmi loglevel
    )
    # env = DummyEnv()
    controller = DualEnvController(env)

    t1 = threading.Thread(target=loop_functor(controller, "EnvClient1"))
    t2 = threading.Thread(target=loop_functor(controller, "EnvClient2"))

    t1.start()
    time.sleep(3)
    t2.start()
