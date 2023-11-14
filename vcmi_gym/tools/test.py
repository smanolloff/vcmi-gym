import time
import logging
import sys
import os
import gymnasium as gym

# from vcmi_gym.envs.v0.vcmi_env import VcmiEnv

logging.basicConfig(
    format="[PY][%(filename)s] (%(funcName)s) %(message)s",
    level=logging.WARN
)

# NOTE (MacOS ONLY):
# To prevent annoying ApplePersistenceIgnoreState message:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO


# not needed - just sleeping after identically formatted prints is OK
def clean_render(env):
    print("\033c", end="")
    print(env.render())


def test(env_kwargs):
    # env = VcmiEnv(**env_kwargs)
    env = gym.make("local/VCMI-v0")
    env.reset()
    print(env.render())

    # enchanters:
    # stay on hex 75, shoot at enemy stack 1:
    action = 2 + 75*8 + 1
    while True:
        obs, rew, term, trunc, info = env.step(action)
        # obs, rew, term, trunc, info = env.step(0)
        # logging.debug("======== obs: (hidden)")
        # logging.debug("======== rew: %s" % rew)
        # logging.debug("======== term: %s" % term)
        # logging.debug("======== trunc: %s" % trunc)
        # logging.debug("======== info: %s" % info)
        action += 1

        if env.unwrapped.last_action_n_errors == 0:
            time.sleep(1)
            print(env.render())
        if term:
            break

    print(env.unwrapped.error_summary())
