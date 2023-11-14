import time
import logging
import sys
from vcmi_gym.envs.v0.vcmi_env import VcmiEnv

logging.basicConfig(
    format="[PY][%(filename)s] (%(funcName)s) %(message)s",
    level=logging.WARN
)

# NOTE (MacOS ONLY):
# To prevent annoying ApplePersistenceIgnoreState message:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO


def test(testmap, vcmi_loglevel):
    env = VcmiEnv(testmap, vcmi_loglevel=vcmi_loglevel)

    # enchanters:
    # stay on hex 75, shoot at enemy stack 1:
    action = 2 + 75*8 + 1
    while True:
        print(env.render())
        obs, rew, term, trunc, info = env.step(action)
        # obs, rew, term, trunc, info = env.step(0)
        logging.debug("======== obs: (hidden)")
        logging.debug("======== rew: %s" % rew)
        logging.debug("======== term: %s" % term)
        logging.debug("======== trunc: %s" % trunc)
        logging.debug("======== info: %s" % info)
        action += 1

        if term:
            break

        # time.sleep(5)

    print(env.render())
    print(env.error_summary())

    time.sleep(5)
    env.reset()
    env.step(0)
    env.step(1)  # reset needed!
