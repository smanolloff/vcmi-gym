import time
import logging
import sys
from vcmi_gym.envs.v0.vcmi_env import VcmiEnv

logging.basicConfig(
    format="[PY][%(filename)s] (%(funcName)s) %(message)s",
    level=logging.INFO
)


def main():
    testmap = "simotest.vmap"
    if len(sys.argv) > 1:
        testmap = sys.argv[1]

    env = VcmiEnv(testmap)

    # enchanters:
    # stay on hex 75, shoot at enemy stack 1:
    action = 2 + 75*8 + 1
    while True:
        obs, rew, term, trunc, info = env.step(action)
        # obs, rew, term, trunc, info = env.step(0)
        logging.info("======== obs: (hidden)")
        logging.info("======== rew: %s" % rew)
        logging.info("======== term: %s" % term)
        logging.info("======== trunc: %s" % trunc)
        logging.info("======== info: %s" % info)
        action += 1

        if term:
            break

        # time.sleep(5)

    time.sleep(5)
    env.reset()
    env.step(0)
    env.step(1)  # reset needed!
