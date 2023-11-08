import time
import logging
import sys
from vcmi_gym.envs.v0.vcmi_env import VcmiEnv

logging.basicConfig(
    format="[PY][%(filename)s] (%(funcName)s) %(message)s",
    level=logging.WARN
)

# To prevent annoying ApplePersistenceIgnoreState message on macOS:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO

def main():
    testmap = "simotest.vmap"
    if len(sys.argv) > 1:
        testmap = sys.argv[1]

    env = VcmiEnv(testmap)

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
    time.sleep(5)
    env.reset()
    env.step(0)
    env.step(1)  # reset needed!
