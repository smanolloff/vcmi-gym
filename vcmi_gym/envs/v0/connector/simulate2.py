import logging
import numpy as np

from simulation2.vcmi_env import VcmiEnv


logging.basicConfig(
    format="[PY][%(filename)s] (%(funcName)s) %(message)s",
    level=logging.INFO
)


if __name__ == '__main__':
    logging.info("start")

    logging.info("Creating VcmiEnv")
    env = VcmiEnv()

    logging.info("Calling env.step([6,6,6])")
    env.step(np.array([6, 6, 6], dtype=np.float32))
