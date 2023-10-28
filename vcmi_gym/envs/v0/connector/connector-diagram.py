import logging

from connector_diagram import gymenv


logging.basicConfig(
    format="[PY][%(filename)s] (%(funcName)s) %(message)s",
    level=logging.INFO
)


if __name__ == '__main__':
    logging.info("start")
    env = gymenv.VcmiEnv()
    logging.info("Initialization complete")
    logging.info("exit")
