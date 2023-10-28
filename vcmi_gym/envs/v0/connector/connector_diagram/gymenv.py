import logging

from . import pyconnector


class VcmiEnv():
    def __init__(self):
        logging.info("start")
        self.state, self.cppcb = pyconnector.start()
        logging.info("obs: %s" % obs)
        logging.info("return")
