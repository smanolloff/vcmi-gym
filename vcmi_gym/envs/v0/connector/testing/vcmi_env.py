import logging
import numpy as np

from . pyconnector import PyConnector


class VcmiEnv():
    def __init__(self):
        logging.info("start")
        self.pc = PyConnector()
        self.state = self.pc.start()
        logging.info("state: %s" % self.state)
        logging.info("return")

    def step(self, action):
        logging.info("start")
        self.state = self.pc.act(action)
        logging.info("state: %s" % self.state)
        logging.info("return")
