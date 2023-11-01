import logging
import numpy as np

from . pyconnector import PyConnector


class VcmiEnv():
    def __init__(self):
        logging.info("start")
        self.pc = PyConnector()
        self.state = self.pc.start()
        # logging.info("state.getA(): %s, state.getB(): %s" % (self.state.getA(), self.state.getB()))
        logging.info("return")

    def step(self, action):
        logging.info("start")
        self.state = self.pc.act(np.array([6, 6, 6], dtype=np.float32))
        # logging.info("state.getA(): %s, state.getB(): %s" % (self.state.getA(), self.state.getB()))
        logging.info("return")
