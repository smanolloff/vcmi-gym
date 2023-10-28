import logging

from . import pyconnector


class VcmiEnv():
    def __init__(self):
        logging.info("start")
        self.state, self.cppcb = pyconnector.start()
        logging.info("state.getA(): %s, state.getB(): %s" % (self.state.getA(), self.state.getB()))
        logging.info("return")

