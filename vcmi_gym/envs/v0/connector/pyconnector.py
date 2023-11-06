import logging
import threading
import time

# don't remove - it is implicitly required by pybind11
import numpy as np

from .build import cppconnector


class PyConnector():
    def __init__(self):
        self.state = "TODO"
        self.cppcb = "TODO"
        self.cppsyscb = "TODO"
        self.event = threading.Event()

    # cb to be called from VCMI root thread on boot only
    def pycbsysinit(self, cppsyscb):
        logging.info("start")
        logging.info("Set shared vars: self.cppsyscb = %s" % cppsyscb)
        self.cppsyscb = cppsyscb
        logging.info("return")

    # cb to be called from VCMI client thread on init only
    def pycbinit(self, cppcb):
        logging.info("start")
        logging.info("Set shared vars: self.cppcb = %s" % cppcb)
        self.cppcb = cppcb
        logging.info("return")

    # cb to be called from VCMI client thread
    def pycb(self, state):
        logging.info("start")
        logging.info("Set shared var: self.state = %s" % state)
        self.state = state
        logging.info("event.set()")
        self.event.set()
        logging.info("return")

    def start(self):
        logging.info("start")
        logging.info("Call cppconnector.start_vcmi(pycbinit, pycb)")
        cppconnector.start(self.pycbsysinit, self.pycbinit, self.pycb)

        # wait until cppconnector.start_vcmi calls pycb, setting state and cppcb
        logging.info("event.wait()")
        self.event.wait()

        logging.info("return state")
        return self.state

    def act(self, action):
        logging.info("start: action=%s" % action)

        logging.info("event.clear()")
        self.event.clear()

        logging.info("self.cppcb(action)")
        self.cppcb(action)

        logging.info("event.wait()")
        self.event.wait()

        logging.info("state=%s" % self.state)

        return self.state
