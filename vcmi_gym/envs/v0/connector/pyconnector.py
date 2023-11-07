import logging
import threading
import time

# don't remove - it is implicitly required by pybind11
import numpy as np

from .build import cppconnector


class PyConnector():
    STATE_SIZE = cppconnector.get_state_size()
    ACTION_MAX = cppconnector.get_action_max()

    def __init__(self, mapname):
        self.state = "TODO"
        self.cppcb = "TODO"
        self.cppsyscb = "TODO"
        self.cppresetcb = "TODO"
        self.mapname = mapname
        self.event = threading.Event()

    # cb to be called from VCMI client thread on init only
    def pycbresetinit(self, cppresetcb):
        logging.info("start")
        logging.debug("Set shared vars: self.cppresetcb = %s" % cppresetcb)
        self.cppresetcb = cppresetcb
        logging.debug("return")

    # cb to be called from VCMI root thread on boot only
    def pycbsysinit(self, cppsyscb):
        logging.info("start")
        logging.debug("Set shared vars: self.cppsyscb = %s" % cppsyscb)
        self.cppsyscb = cppsyscb
        logging.debug("return")

    # cb to be called from VCMI client thread on init only
    def pycbinit(self, cppcb):
        logging.info("start")
        logging.debug("Set shared vars: self.cppcb = %s" % cppcb)
        self.cppcb = cppcb
        logging.debug("return")

    # cb to be called from VCMI client thread
    def pycb(self, result):
        logging.info("start")
        logging.debug("Set shared var: self.result = %s" % result)
        self.result = result
        logging.debug("event.set()")
        self.event.set()
        logging.debug("return")

    def start(self):
        logging.info("start")
        logging.debug("Call cppconnector.start_vcmi(...)")
        cppconnector.start(self.pycbresetinit, self.pycbsysinit, self.pycbinit, self.pycb, self.mapname)

        # wait until cppconnector.start_vcmi calls pycb, setting result and cppcb
        logging.debug("event.wait()")
        self.event.wait()

        logging.debug("return result")
        return self.result

    def act(self, action):
        logging.info("start: action=%s" % action)

        logging.debug("event.clear()")
        self.event.clear()

        logging.debug("self.cppcb(action)")
        self.cppcb(action)

        logging.debug("event.wait()")
        self.event.wait()

        logging.debug("result=%s" % self.result)

        return self.result

    def reset(self):
        logging.info("start")

        logging.debug("event.clear()")
        self.event.clear()

        logging.debug("self.cppresetcb()")
        self.cppresetcb()

        logging.debug("event.wait()")
        self.event.wait()

        logging.debug("result=%s" % self.result)

        return self.result
