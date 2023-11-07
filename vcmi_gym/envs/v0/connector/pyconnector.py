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
        logging.info("Set shared vars: self.cppresetcb = %s" % cppresetcb)
        self.cppresetcb = cppresetcb
        logging.info("return")

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
    def pycb(self, result):
        logging.info("start")
        logging.info("Set shared var: self.result = %s" % result)
        self.result = result
        logging.info("event.set()")
        self.event.set()
        logging.info("return")

    def start(self):
        logging.info("start")
        logging.info("Call cppconnector.start_vcmi(...)")
        cppconnector.start(self.pycbresetinit, self.pycbsysinit, self.pycbinit, self.pycb, self.mapname)

        # wait until cppconnector.start_vcmi calls pycb, setting result and cppcb
        logging.info("event.wait()")
        self.event.wait()

        logging.info("return result")
        return self.result

    def act(self, action):
        logging.info("start: action=%s" % action)

        logging.info("event.clear()")
        self.event.clear()

        logging.info("self.cppcb(action)")
        self.cppcb(action)

        logging.info("event.wait()")
        self.event.wait()

        logging.info("result=%s" % self.result)

        return self.result

    def reset(self):
        logging.info("start")

        logging.info("event.clear()")
        self.event.clear()

        logging.info("self.cppresetcb()")
        self.cppresetcb()

        logging.info("event.wait()")
        self.event.wait()

        logging.info("result=%s" % self.result)

        return self.result
