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
        self.actioncb = "TODO"
        self.syscb = "TODO"
        self.resetcb = "TODO"
        self.mapname = mapname
        self.event = threading.Event()

    # cb to be called from VCMI client thread on init only
    def resetcbcb(self, resetcb):
        logging.info("start")
        logging.debug("Set shared vars: self.resetcb = %s" % resetcb)
        self.resetcb = resetcb
        logging.debug("return")

    # cb to be called from VCMI root thread on boot only
    def syscbcb(self, syscb):
        logging.info("start")
        logging.debug("Set shared vars: self.syscb = %s" % syscb)
        self.syscb = syscb
        logging.debug("return")

    # cb to be called from VCMI client thread on init only
    def actioncbcb(self, actioncb):
        logging.info("start")
        logging.debug("Set shared vars: self.actioncb = %s" % actioncb)
        self.actioncb = actioncb
        logging.debug("return")

    # cb to be called from VCMI client thread
    def resultcb(self, result):
        logging.info("start")
        logging.debug("Set shared var: self.result = %s" % result)
        self.result = result
        logging.debug("event.set()")
        self.event.set()
        logging.debug("return")

    def start(self):
        logging.info("start")
        logging.debug("Call cppconnector.start_vcmi(...)")
        cppconnector.start(self.resetcbcb, self.syscbcb, self.actioncbcb, self.resultcb, self.mapname)

        # wait until cppconnector.start_vcmi calls resultcb, setting result and actioncb
        logging.debug("event.wait()")
        self.event.wait()

        logging.debug("return result")
        return self.result

    def act(self, action):
        logging.info("start: action=%s" % action)

        logging.debug("event.clear()")
        self.event.clear()

        logging.debug("self.actioncb(action)")
        self.actioncb(action)

        logging.debug("event.wait()")
        self.event.wait()

        logging.debug("result=%s" % self.result)

        return self.result

    def reset(self):
        logging.info("start")

        logging.debug("event.clear()")
        self.event.clear()

        logging.debug("self.resetcb()")
        self.resetcb()

        logging.debug("event.wait()")
        self.event.wait()

        logging.debug("result=%s" % self.result)

        return self.result

    # def render_ansi(self):
    #     logging.info("start")
