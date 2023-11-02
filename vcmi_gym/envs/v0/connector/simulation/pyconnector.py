import logging
# import asyncio
import threading
import numpy as np
import time

from build import connsimulator


class PyConnector():
    def __init__(self):
        self.state = "TODO"
        self.cppcb = "TODO"
        self.event = threading.Event()

    # cb to be called from VCMI client thread on init only
    def pycbinit(self, cppcb):
        logging.info("start")

        # XXX: Danger: SIGSERV?
        logging.info("Set shared vars: self.cppcb = %s" % cppcb)
        self.cppcb = cppcb

        logging.info("return")

    # cb to be called from VCMI client thread
    def pycb(self, state):
        logging.info("start")
        logging.info("state.getStr(): %s, state.getState(): %s" % (state.getStr(), state.getState()))

        logging.info("sleep(0.1)")
        time.sleep(0.1)

        # XXX: Danger: SIGSERV?
        logging.info("Set shared var: self.state = %s" % state)
        self.state = state

        logging.info("event.set()")
        self.event.set()

        logging.info("return")

    def start(self):
        logging.info("start")

        # np1 = np.array([1,2,3.0])
        # np2 = np.array([1,2,5.0])
        # t = threading.Thread(target=connsimulator.add_arrays, args=(np1, np2), daemon=True)
        # t.run()

        logging.info("Call connsimulator.start_vcmi(pycbinit, pycb)")
        connsimulator.start_vcmi(self.pycbinit, self.pycb)

        # wait until connsimulator.start_vcmi calls pycb, setting state and cppcb
        logging.info("event.wait()")
        self.event.wait()

        logging.info("return state")
        return self.state

    def act(self, action):
        logging.info("start: action=%s" % action)
        cppaction = connsimulator.Action(str(action), action)

        logging.info("event.clear()")
        self.event.clear()

        self.cppcb(cppaction)
        logging.info("event.wait()")

        self.event.wait()
        logging.info("state=%s" % self.state)

        return self.state
