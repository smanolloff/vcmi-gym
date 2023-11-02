import logging
# import asyncio
import threading
import numpy as np
import time

from build import conntest


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

        # logging.info("<DEBUG> schedule restart in 10s...")
        # def debugtest():
        #     logging.info("sleep(10)")
        #     time.sleep(10)
        #     logging.info('Call self.cppsyscb("reset")')
        #     self.cppsyscb("reset")
        # threading.Thread(target=debugtest, daemon=True).start()

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

        logging.info("Call conntest.start_connector(pycbsysinit, pycbinit, pycb)")
        conntest.start_connector(self.pycbsysinit, self.pycbinit, self.pycb)

        # wait until conntest.start_vcmi calls pycb, setting state and cppcb
        logging.info("event.wait()")
        self.event.wait()

        logging.info("return state")
        return self.state

    def act(self, action):
        logging.info("start: action=%s" % action)
        cppaction = conntest.Action(str(action), action)

        logging.info("event.clear()")
        self.event.clear()

        self.cppcb(cppaction)
        logging.info("event.wait()")

        self.event.wait()
        logging.info("state=%s" % self.state)

        return self.state
