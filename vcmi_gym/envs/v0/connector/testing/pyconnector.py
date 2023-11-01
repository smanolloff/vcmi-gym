import logging
# import asyncio
import threading
import numpy as np
import time

from build import connector


class PyConnector():
    def __init__(self):
        self.state = None
        self.cppcb = None
        self.event = threading.Event()

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
        logging.info("Create thread for connector.start_vcmi(pycbinit, pycb)")
        t = threading.Thread(target=connector.start_vcmi, args=(self.pycbinit, self.pycb), daemon=True)
        t.start()

        # wait until connector.start_vcmi calls pycb, setting state and cppcb
        logging.info("event.wait()")
        self.event.wait()

        logging.info("return state")
        return self.state

    def act(self, action):
        logging.info("start: action=%s" % action)
        cppaction = connector.Action()
        cppaction.set(action)
        logging.info("event.clear()")
        self.event.clear()
        self.cppcb(cppaction)

        # wait until the next activeStack() is called on the AI
        # which will call pycb
        # which will set this event
        logging.info("event.wait()")
        self.event.wait()
        logging.info("state=%s" % self.state)

        return self.state
