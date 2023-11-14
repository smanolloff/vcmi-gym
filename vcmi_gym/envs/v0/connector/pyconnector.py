import logging
import threading
import time

# don't remove - it is implicitly required by pybind11
import numpy as np

from .build import cppconnector


class PyConnector():
    STATE_SIZE = cppconnector.get_state_size()
    N_ACTIONS = cppconnector.get_n_actions()
    ERROR_MAPPING = cppconnector.get_error_mapping()

    def __init__(self, mapname, vcmi_loglevel):
        logging.debug("begin")
        self.cppconn = cppconnector.create_cppconnector(mapname, vcmi_loglevel)
        self.cppconn.prestart()
        logging.debug("end")

    def start(self):
        logging.debug("begin")
        res = self.cppconn.start()
        logging.debug("end")
        return res

    def act(self, action):
        logging.debug("begin")
        res = self.cppconn.act(action)
        logging.debug("end")
        return res

    def reset(self):
        logging.debug("Return self.cppconn.reset()")
        return self.cppconn.reset()

    def render_ansi(self):
        logging.debug("Return self.cppconn.renderAnsi()")
        return self.cppconn.renderAnsi()
