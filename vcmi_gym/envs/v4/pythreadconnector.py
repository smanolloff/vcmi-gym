# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import types
import os
import threading
import warnings
import random
from collections import OrderedDict

from ..util import log

if (os.getenv("VCMIGYM_DEBUG", None) == "1"):
    from ...connectors.build import exporter_v4
else:
    from ...connectors.rel import exporter_v4

EXPORTER = exporter_v4.Exporter()

N_NONHEX_ACTIONS = EXPORTER.get_n_nonhex_actions()
N_HEX_ACTIONS = EXPORTER.get_n_hex_actions()
N_ACTIONS = EXPORTER.get_n_actions()

STATE_SIZE = EXPORTER.get_state_size()
STATE_SIZE_HEXES = EXPORTER.get_state_size_hexes()
STATE_SIZE_STACKS = EXPORTER.get_state_size_stacks()
STATE_SIZE_ONE_HEX = EXPORTER.get_state_size_one_hex()
STATE_SIZE_ONE_STACK = EXPORTER.get_state_size_one_stack()
STATE_VALUE_NA = EXPORTER.get_state_value_na()

HEXATTRMAP = types.MappingProxyType(OrderedDict([(k, tuple(v)) for k, *v in EXPORTER.get_hex_attribute_mapping()]))
STACKATTRMAP = types.MappingProxyType(OrderedDict([(k, tuple(v)) for k, *v in EXPORTER.get_stack_attribute_mapping()]))

HEXACTMAP = types.MappingProxyType(OrderedDict([(action, i) for i, action in enumerate(EXPORTER.get_hexactions())]))
HEXSTATEMAP = types.MappingProxyType(OrderedDict([(state, i) for i, state in enumerate(EXPORTER.get_hexstates())]))
SIDEMAP = types.MappingProxyType(OrderedDict([("LEFT", EXPORTER.get_side_left()), ("RIGHT", EXPORTER.get_side_right())]))

TRACE = False
MAXLEN = 80


def tracelog(func, maxlen=MAXLEN):
    if not TRACE:
        return func

    def wrapper(*args, **kwargs):
        this = args[0]
        this.logger.debug("Begin: %s (args=%s, kwargs=%s)" % (func.__name__, args[1:], log.trunc(repr(kwargs), maxlen)))
        result = func(*args, **kwargs)
        this.logger.debug("End: %s (return %s)" % (func.__name__, log.trunc(repr(result), maxlen)))
        return result

    return wrapper


# Same as connector's P_Result, but with values converted to ctypes
# TODO: this class is redundant, return original result to VcmiEnv instead
class PyResult():
    def __init__(self, result):
        self.state = result.get_state()
        self.actmask = result.get_actmask()
        # self.attnmask = np.array(result.get_attnmask(), dtype=np.float32).reshape(165, 165)
        self.errcode = result.get_errcode()
        self.side = result.get_side()
        self.dmg_dealt = result.get_dmg_dealt()
        self.dmg_received = result.get_dmg_received()
        self.units_lost = result.get_units_lost()
        self.units_killed = result.get_units_killed()
        self.value_lost = result.get_value_lost()
        self.value_killed = result.get_value_killed()
        self.initial_side0_army_value = result.get_initial_side0_army_value()
        self.initial_side1_army_value = result.get_initial_side1_army_value()
        self.current_side0_army_value = result.get_current_side0_army_value()
        self.current_side1_army_value = result.get_current_side1_army_value()
        self.is_battle_over = result.get_is_battle_over()
        self.is_victorious = result.get_is_victorious()


class PyThreadConnector():
    #
    # MAIN PROCESS
    #
    def __init__(self, loglevel, maxlogs, user_timeout, vcmi_timeout, boot_timeout, allow_retreat):
        self.loglevel = loglevel
        self.maxlogs = maxlogs
        self.boot_timeout = boot_timeout or 99999
        self.vcmi_timeout = vcmi_timeout or 99999
        self.user_timeout = user_timeout or 99999
        self.logger = log.get_logger("PyThreadConnector", self.loglevel)
        self.startlock = threading.RLock()
        self.termlock = threading.RLock()
        self.starting = threading.Event()
        self.shutting_down = threading.Event()

    @tracelog
    def start(self, *args):
        with self.startlock:
            if self.starting.is_set():
                warnings.warn("PyThreadConnector is already started")
                return

            self.thread = threading.Thread(
                target=self.start_connector,
                args=args,
                name="VCMI",
                daemon=True
            )

            self.thread.start()
            self.starting.wait()

    def shutdown(self):
        warnings.warn("using 'thread' connector does not support env shutdown")
        if self.shutting_down.is_set():
            return

        self.shutting_down.set()

        # proc may not have been started at all
        # it is also already joined by deathwatch, but joining it again is ok
        if self.thread:
            try:
                if self.thread.is_alive():
                    self._connector.shutdown()
                self.thread.join()
            except Exception as e:
                self.logger.error("Could not join self.thread: %s" % str(e))

        # attempt to prevent log duplication from ray PB2 training
        # Close logger last (because of the "Resources released" message)
        self.logger.info("Resources released")
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()
        del self.logger

    @tracelog
    def reset(self):
        try:
            assert self.thread.is_alive(), "VCMI thread is dead."
            code, res = self._connector.reset(self.side)
            assert code == 0, "bad return code: %s" % code
            return PyResult(res)
        except Exception as e:
            self.logger.error(f"Exception caught: {str(e)}\nConnector log dump:")
            for msg in self._connector.getLogs():
                print(msg)
            raise

    @tracelog
    def step(self, action):
        try:
            assert self.thread.is_alive(), "VCMI thread is dead."
            code, res = self._connector.step(self.side, action)
            assert code == 0, "bad return code: %s" % code
            return PyResult(res)
        except Exception as e:
            self.logger.error(f"Exception caught: {str(e)}\nConnector log dump:")
            for msg in self._connector.getLogs():
                print(msg)
            raise

    @tracelog
    def render(self):
        try:
            assert self.thread.is_alive(), "VCMI thread is dead."
            code, res = self._connector.render(self.side)
            assert code == 0, "bad return code: %s" % code
            return res
        except Exception as e:
            self.logger.error(f"Exception caught: {str(e)}\nConnector log dump:")
            for msg in self._connector.getLogs():
                print(msg)
            raise

    @tracelog
    def connect_as(self, perspective):
        try:
            assert self.thread.is_alive(), "VCMI thread is dead."
            self.side = 0 if perspective == "attacker" else 1
            code, res = self._connector.connect(self.side)
            assert code == 0, "bad return code: %s" % code
            return PyResult(res)
        except Exception as e:
            self.logger.error(f"Exception caught: {str(e)}\nConnector log dump:")
            for msg in self._connector.getLogs():
                print(msg)
            raise

    #
    # This method is the SUB-THREAD
    # It never returns (enters infinite loop in vcmi's main function)
    #
    @tracelog
    def start_connector(self, *args):
        self.logger.info("Starting VCMI")

        if (os.getenv("VCMIGYM_DEBUG", None) == "1"):
            print("Using debug connector...")
            from ...connectors.build import connector_v4
        else:
            from ...connectors.rel import connector_v4

        self.logger.debug("VCMI connector args: %s" % str(args))
        self._connector = connector_v4.ThreadConnector(
            self.maxlogs,
            self.boot_timeout,
            self.vcmi_timeout,
            self.user_timeout,
            *args
        )

        self.starting.set()
        self._connector.start()
