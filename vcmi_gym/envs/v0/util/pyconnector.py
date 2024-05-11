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

import multiprocessing
import ctypes
import types
import numpy as np
import os
import signal
import atexit
from collections import OrderedDict

from . import log

from ..connector.build import connexport

N_NONHEX_ACTIONS = connexport.get_n_nonhex_actions()
N_HEX_ACTIONS = connexport.get_n_hex_actions()
N_ACTIONS = connexport.get_n_actions()

STATE_SIZE_DEFAULT = connexport.get_state_size_default()
STATE_SIZE_DEFAULT_ONE_HEX = connexport.get_state_size_default_one_hex()
STATE_SIZE_FLOAT = connexport.get_state_size_float()
STATE_SIZE_FLOAT_ONE_HEX = connexport.get_state_size_float_one_hex()
STATE_VALUE_NA = connexport.get_state_value_na()
STATE_ENCODING_DEFAULT = connexport.get_encoding_type_default()
STATE_ENCODING_FLOAT = connexport.get_encoding_type_float()

ATTRMAP_DEFAULT = types.MappingProxyType(OrderedDict([(k, tuple(v)) for k, *v in connexport.get_attribute_mapping(STATE_ENCODING_DEFAULT)]))
ATTRMAP_FLOAT = types.MappingProxyType(OrderedDict([(k, tuple(v)) for k, *v in connexport.get_attribute_mapping(STATE_ENCODING_FLOAT)]))

HEXACTMAP = types.MappingProxyType(OrderedDict([(action, i) for i, action in enumerate(connexport.get_hexactions())]))
HEXSTATEMAP = types.MappingProxyType(OrderedDict([(state, i) for i, state in enumerate(connexport.get_hexstates())]))
DMGMODMAP = types.MappingProxyType(OrderedDict([(mod, i) for i, mod in enumerate(connexport.get_dmgmods())]))
SHOOTDISTMAP = types.MappingProxyType(OrderedDict([(dist, i) for i, dist in enumerate(connexport.get_shootdistances())]))
MELEEDISTMAP = types.MappingProxyType(OrderedDict([(dist, i) for i, dist in enumerate(connexport.get_meleedistances())]))
SIDEMAP = types.MappingProxyType(OrderedDict([("LEFT", connexport.get_side_left()), ("RIGHT", connexport.get_side_right())]))

ERRMAP = connexport.get_error_mapping()
ERRSIZE = len(ERRMAP)
ERRNAMES = [errname for (errname, _) in ERRMAP.values()]
ERRFLAGS = list(ERRMAP.keys())

PyStateDefault = ctypes.c_float * STATE_SIZE_DEFAULT
PyStateFloat = ctypes.c_float * STATE_SIZE_FLOAT
PyAction = ctypes.c_int
PyActmask = ctypes.c_bool * N_ACTIONS
PyAttnmasks = ctypes.c_float * (165*165)

TRACE = True
MAXLEN = 80


class UserTimeout(Exception):
    pass


class VcmiTimeout(Exception):
    pass


class BootTimeout(Exception):
    pass


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


COMMON_FIELDS = [
    ("actmask", PyActmask),
    ("attnmasks", PyAttnmasks),
    ("errmask", ctypes.c_ushort),
    ("side", ctypes.c_int),
    ("dmg_dealt", ctypes.c_int),
    ("dmg_received", ctypes.c_int),
    ("units_lost", ctypes.c_int),
    ("units_killed", ctypes.c_int),
    ("value_lost", ctypes.c_int),
    ("value_killed", ctypes.c_int),
    ("side0_army_value", ctypes.c_int),
    ("side1_army_value", ctypes.c_int),
    ("is_battle_over", ctypes.c_bool),
    ("is_victorious", ctypes.c_bool),
]


class PyRawResultDefault(ctypes.Structure):
    _fields_ = [("state", PyStateDefault)] + COMMON_FIELDS


class PyRawResultFloat(ctypes.Structure):
    _fields_ = [("state", PyStateFloat)] + COMMON_FIELDS


# Same as connector's P_Result, but with values converted to ctypes
class PyResult():
    def __init__(self, cres):
        self.state = np.ctypeslib.as_array(cres.state).reshape(11, 15, -1)
        self.actmask = np.ctypeslib.as_array(cres.actmask)
        self.attnmasks = np.ctypeslib.as_array(cres.attnmasks).reshape(165, 165)
        self.errmask = cres.errmask
        self.side = cres.side
        self.dmg_dealt = cres.dmg_dealt
        self.dmg_received = cres.dmg_received
        self.units_lost = cres.units_lost
        self.units_killed = cres.units_killed
        self.value_lost = cres.value_lost
        self.value_killed = cres.value_killed
        self.side0_army_value = cres.side0_army_value
        self.side1_army_value = cres.side1_army_value
        self.is_battle_over = cres.is_battle_over
        self.is_victorious = cres.is_victorious


class PyConnector():
    COMMAND_TYPE_UNSET = -1
    COMMAND_TYPE_RESET = 0
    COMMAND_TYPE_ACT = 1
    COMMAND_TYPE_RENDER_ANSI = 2

    #
    # MAIN PROCESS
    #
    def __init__(self, loglevel, user_timeout, vcmi_timeout, boot_timeout):
        self.started = False
        self.loglevel = loglevel
        self.shutdown_lock = multiprocessing.Lock()
        self.shutdown_complete = False
        self.logger = log.get_logger("PyConnector", self.loglevel)
        # use "or" to catch zeros

        self.user_timeout = user_timeout or 999999
        self.vcmi_timeout = vcmi_timeout or 999999
        self.boot_timeout = boot_timeout or 999999

    @tracelog
    def start(self, encoding, *args):
        assert not self.started, "Already started"
        self.started = True

        self.v_action = multiprocessing.Value(PyAction)
        self.v_result_render_ansi = multiprocessing.Array(ctypes.c_char, 8192)
        self.v_command_type = multiprocessing.Value(ctypes.c_int)
        self.v_command_type.value = PyConnector.COMMAND_TYPE_UNSET

        if encoding == STATE_ENCODING_DEFAULT:
            self.v_result_act = multiprocessing.Value(PyRawResultDefault)
        elif encoding == STATE_ENCODING_FLOAT:
            self.v_result_act = multiprocessing.Value(PyRawResultFloat)
        else:
            raise Exception("Unexpected encoding: %s" % encoding)

        # Process synchronization:
        # cond.notify() will wake up the other proc (which immediately tries to acquire the lock)
        # cond.wait() will release the lock and wait (other proc now successfully acquires the lock)
        self.cond = multiprocessing.Condition()
        self.proc = multiprocessing.Process(
            target=self.start_connector,
            args=(encoding, *args),
            name="VCMI",
            daemon=True
        )

        atexit.register(self.shutdown)

        # Multiple VCMIs booting simultaneously is not OK
        # (they all write to the same files at boot)
        # Boot time is <1s, 6 simultaneously procs should not take <10s
        if not self._try_start(100, 0.1):
            # self.semaphore.release()

            # Is this needed?
            # (NOTE: it might raise posix_ipc.ExistentialError)
            # posix_ipc.unlink_semaphore(self.semaphore.name)

            # try once again after releasing
            if not self._try_start(1, 0):
                raise Exception("Failed to acquire semaphore")

        return PyResult(self.v_result_act)

    def shutdown(self):
        if self.shutdown_complete:
            return

        with self.shutdown_lock:
            self.shutdown_complete = True

        self.logger.info("Terminating VCMI PID=%s" % self.proc.pid)

        # proc may not have been started at all (semaphore failed to acquire)
        if self.proc:
            self.proc.terminate()
            self.proc.join()
            self.proc.close()

        try:
            self.cond.release()
        except Exception:
            pass

        # release all multiprocessing resources
        del self.shutdown_lock
        del self.v_action
        del self.v_result_act
        del self.v_result_render_ansi
        del self.v_command_type
        del self.cond
        del self.proc

        # attempt to prevent log duplication from ray PB2 training
        # Close logger last (because of the "Resources released" message)
        self.logger.info("Resources released")
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()
        del self.logger

    @tracelog
    def reset(self):
        with self.cond:
            self.v_command_type.value = PyConnector.COMMAND_TYPE_RESET
            self.cond.notify()
            self._wait_vcmi()
        return PyResult(self.v_result_act)

    @tracelog
    def act(self, action):
        with self.cond:
            self.v_command_type.value = PyConnector.COMMAND_TYPE_ACT
            self.v_action.value = action
            self.cond.notify()
            self._wait_vcmi()
        return PyResult(self.v_result_act)

    @tracelog
    def render_ansi(self):
        with self.cond:
            self.v_command_type.value = PyConnector.COMMAND_TYPE_RENDER_ANSI
            self.cond.notify()
            self._wait_vcmi()
        return self.v_result_render_ansi.value.decode("utf-8")

    def _try_start(self, _retries, _retry_interval):
        with self.cond:
            self.proc.start()
            # boot time may be long, add extra 10s
            self._wait_boot()
        return True

    def _wait_vcmi(self):
        self._wait("vcmi", self.vcmi_timeout, VcmiTimeout)

    def _wait_boot(self):
        self._wait("boot", self.boot_timeout, BootTimeout)

    def _wait(self, actor, timeout, err_cls):
        if not self.cond.wait(timeout):
            msg = "No response from %s for %s seconds (last command_type was: %d)" % (
                actor,
                timeout,
                self.v_command_type.value
            )

            self.logger.error(msg)
            raise err_cls(msg)

    #
    # This method is the SUB-PROCESS
    # It enters an infinite loop, waiting for commands
    #
    def start_connector(self, *args):
        # The sub-process is a daemon, it shouldn't handle SIGINT
        if os.name == "posix":
            signal.signal(signal.SIGINT, self.ignore_signal)

        atexit.register(self.shutdown_proc)

        # NOTE: import is done here to ensure VCMI inits
        #       in the newly created process
        from ..connector.build import connector

        self.logger = log.get_logger("PyConnector-sub", self.loglevel)

        self.logger.info("Starting VCMI")
        self.logger.debug("VCMI connector args: %s" % str(args))
        # XXX: self.__connector is ONLY available in the sub-process!
        self.__connector = connector.Connector(*args)

        with self.cond:
            self.set_v_result_act(self.__connector.start())
            self.cond.notify()

            # use boot_timeout for 1st user action
            # (as it may take longer, eg. until all vec envs are UP)
            self._wait("user (boot)", self.boot_timeout, BootTimeout)
            self.process_command()
            self.cond.notify()

            while True:
                # release the lock and wait (main proc now successfully acquires the lock)
                self._wait("user", self.user_timeout, UserTimeout)
                # perform action only after main proc calls cond.notify() and cond.wait()
                self.process_command()
                # wake up the subproc (which immediately tries to acquire the lock)
                self.cond.notify()

    def set_v_result_act(self, result):
        self.v_result_act.state = np.ctypeslib.as_ctypes(result.get_state())
        self.v_result_act.actmask = np.ctypeslib.as_ctypes(result.get_actmask())
        self.v_result_act.attnmasks = np.ctypeslib.as_ctypes(result.get_attnmasks())
        self.v_result_act.errmask = ctypes.c_ushort(result.get_errmask())
        self.v_result_act.side = ctypes.c_int(result.get_side())
        self.v_result_act.dmg_dealt = ctypes.c_int(result.get_dmg_dealt())
        self.v_result_act.dmg_received = ctypes.c_int(result.get_dmg_received())
        self.v_result_act.units_lost = ctypes.c_int(result.get_units_lost())
        self.v_result_act.units_killed = ctypes.c_int(result.get_units_killed())
        self.v_result_act.value_lost = ctypes.c_int(result.get_value_lost())
        self.v_result_act.value_killed = ctypes.c_int(result.get_value_killed())
        self.v_result_act.side0_army_value = ctypes.c_int(result.get_side0_army_value())
        self.v_result_act.side1_army_value = ctypes.c_int(result.get_side1_army_value())
        self.v_result_act.is_battle_over = ctypes.c_bool(result.get_is_battle_over())
        self.v_result_act.is_victorious = ctypes.c_bool(result.get_is_victorious())

    def process_command(self):
        match self.v_command_type.value:
            case PyConnector.COMMAND_TYPE_ACT:
                self.set_v_result_act(self.__connector.act(self.v_action.value))
            case PyConnector.COMMAND_TYPE_RESET:
                self.set_v_result_act(self.__connector.reset())
            case PyConnector.COMMAND_TYPE_RENDER_ANSI:
                self.v_result_render_ansi.value = bytes(self.__connector.renderAnsi(), 'utf-8')
            case _:
                raise Exception("Unknown command: %s" % self.v_command_type.value)

    def ignore_signal(self, _sig, _frame):
        pass

    def shutdown_proc(self):
        try:
            self.cond.release()
        except Exception:
            pass
