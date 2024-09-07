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
import threading
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
STATE_SIZE_MISC = EXPORTER.get_state_size_misc()
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

PyState = ctypes.c_float * STATE_SIZE
PyAction = ctypes.c_int
PyActmask = ctypes.c_bool * N_ACTIONS
PyAttnmask = ctypes.c_float * (165*165)

TRACE = False
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


# Same as connector's P_Result, but with values converted to ctypes
class PyResult():
    def __init__(self, cres):
        self.state = np.ctypeslib.as_array(cres.state)
        self.actmask = np.ctypeslib.as_array(cres.actmask)
        self.attnmask = np.ctypeslib.as_array(cres.attnmask).reshape(165, 165)
        self.errcode = cres.errcode
        self.side = cres.side
        self.dmg_dealt = cres.dmg_dealt
        self.dmg_received = cres.dmg_received
        self.units_lost = cres.units_lost
        self.units_killed = cres.units_killed
        self.value_lost = cres.value_lost
        self.value_killed = cres.value_killed
        self.initial_side0_army_value = cres.initial_side0_army_value
        self.initial_side1_army_value = cres.initial_side1_army_value
        self.current_side0_army_value = cres.current_side0_army_value
        self.current_side1_army_value = cres.current_side1_army_value
        self.is_battle_over = cres.is_battle_over
        self.is_victorious = cres.is_victorious


class PyProcConnector():
    COMMAND_TYPE_UNSET = -1
    COMMAND_TYPE_RESET = 0
    COMMAND_TYPE_ACT = 1
    COMMAND_TYPE_RENDER_ANSI = 2

    # Needs to be overwritten by subclasses => define in PyProcConnector body
    class PyRawState(ctypes.Structure):
        _fields_ = [
            ("state", PyState),
            ("actmask", PyActmask),
            ("attnmask", PyAttnmask),
            ("errcode", ctypes.c_int),
            ("side", ctypes.c_int),
            ("dmg_dealt", ctypes.c_int),
            ("dmg_received", ctypes.c_int),
            ("units_lost", ctypes.c_int),
            ("units_killed", ctypes.c_int),
            ("value_lost", ctypes.c_int),
            ("value_killed", ctypes.c_int),
            ("initial_side0_army_value", ctypes.c_int),
            ("initial_side1_army_value", ctypes.c_int),
            ("current_side0_army_value", ctypes.c_int),
            ("current_side1_army_value", ctypes.c_int),
            ("is_battle_over", ctypes.c_bool),
            ("is_victorious", ctypes.c_bool),
        ]

    #
    # MAIN PROCESS
    #
    def __init__(self, loglevel, maxlogs, user_timeout, vcmi_timeout, boot_timeout, allow_retreat):
        self.started = False
        self.loglevel = loglevel
        self.maxlogs = maxlogs
        self.shutdown_lock = multiprocessing.Lock()
        self.shutting_down = multiprocessing.Event()
        self.logger = log.get_logger("PyProcConnector", self.loglevel)
        self.allow_retreat = allow_retreat

        # use "or" to catch zeros
        self.user_timeout = user_timeout or 999999
        self.vcmi_timeout = vcmi_timeout or 999999
        self.boot_timeout = boot_timeout or 999999

    @classmethod
    def deathwatch(cls, proc, cond, logger, shutting_down):
        proc.join()  # Wait for the process to complete

        try:
            proc.close()  # shutdown's close does not seem to work if proc is joined here
        except Exception:
            pass  # AttributeError: _sentinel

        if not shutting_down.is_set():
            logger.error("VCMI process has died")

        # VCMI process is has died, most likely while holding the lock
        # for some reason this prevents cond.wait(timeout) in `self._wait()`
        # process from working (hangs forever)
        # However, releasing the lock here un-hangs the wait above.
        # (...NOTE: does not work if monitor_process is an instance method?!)
        # Not sure what's going on, but this resolves the issue.
        try:
            cond._lock.release()
        except ValueError:
            # ValueError: semaphore or lock released too many times
            # i.e. lock was not owned; just ignore it
            pass

    @tracelog
    def start(self, *args):
        assert not self.started, "Already started"
        self.started = True

        self.v_action = multiprocessing.Value(PyAction)
        self.v_result_render_ansi = multiprocessing.Array(ctypes.c_char, 65536)
        self.v_command_type = multiprocessing.Value(ctypes.c_int)
        self.v_command_type.value = PyProcConnector.COMMAND_TYPE_UNSET
        self.v_result_act = multiprocessing.Value(self.__class__.PyRawState)

        # Process synchronization:
        # cond.notify() will wake up the other proc (which immediately tries to acquire the lock)
        # cond.wait() will release the lock and wait (other proc now successfully acquires the lock)
        self.cond = multiprocessing.Condition(multiprocessing.Lock())
        self.proc = multiprocessing.Process(
            target=self.start_connector,
            args=args,
            name="VCMI",
            daemon=True
        )

        atexit.register(self.shutdown)

        # Boot time is <1s, 6 simultaneously procs should not take <10s
        if not self._try_start(100, 0.1):
            # try once again after releasing
            if not self._try_start(1, 0):
                raise Exception("Failed to acquire semaphore")

        return PyResult(self.v_result_act)

    def shutdown(self):
        if self.shutting_down.is_set():
            return

        with self.shutdown_lock:
            self.shutting_down.set()

        self.logger.info("Terminating VCMI PID=%s" % self.proc.pid)

        # proc may not have been started at all
        # it is also already joined by deathwatch, but joining it again is ok
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.join()
                self.proc.close()
            except ValueError:
                # most likely already closed by deathwatch
                pass
            except Exception as e:
                self.logger.warn("Could not close self.proc: %s" % str(e))

        if getattr(self, "deathwatch_thread", None):
            try:
                self.deathwatch_thread.join()
            except Exception as e:
                self.logger.warn("Could not join deathwatch_thread: %s" % str(e))

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
        del self.deathwatch_thread

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
            self.v_command_type.value = PyProcConnector.COMMAND_TYPE_RESET
            self.cond.notify()
            self._wait_vcmi()
        return PyResult(self.v_result_act)

    @tracelog
    def step(self, action):
        with self.cond:
            self.v_command_type.value = PyProcConnector.COMMAND_TYPE_ACT
            self.v_action.value = action
            self.cond.notify()
            self._wait_vcmi()
        return PyResult(self.v_result_act)

    @tracelog
    def render(self):
        with self.cond:
            self.v_command_type.value = PyProcConnector.COMMAND_TYPE_RENDER_ANSI
            self.cond.notify()
            self._wait_vcmi()
        return self.v_result_render_ansi.value.decode("utf-8")

    def _try_start(self, _retries, _retry_interval):
        with self.cond:
            self.proc.start()

            # XXX: deathwatch must be started AFTER self.proc.start()
            #      (they are not pickle-able, and won't exist in sub-proc)
            self.deathwatch_thread = threading.Thread(
                target=self.__class__.deathwatch,
                args=(self.proc, self.cond, self.logger, self.shutting_down),
                daemon=True
            )
            self.deathwatch_thread.start()

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

        self.logger = log.get_logger("PyProcConnector-sub", self.loglevel)

        self.logger.info("Starting VCMI")
        self.logger.debug("VCMI connector args: %s" % str(args))
        # XXX: self.__connector is ONLY available in the sub-process!
        self.__connector = self._get_connector().ProcConnector(self.maxlogs, *args)

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

    # NOTE: import is done at runtime to ensure VCMI inits
    #       in the newly created process
    def _get_connector(self):
        if (os.getenv("VCMIGYM_DEBUG", None) == "1"):
            print("Using debug connector...")
            from ...connectors.build import connector_v4
        else:
            from ...connectors.rel import connector_v4

        return connector_v4

    def set_v_result_act(self, result):
        self.v_result_act.state = np.ctypeslib.as_ctypes(result.get_state())
        self.v_result_act.actmask = np.ctypeslib.as_ctypes(result.get_actmask())
        # self.v_result_act.attnmask = np.ctypeslib.as_ctypes(np.array(result.get_attnmask(), dtype=np.float32))
        self.v_result_act.errcode = ctypes.c_int(result.get_errcode())
        self.v_result_act.side = ctypes.c_int(result.get_side())
        self.v_result_act.dmg_dealt = ctypes.c_int(result.get_dmg_dealt())
        self.v_result_act.dmg_received = ctypes.c_int(result.get_dmg_received())
        self.v_result_act.units_lost = ctypes.c_int(result.get_units_lost())
        self.v_result_act.units_killed = ctypes.c_int(result.get_units_killed())
        self.v_result_act.value_lost = ctypes.c_int(result.get_value_lost())
        self.v_result_act.value_killed = ctypes.c_int(result.get_value_killed())
        self.v_result_act.initial_side0_army_value = ctypes.c_int(result.get_initial_side0_army_value())
        self.v_result_act.initial_side1_army_value = ctypes.c_int(result.get_initial_side1_army_value())
        self.v_result_act.current_side0_army_value = ctypes.c_int(result.get_current_side0_army_value())
        self.v_result_act.current_side1_army_value = ctypes.c_int(result.get_current_side1_army_value())
        self.v_result_act.is_battle_over = ctypes.c_bool(result.get_is_battle_over())
        self.v_result_act.is_victorious = ctypes.c_bool(result.get_is_victorious())

        if not self.allow_retreat:
            self.v_result_act.actmask[0] = False

    def process_command(self):
        match self.v_command_type.value:
            case PyProcConnector.COMMAND_TYPE_ACT:
                self.set_v_result_act(self.__connector.step(self.v_action.value))
            case PyProcConnector.COMMAND_TYPE_RESET:
                self.set_v_result_act(self.__connector.reset())
            case PyProcConnector.COMMAND_TYPE_RENDER_ANSI:
                self.v_result_render_ansi.value = bytes(self.__connector.render(), 'utf-8')
            case _:
                raise Exception("Unknown command: %s" % self.v_command_type.value)

    def ignore_signal(self, _sig, _frame):
        pass

    def shutdown_proc(self):
        try:
            self.cond.release()
        except Exception:
            pass
