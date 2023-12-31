import multiprocessing
import ctypes
import numpy as np
import os
import signal
import atexit
from datetime import datetime

from . import log

from ..connector.build import connexport

N_NONHEX_ACTIONS = connexport.get_n_nonhex_actions()
N_HEX_ACTIONS = connexport.get_n_hex_actions()
N_ACTIONS = connexport.get_n_actions()

_STATE_SIZE = connexport.get_state_size()
N_STACK_ATTRS = connexport.get_n_stack_attrs()
N_HEX_ATTRS = connexport.get_n_hex_attrs()

NV_MIN = connexport.get_nv_min()
NV_MAX = connexport.get_nv_max()

STATE_SIZE_X = 15 * N_HEX_ATTRS
STATE_SIZE_Y = 11
STATE_SIZE_Z = 1

assert STATE_SIZE_X * STATE_SIZE_Y == _STATE_SIZE

ERRMAP = connexport.get_error_mapping()
ERRSIZE = len(ERRMAP)
ERRNAMES = [errname for (errname, _) in ERRMAP.values()]
ERRFLAGS = list(ERRMAP.keys())

PyState = ctypes.c_float * _STATE_SIZE
PyAction = ctypes.c_int16
PyActmask = ctypes.c_bool * N_ACTIONS


class PyRawResult(ctypes.Structure):
    _fields_ = [
        ("state", PyState),
        ("actmask", PyActmask),
        ("errmask", ctypes.c_ushort),
        ("side", ctypes.c_int),
        ("dmg_dealt", ctypes.c_int),
        ("dmg_received", ctypes.c_int),
        ("units_lost", ctypes.c_int),
        ("units_killed", ctypes.c_int),
        ("value_lost", ctypes.c_int),
        ("value_killed", ctypes.c_int),
        ("is_battle_over", ctypes.c_bool),
        ("is_victorious", ctypes.c_bool),
    ]


# Same as connector's P_Result, but with values converted to ctypes
class PyResult():
    def __init__(self, cres):
        self.state = np.ctypeslib.as_array(cres.state).reshape(STATE_SIZE_Z, STATE_SIZE_Y, STATE_SIZE_X)
        self.actmask = np.ctypeslib.as_array(cres.actmask)
        self.errmask = cres.errmask
        self.side = cres.side
        self.dmg_dealt = cres.dmg_dealt
        self.dmg_received = cres.dmg_received
        self.units_lost = cres.units_lost
        self.units_killed = cres.units_killed
        self.value_lost = cres.value_lost
        self.value_killed = cres.value_killed
        self.is_battle_over = cres.is_battle_over
        self.is_victorious = cres.is_victorious


class PyConnector():
    COMMAND_TYPE_RESET = 0
    COMMAND_TYPE_ACT = 1
    COMMAND_TYPE_RENDER_ANSI = 2

    #
    # MAIN PROCESS
    #
    def __init__(self, loglevel):
        self.started = False
        self.loglevel = loglevel
        self.shutdown_lock = multiprocessing.Lock()
        self.shutdown_complete = False

    def start(self, *args):
        assert not self.started, "Already started"
        self.started = True
        self.logger = log.get_logger("PyConnector", self.loglevel)

        self.v_action = multiprocessing.Value(PyAction)
        self.v_result_act = multiprocessing.Value(PyRawResult)
        self.v_result_render_ansi = multiprocessing.Array(ctypes.c_char, 8192)
        self.v_command_type = multiprocessing.Value(ctypes.c_int)

        # Process synchronization:
        # cond.notify() will wake up the other proc (which immediately tries to acquire the lock)
        # cond.wait() will release the lock and wait (other proc now successfully acquires the lock)
        self.cond = multiprocessing.Condition()
        self.cond.acquire()
        self.proc = multiprocessing.Process(
            target=self.start_connector,
            args=args,
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

    def reset(self):
        self.v_command_type.value = PyConnector.COMMAND_TYPE_RESET
        self.cond.notify()
        self.cond.wait()
        return PyResult(self.v_result_act)

    def act(self, action):
        self.v_command_type.value = PyConnector.COMMAND_TYPE_ACT
        self.v_action.value = action
        self.cond.notify()
        self.cond.wait()
        return PyResult(self.v_result_act)

    def render_ansi(self):
        self.v_command_type.value = PyConnector.COMMAND_TYPE_RENDER_ANSI
        self.cond.notify()
        self.cond.wait()
        return self.v_result_render_ansi.value.decode("utf-8")

    def _try_start(self, _retries, _retry_interval):
        self.proc.start()
        self.cond.wait()
        return True

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

        self.cond.acquire()
        self.set_v_result_act(self.__connector.start())
        self.cond.notify()

        while True:
            # release the lock and wait (main proc now successfully acquires the lock)
            self.cond.wait()
            # perform action only after main proc calls cond.notify() and cond.wait()
            self.process_command()
            # wake up the subproc (which immediately tries to acquire the lock)
            self.cond.notify()

    def set_v_result_act(self, result):
        self.v_result_act.state = np.ctypeslib.as_ctypes(result.get_state())
        self.v_result_act.actmask = np.ctypeslib.as_ctypes(result.get_actmask())
        self.v_result_act.errmask = ctypes.c_ushort(result.get_errmask())
        self.v_result_act.side = ctypes.c_int(result.get_side())
        self.v_result_act.dmg_dealt = ctypes.c_int(result.get_dmg_dealt())
        self.v_result_act.dmg_received = ctypes.c_int(result.get_dmg_received())
        self.v_result_act.units_lost = ctypes.c_int(result.get_units_lost())
        self.v_result_act.units_killed = ctypes.c_int(result.get_units_killed())
        self.v_result_act.value_lost = ctypes.c_int(result.get_value_lost())
        self.v_result_act.value_killed = ctypes.c_int(result.get_value_killed())
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
