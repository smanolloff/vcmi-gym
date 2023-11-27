import multiprocessing
import ctypes
import numpy as np
import os
import signal
# import time
# NOTE: There *will* be orphaned system resources when using ray
#       as it seems we can't perform proper cleaning up.
#       After a ray PB2 run, this is one way to manually cleanup:
#           ps aux | grep multiprocessing | awk '{print $2}' | xargs kill -15
# import posix_ipc
import atexit

from . import log

#
# NOTE: those are hard-coded here as there is no way
#       to obtain them without importing connector first
#

N_ACTIONS = 1323    # !!! SYNC with aitypes.h !!!
STATE_SIZE = 306    # !!! SYNC with aitypes.h !!!
ERRSIZE = 9         # !!! SYNC with aitypes.h !!!
ERRNAMES = [        # !!! SYNC with aitypes.h !!!
    "ERR_ALREADY_WAITED",
    "ERR_MOVE_SELF",
    "ERR_HEX_UNREACHABLE",
    "ERR_HEX_BLOCKED",
    "ERR_STACK_NA",
    "ERR_STACK_DEAD",
    "ERR_STACK_INVALID",
    "ERR_MOVE_SHOOT",
    "ERR_ATTACK_IMPOSSIBLE",
]

PyAction = ctypes.c_int16
PyState = ctypes.c_float * STATE_SIZE
PyErrnames = ctypes.c_wchar * ERRSIZE
PyErrflags = ctypes.c_uint16 * ERRSIZE


class PyRawResult(ctypes.Structure):
    _fields_ = [
        ("state", PyState),
        ("errmask", ctypes.c_ushort),
        ("dmg_dealt", ctypes.c_int),
        ("dmg_received", ctypes.c_int),
        ("units_lost", ctypes.c_int),
        ("units_killed", ctypes.c_int),
        ("value_lost", ctypes.c_int),
        ("value_killed", ctypes.c_int),
        ("is_battle_over", ctypes.c_bool),
        ("is_victorious", ctypes.c_bool),
    ]


# Same as PyResult, but converts state to a numpy array
class PyResult():
    def __init__(self, cres):
        self.state = np.ctypeslib.as_array(cres.state)
        self.errmask = cres.errmask
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

        self.v_statesize = multiprocessing.Value(ctypes.c_int)
        self.v_nactions = multiprocessing.Value(ctypes.c_int)
        self.v_errflags = multiprocessing.Value(PyErrflags)
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

        return PyResult(self.v_result_act), list(self.v_errflags)

    def shutdown(self):
        with self.shutdown_lock:
            if self.shutdown_complete:
                return

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

        # attempt to prevent log duplication from ray PB2 training
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()

        self.cond = None

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

    # XXX: DISABLED as it does not work OK:
    #      At some point simply ANYONE can acquire the semaphore at ANY TIME
    #      Initially that's not the case, but after some ray tune runs it gets
    #      screwed and semaphore.acquire() becomes a no-op...
    #      Instead, with vcmi commit c2daffa43 I just removed writing to files
    #
    # def _try_start(self, retries, retry_interval):
    #     semaphore = posix_ipc.Semaphore(
    #         "vcmi-gym",
    #         flags=posix_ipc.O_CREAT,
    #         mode=0o600,
    #         initial_value=1
    #     )

    #     try:
    #         for attempt in range(retries):
    #             try:
    #                 semaphore.acquire(timeout=0)
    #                 self.logger.warn("SEMAPHORE ACQUIRED")
    #                 self.proc.start()
    #                 self.cond.wait()
    #                 return True
    #             except posix_ipc.BusyError:
    #                 time.sleep(retry_interval)
    #     finally:
    #         semaphore.release()
    #         semaphore.close()
    #         self.logger.warn("SEMAPHORE RELEASED")
    #
    #     return False

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
        self.logger.debug("VCMI args: %s" % str(args))
        # XXX: self.__connector is ONLY available in the sub-process!
        self.__connector = connector.Connector(*args)

        self.v_statesize.value = connector.get_state_size()
        self.v_nactions.value = connector.get_n_actions()
        errmapping = connector.get_error_mapping()

        # Sync sanity-check
        assert self.v_statesize.value == STATE_SIZE
        assert self.v_nactions.value == N_ACTIONS
        assert len(errmapping) == ERRSIZE

        errnames = []

        for (i, (errflag, (errname, _))) in enumerate(errmapping.items()):
            errnames.append(errname)
            self.v_errflags[i] = ctypes.c_uint16(errflag)
            i += 1

        # Sync sanity-check
        assert errnames == ERRNAMES

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
        self.v_result_act.errmask = ctypes.c_ushort(result.get_errmask())
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
