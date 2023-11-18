import multiprocessing
import ctypes
import numpy as np

from . import log

# NOTE: those are hard-coded here as there is no way
#       to obtain them without importing connector
PyAction = ctypes.c_int16
PyState = ctypes.c_float * 334      # !!! SYNC with aitypes.h !!!
PyErrnames = ctypes.c_wchar * 9     # !!! SYNC with aitypes.h !!!
PyErrflags = ctypes.c_uint16 * 9    # !!! SYNC with aitypes.h !!!

ERRNAMES = [
    # !!! SYNC with aitypes.h !!!
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
    def __init__(self, *args):
        self.started = False
        self.connargs = args

    def start(self):
        assert not self.started, "Already started"
        self.started = True
        self.logger = log.get_logger("PyConnector", "DEBUG")

        self.v_statesize = multiprocessing.Value(ctypes.c_int)
        self.v_nactions = multiprocessing.Value(ctypes.c_int)
        self.v_errflags = multiprocessing.Value(PyErrflags)
        self.v_action = multiprocessing.Value(PyAction)
        self.v_result_act = multiprocessing.Value(PyRawResult)
        self.v_result_render_ansi = PyRenderAnsiResult = multiprocessing.Array(ctypes.c_char, 5000)
        self.v_command_type = multiprocessing.Value(ctypes.c_int)

        # Process synchronization:
        # cond.notify() will wake up the other proc (which immediately tries to acquire the lock)
        # cond.wait() will release the lock and wait (other proc now successfully acquires the lock)
        self.cond = multiprocessing.Condition()
        self.cond.acquire()

        self.proc = multiprocessing.Process(target=self.start_connector, args=self.connargs)
        self.proc.start()
        self.cond.wait()

        return (
            PyResult(self.v_result_act),
            self.v_statesize.value,
            self.v_nactions.value,
            list(self.v_errflags)
        )

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

    #
    # This method is the SUB-PROCESS
    # It enters an infinite loop, waiting for commands
    #
    def start_connector(self, *args):
        # NOTE: import is done here to ensure VCMI inits
        #       in the newly created process
        from ..connector.build import connector

        self.logger = log.get_logger("PyConnector-sub", "DEBUG")

        self.logger.debug("Starting with args: %s" % str(args))
        # XXX: self.__connector is ONLY available in the sub-process!
        self.__connector = connector.Connector(*args)

        self.v_statesize.value = connector.get_state_size()
        self.v_nactions.value = connector.get_n_actions()

        i = 0
        errnames = []

        for (errflag, (errname, _)) in connector.get_error_mapping().items():
            errnames.append(errname)
            self.v_errflags[i] = ctypes.c_uint16(errflag)
            i += 1

        # Sync sanity-check
        assert i == len(connector.get_error_mapping())
        assert errnames == ERRNAMES
        assert self.v_result_act.state._length_ == self.v_statesize.value

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
        x = result.get_state()
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
