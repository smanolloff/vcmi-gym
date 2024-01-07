import gymnasium as gym
import threading
import enum
import collections

from .util import log

DEBUG = True
MAXLEN = 100


def withcond(func):
    def wrapper(*args, **kwargs):
        with args[0].cond:
            return func(*args, **kwargs)

    return wrapper


def tracelog(func):
    if not DEBUG:
        return func

    def wrapper(*args, **kwargs):
        this = args[0]
        this._debug("Start: %s (args=%s, kwargs=%s)" % (func.__name__, args[1:], log.trunc(repr(kwargs), MAXLEN)))
        result = func(*args, **kwargs)
        this._debug("End: %s (return %s)" % (func.__name__, log.trunc(repr(result), MAXLEN)))
        return result

    return wrapper


class State(enum.Enum):
    RESET = enum.auto()
    REG_A = enum.auto()
    REG_B = enum.auto()
    OBS_A = enum.auto()
    OBS_B = enum.auto()
    DEREG = enum.auto()
    BATTLE_END = enum.auto()


class Side(enum.Enum):
    # Must correspond to VCMI's MMAI::Export::Side enum
    ATTACKER = 0
    DEFENDER = 1


# XXX: can't be a nametuple (immutable)
class Clients():
    def __init__(self):
        self.A = None
        self.B = None

    def __repr__(self):
        return "Clients(A=%s, B=%s)" % (self.A, self.B)


Client = collections.namedtuple("Client", ["side"])
StepResult = collections.namedtuple("StepResult", ["obs", "rew", "term", "trunc", "info"])
ResetResult = collections.namedtuple("ResetResult", ["obs", "info"])


class DualEnvController():
    def __init__(self, env):
        self.env = env
        self.clients = Clients()
        self.state = State.RESET
        self.cond = threading.Condition()
        self.logger = log.get_logger("Controller", "DEBUG")

    #
    # XXX:
    # After a regular battle end, the terminal result is received twice.
    # (once for RED and once for BLUE)
    # In case of BATTLE_END, self.result holds the first of those results,
    # and the second one is queued to be returned immediately, on any action.
    # Since AAI expects the action to be RESET, we do a pre-reset in order to
    # obtain the second non-terminal result.
    # The "regular" reset afterwards (in flow_reg_a) will work as expected.
    #
    # NOTE: There are 2 calls to env.reset(), but only 1 restart occurs!
    # (because the defending does nothing with the reset action)
    #
    @withcond
    @tracelog
    def reset(self, *args, **kwargs):
        if self.state == State.BATTLE_END:
            self.env.reset(*args, **kwargs)
            self._transition(State.RESET)

        match self.state:
            case State.RESET:
                self._flow_reg_a(args, kwargs)
                return self.clients.A.side, tuple(self.result)
            case State.REG_A:
                self._flow_reg_b()
                # the result here may be a StepResult => extract only obs&info
                return self.clients.B.side, (self.result.obs, self.result.info)
            case State.OBS_A | State.OBS_B:
                # Can't reset mid-battle --  what are we to return to the
                # other client? (which called .step(), expecting a StepResult)
                # We must somehow return a terminal StepResult, but we only
                # have a ResetResult when resetting.
                # => just re-return the last result
                self.logger.warn("Ignoring mid-battle reset!")
                return Side(self.result.info["side"]), (self.result.obs, self.result.info)
            case _:
                raise Exception("Cannot reset while in state: %s" % self.state)

    @withcond
    @tracelog
    def step(self, side, *args, **kwargs):
        assert self.result.info["side"] == side.value, "expected last res side %s, got: %s" % (self.result.info["side"], side.value)  # noqa: E501

        # XXX: "B" can still be None here (waiting to complete registration)

        match side:
            case self.clients.A.side:
                name, other_name = "A", "B"
                state, other_state = State.OBS_A, State.OBS_B
            case self.clients.B.side:
                name, other_name = "B", "A"
                state, other_state = State.OBS_B, State.OBS_A
            case _:
                raise Exception("Unexpected side: %s" % side)

        assert getattr(self.clients, name).side == side, "expected side %s, got: %s" % (getattr(self.clients, name).side, side)  # noqa: E501
        self._assert_state(state)
        self.result = StepResult(*self.env.step(*args, **kwargs))

        if self.result.term or self.result.trunc:
            self._flow_dereg(name, other_name, state, other_state)
        elif self.result.info["side"] != side.value:
            self._flow_other(name, other_name, state, other_state)
        else:
            # nothing to do - flow = same
            pass

        return tuple(self.result)

    #
    # private
    #

    def _debug(self, msg):
        self.logger.debug("[%s] %s" % (getattr(self.state, "name", ""), msg))

    def _transition(self, newstate):
        self._debug("%s => %s" % (self.state, newstate))
        self.state = newstate

    def _assert_state(self, state):
        assert self.state == state, "%s != %s" % (self.state, state)

    def _assert_any_state(self, states):
        assert any(self.state == state for state in states), "expected one of: %s, got: %s" % (states, self.state)

    def _assert_client(self, name, client):
        assert getattr(self.clients, name) == client, "client %s != %s: %s" % (name, client, self.clients)

    def _flow_reg_a(self, env_args, env_kwargs):
        self._assert_client("A", None)
        self._transition(State.REG_A)

        self.cond.wait()  # wait-1

        self._assert_state(State.REG_B)
        self.result = ResetResult(*self.env.reset(*env_args, **env_kwargs))
        self.clients.A = Client(side=Side(self.result.info["side"]))
        self._transition(State.OBS_A)

    def _flow_reg_b(self):
        self._assert_client("B", None)
        self._transition(State.REG_B)

        self.cond.notify()  # for wait-1 or wait-3
        self.cond.wait()  # wait-2

        # Technically, the state here could also be DEREG
        # (if the battle ended without "B" ever receiving a move)
        # This, however, is unsupported (no "terminated" in reset's result)
        # => assert OBS_B
        self._assert_state(State.OBS_B)
        self.clients.B = Client(side=Side(self.result.info["side"]))
        assert self.clients.B != self.clients.A, "same clients: %s" % self.clients

    def _flow_other(self, name, other_name, state, other_state):
        self._transition(other_state)
        self.cond.notify()  # for wait-2 or wait-4
        self.cond.wait()  # wait-4

        if self.state == State.DEREG:
            self._assert_client(other_name, None)
            setattr(self.clients, name, None)
            self._transition(State.BATTLE_END)
            self.cond.notify()  # for wait-5
        else:
            self._assert_state(state)

    def _flow_dereg(self, name, _other_name, _state, _other_state):
        self._transition(State.DEREG)
        setattr(self.clients, name, None)

        self.cond.notify()  # for wait-2 or wait-4
        self.cond.wait()  # wait-5
        # XXX: no assert here - race cond (other thread may have called reset)


class DualEnvClient(gym.Env):
    def __init__(self, controller, name):
        self.controller = controller
        self.side = None
        self.logger = log.get_logger(name, "DEBUG")

    @tracelog
    def reset(self, *args, **kwargs):
        self.side, result = self.controller.reset(*args, **kwargs)
        return result

    @tracelog
    def step(self, *args, **kwargs):
        assert self.side is not None
        result = self.controller.step(self.side, *args, **kwargs)
        return result

    @tracelog
    def render(self, *args, **kwargs):
        # call env directly
        return self.controller.env.render(*args, **kwargs)

    @tracelog
    def close(self, *args, **kwargs):
        # call env directly
        return self.controller.env.close(*args, **kwargs)

    @tracelog
    def action_masks(self):
        # call env directly
        return self.controller.env.action_masks()

    #
    # private
    #

    def _debug(self, msg):
        self.logger.debug("[%s] %s" % (self.side, msg))
