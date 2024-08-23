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

import gymnasium as gym
import threading
import enum
import numpy as np
import sys
from typing import NamedTuple
import warnings

from . import log


DEBUG = False
MAXLEN = 100


def withcond(func):
    def wrapper(self, *args, **kwargs):
        self._debug("obtaining lock...")
        with self.cond:
            self._debug("lock obtained")
            try:
                retval = func(self, *args, **kwargs)
            except Exception:
                print("DUMP: state=%s, clients=%s, result=%s, stored_result=%s" % (self.state, self.clients, self.result, self.stored_result))
                raise
        self._debug("lock released")
        return retval
    return wrapper


def tracelog(func):
    if not DEBUG:
        return func

    def wrapper(*args, **kwargs):
        this = args[0]
        this._debug("Begin: %s (args=%s, kwargs=%s)" % (func.__name__, args[1:], log.trunc(repr(kwargs), MAXLEN)))
        result = func(*args, **kwargs)
        this._debug("End: %s (return %s)" % (func.__name__, log.trunc(repr(result), MAXLEN)))
        return result

    return wrapper


class State(enum.Enum):
    RESET = enum.auto()
    REG = enum.auto()
    OBS_A = enum.auto()
    OBS_B = enum.auto()
    BATTLE_END = enum.auto()
    MIDBATTLE_RESET = enum.auto()


class Side(enum.Enum):
    # Must correspond to VCMI's MMAI::Schema::V1::Side enum
    ATTACKER = 0
    DEFENDER = 1


# XXX: can't be a nametuple (immutable)
class Clients:
    def __init__(self):
        self.A = None
        self.B = None

    def __repr__(self):
        return "Clients(A=%s, B=%s)" % (self.A, self.B)


class Client(NamedTuple):
    side: Side


class ResetResult(NamedTuple):
    obs: np.ndarray
    info: bool


class StepResult(NamedTuple):
    obs: np.ndarray
    rew: float
    term: bool
    trunc: bool
    info: dict


class DualEnvController:
    def __init__(self, env, timeout=5, loglevel="DEBUG"):
        # See note in flow_battle_end
        assert env.allow_retreat, "Retreats must be allowed"
        self.env = env
        self.timeout = timeout
        self._timeout = timeout  # modified at quit
        self.clients = Clients()
        self.state = State.RESET
        self.cond = threading.Condition()
        self.logger = log.get_logger("Controller", loglevel)
        self.termresults = {Side.ATTACKER: None, Side.DEFENDER: None}
        self.stored_result = None
        self.aside = None  # side to use for client "A"
        self.quit_requested = False

    @withcond
    @tracelog
    def reset(self, side, desired_side):
        assert desired_side in [None, Side.ATTACKER, Side.DEFENDER]

        match self.state:
            case State.RESET:
                self._flow_reg_a(desired_side)
                # the result here may be a StepResult => extract only obs&info
                return self.clients.A.side, (self.result.obs, self.result.info)
            case State.REG:
                self._flow_reg_b(desired_side)
                # the result here may be a StepResult => extract only obs&info
                return self.clients.B.side, (self.result.obs, self.result.info)
            case State.BATTLE_END:
                self._info("flow STORED")
                assert self.result.info["side"] == side.value  # side should not change
                name, _ = self._infer_client_names(side)
                self._transition(State.OBS_A if name == "A" else State.OBS_B)
                return side, tuple(self.result)
            case State.OBS_A | State.OBS_B:
                self._flow_midbattle_reset(side)
                assert self.result.info["side"] == side.value  # side should not change
                return side, (self.result.obs, self.result.info)
            case _:
                raise Exception("Cannot reset while in state: %s" % self.state)

    @withcond
    @tracelog
    def step(self, side, action):
        assert self.result.info["side"] == side.value, "expected last res side %s, got: %s" % (self.result.info["side"], side.value)  # noqa: E501

        name, other_name = self._infer_client_names(side)

        if name == "A":
            state, other_state = State.OBS_A, State.OBS_B
        else:
            state, other_state = State.OBS_B, State.OBS_A

        assert self._get_client(name).side == side, "expected side %s, got: %s" % (self._get_client(name).side, side)  # noqa: E501
        self._assert_state(state)
        self.result = StepResult(*self._env_step(action))

        if self.result.term:
            self._flow_battle_end(name, other_name, state, other_state)
        elif self.result.info["side"] != side.value:
            self._flow_other(name, other_name, state, other_state)
        else:
            # nothing to do - flow = same
            self._info("flow SAME")
            pass

        return tuple(self.result)

    @tracelog
    def close(self, *args, **kwargs):
        self._timeout = 0
        self.quit_requested = True

        if self.cond.acquire(timeout=self.timeout):
            self.cond.notify_all()
            self.cond.release()

        return self.env.close(*args, **kwargs)
    #
    # private
    #

    def _log(self, level, msg):
        getattr(self.logger, level)("[%s] %s" % (getattr(self.state, "name", ""), msg))

    def _debug(self, msg):
        self._log("debug", msg)

    def _info(self, msg):
        self._log("info", msg)

    def _warn(self, msg):
        self._log("warn", msg)

    def _error(self, msg):
        self._log("error", msg)

    def _infer_client_names(self, side):
        match side:
            case self.clients.A.side:
                return "A", "B"
            case self.clients.B.side:
                return "B", "A"
            case _:
                raise Exception("Unexpected side: %s" % side)

    def _cond_notify(self):
        self._info("cond.notify()...")
        self.cond.notify()

    @tracelog
    def _env_step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._info(f"Step ({action}): rew={rew}, term={term}, trunc={trunc}, side={info['side']}")
        return obs, rew, term, trunc, info

    @tracelog
    def _cond_wait(self, wait_name):
        self._info("waiting %s ..." % wait_name)
        if self.cond.wait(timeout=self._timeout):
            self._info("waited %s" % wait_name)
            # notification received on time, all is good
            if not self.quit_requested:
                return

            sys.exit(0)

        if self.quit_requested:
            self._info("Cond wait timeout (quit_requested=%s)" % self.quit_requested)
            sys.exit(0)
        else:
            self._error("Cond wait timeout (quit_requested=%s)" % self.quit_requested)
            sys.exit(1)

    def _transition(self, newstate):
        self._info("%s => %s" % (self.state, newstate))
        self.state = newstate

    def _assert_state(self, state):
        assert self.state == state, "%s != %s" % (self.state, state)

    def _assert_any_state(self, states):
        assert any(self.state == state for state in states), "expected one of: %s, got: %s" % (states, self.state)

    @tracelog
    def _set_termresult(self, res):
        self.termresults[Side(res.info["side"])] = res

    def _assert_client(self, name, client):
        assert self._get_client(name) == client, "client %s != %s: %s" % (name, client, self.clients)

    def _get_client(self, name):
        return getattr(self.clients, name)

    def _flow_reg_a(self, desired_side):
        self._info("flow REG_A")
        self._assert_client("A", None)
        self._transition(State.REG)
        self.aside = desired_side
        self._cond_notify()  # for wait-4
        self._cond_wait("wait-1 (flow REG_A)")  # wait-1
        self._assert_state(State.OBS_A)

    def _flow_reg_b(self, desired_side):
        self._info("flow REG_B")
        self._assert_client("B", None)

        if desired_side == Side.ATTACKER:
            assert self.aside in [None, Side.DEFENDER], "desired_side conflict: %s" % self.aside
            self.clients.A = Client(side=Side.DEFENDER)
            self.clients.B = Client(side=Side.ATTACKER)
        elif desired_side == Side.DEFENDER:
            assert self.aside in [None, Side.ATTACKER], "desired_side conflict: %s" % self.aside
            self.clients.A = Client(side=Side.ATTACKER)
            self.clients.B = Client(side=Side.DEFENDER)
        elif self.aside is Side.DEFENDER:
            self.clients.A = Client(side=Side.DEFENDER)
            self.clients.B = Client(side=Side.ATTACKER)
        else:  # self.aside in [None, Side.ATTACKER]
            self.clients.A = Client(side=Side.ATTACKER)
            self.clients.B = Client(side=Side.DEFENDER)

        self.result = ResetResult(*self.env.reset())

        if self.result.info["side"] == self.clients.B.side.value:
            self._transition(State.OBS_B)
        else:
            self._transition(State.OBS_A)
            self._cond_notify()  # for wait-1
            self._cond_wait("wait-2 (flow REG_B)")

    def _flow_other(self, name, other_name, state, other_state):
        self._info("flow OTHER")
        self._transition(other_state)
        self._cond_notify()  # for wait-1, wait-2 or wait-3
        self._cond_wait("wait-3 (flow OTHER)")

        if self.state == State.BATTLE_END:
            self._assert_client(other_name, None)
            client_side = self._get_client(name).side
            assert self.termresults[client_side], "missing term result for %s: %s" % (client_side, self.termresults)
            self.result = self.termresults[client_side]
            self.termresults[client_side] = None
            setattr(self.clients, name, None)
            self._transition(State.RESET)
        else:
            self._assert_state(state)

    #
    # XXX:
    # After a regular battle end, the terminal result is received twice:
    # once for ATTACKER and once for DEFENDER. Both BAI send it, but
    # in UNDEFINED order.
    #
    # Currently, self.result holds the first of those results and its BAI
    # expects ACTION_RESET. The result itself may be for the other side.
    # => we will call step(ACTION_RESET). The first BAI is then destroyed.
    #    NOTE: we don't call reset() here, as we need a StepResult.
    # As soon as the BAI is destroyed, VCMI client calls battleEnd() on the
    # other BAI, which calls getAction() with its own terminal result.
    # => our reset() will receive the 2nd terminal result.
    # Now we have both term results, and we must return the one matching our
    # client's side.
    #
    # The RL algo/models of both clients will then call reset() but
    # the controller will call env.reset() only once (the regular REG flow).
    #
    # This means there will be 2 ACTION_RESETs sent to VCMI after battle end,
    # but only 1 restart will occur!
    # (because the defending BAI does nothing with the reset action)
    #
    def _flow_battle_end(self, name, other_name, state, other_state):
        self._info("flow BATTLE_END")

        client_side = self._get_client(name).side

        setattr(self.clients, name, None)

        assert not self.termresults[Side.ATTACKER], "expected no term results, have: %s" % self.termresults
        assert not self.termresults[Side.DEFENDER], "expected no term results, have: %s" % self.termresults
        self._set_termresult(self.result)
        self.result = StepResult(*self._env_step(-1))  # -1 is ACTION_RESET

        if self.result.term:
            # BATTLE_END: case 1 of 3
            self._transition(State.BATTLE_END)
            self._set_termresult(self.result)
            assert self.termresults[Side.ATTACKER], "missing term result: %s" % self.termresults
            assert self.termresults[Side.DEFENDER], "missing term result: %s" % self.termresults

            self.result = self.termresults[client_side]
            self._cond_notify()  # for wait-2 or wait-4
            self._cond_wait("wait-4 (flow BATTLE_END)")

            self.termresults[client_side] = None
        elif self.result.info["side"] == client_side.value:
            # BATTLE_END: case 2 of 3
            self._transition(State.BATTLE_END)
            self.stored_result = self.result
            self.result = self.termresults[client_side]
        else:
            # BATTLE_END: case 3 of 3
            self._flow_other(name, other_name, state, other_state)

    def _flow_midbattle_reset(self, side):
        self._info("flow MIDBATTLE_RESET")
        self.result = StepResult(*self._env_step(-1))  # -1 is ACTION_RESET

        if self.result.info["side"] == side.value:
            # MIDBATTLE_RESET: case 1
            self._info("flow SAME")
        else:
            # MIDBATTLE_RESET: case 2
            name, other_name = self._infer_client_names(side)

            if name == "A":
                state, other_state = State.OBS_A, State.OBS_B
            else:
                state, other_state = State.OBS_B, State.OBS_A

            self._flow_other(name, other_name, state, other_state)


class DualEnvClient(gym.Env):
    def __init__(self, controller, name, desired_side=None, loglevel="DEBUG"):
        assert desired_side in [None, Side.ATTACKER, Side.DEFENDER], "invalid side: %s" % desired_side
        self.controller = controller
        self.side = None
        self.desired_side = desired_side
        self.logger = log.get_logger(name, loglevel)

        # VcmiEnv public attributes
        self.action_space = controller.env.action_space
        self.observation_space = controller.env.observation_space
        self.render_mode = controller.env.render_mode

    # XXX: args/kwargs not supported
    #      (two envs will call reset with potentially different args)
    @tracelog
    def reset(self, *args, **kwargs):
        if args:
            warnings.warn("arguments for reset will be ignored: %s" % args)
        if kwargs:
            warnings.warn("keyword arguments for reset will be ignored: %s" % kwargs)

        new_side, result = self.controller.reset(self.side, self.desired_side)

        assert new_side == self.desired_side, "controller gave wrong side: want: %s, have: %s" % (self.desired_side, new_side)
        self.side = new_side

        return result

    @tracelog
    def step(self, action):
        assert self.side is not None
        result = self.controller.step(self.side, action)
        return result

    @tracelog
    def render(self, *args, **kwargs):
        # call env directly
        return self.controller.env.render(*args, **kwargs)

    @tracelog
    def close(self, *args, **kwargs):
        return self.controller.close(*args, **kwargs)

    @tracelog
    def action_mask(self):
        return self.controller.env.action_mask()

    @tracelog
    def attn_mask(self):
        return self.controller.env.attn_mask()

    def decode(self):
        return self.controller.env.decode()

    #
    # private
    #

    def _log(self, level, msg):
        getattr(self.logger, level)("[%s] %s" % (self.side, msg))

    def _debug(self, msg):
        self._log("debug", msg)


if __name__ == "__main__":
    import torch
    import numpy as np
    import time
    import logging
    from ..v3.vcmi_env import VcmiEnv

    def npstr(self):
        return f"ndarray{self.shape}"

    np.set_string_function(npstr)

    def predictor(model):
        def pred(obs, mask):
            return model.predict(torch.as_tensor(obs), torch.as_tensor(mask))
        return pred

    def legacy_predictor(model):
        def pred(obs, mask):
            return model.predict(torch.as_tensor(obs[:, :, 1:]), torch.as_tensor(mask))
        return pred

    attacker = predictor(torch.jit.load("rl/models/Attacker model:v9/jit-agent.pt"))
    defender = predictor(torch.jit.load("rl/models/Defender Model:v5/jit-agent.pt"))

    steps = 0

    def play(controller, side, predictors, games, name):
        global steps
        client = DualEnvClient(controller, name, side, logging.getLevelName(controller.logger.level))
        pred_att, pred_def = predictors
        logger = log.get_logger(f"player-{name}", logging.getLevelName(controller.logger.level))
        counter = 0

        while counter < games:
            # time.sleep(np.random.rand())
            logger.info("Will reset (%d)" % counter)
            counter += 1
            obs, _ = client.reset()
            if client.side == Side.ATTACKER:
                logger.info("Will play as ATTACKER")
                pred = pred_att
            else:
                logger.info("Will play as DEFENDER")
                pred = pred_def

            action = pred(obs, client.action_mask())

            while True:
                # time.sleep(np.random.rand())
                steps += 1

                if steps % 5 < 3:
                    logger.info("Steps: %d, will restart" % steps)
                    obs, info = client.reset()
                    term = trunc = False
                else:
                    logger.info("Steps: %d, will step" % steps)
                    obs, _, term, trunc, info = client.step(action)

                if term or trunc:
                    logger.info("Got terminal state. Side: %s" % info["side"])
                    break
                action = pred(obs, client.action_mask())

        logger.info("Done.")

    env = VcmiEnv(
        mapname="gym/A1.vmap",
        attacker="MMAI_USER",
        defender="MMAI_USER",
        allow_retreat=True,
        max_steps=50,
        vcmi_loglevel_ai="debug"
    )
    controller = DualEnvController(env, timeout=1, loglevel="DEBUG")

    kwargs = dict(controller=controller, predictors=(attacker, defender), games=2)
    baba = threading.Thread(target=play, kwargs=dict(kwargs, name="att", side=Side.ATTACKER))
    pena = threading.Thread(target=play, kwargs=dict(kwargs, name="def", side=Side.DEFENDER))

    ts = time.time()
    baba.start()
    pena.start()

    baba.join()
    pena.join()
    elapsed = time.time() - ts
    print("Elapsed: %.3fs, steps: %d, steps/s: %.2f" % (elapsed, steps, steps/elapsed))
