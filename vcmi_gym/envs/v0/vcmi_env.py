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

import numpy as np
import gymnasium as gym
import os

from .util import log
from .util.analyzer import Analyzer, ActionType
from .util.pyconnector import (
    PyConnector,
    STATE_SIZE_X,
    STATE_SIZE_Y,
    STATE_SIZE_Z,
    N_HEX_ATTRS,
    N_ACTIONS,
    NV_MAX,
    NV_MIN
)


# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ZERO = DTYPE(0)
ONE = DTYPE(1)

TRACE = True
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


class InfoDict(dict):
    SCALAR_VALUES = [
        "net_value",
        "is_success",
        # "reward_clip_abs_total",
        # "reward_clip_abs_max",
    ]

    D1_ARRAY_VALUES = {
        "action_type_counters": [at.name for at in ActionType],
    }

    ALL_KEYS = ["side"] + SCALAR_VALUES + list(D1_ARRAY_VALUES.keys())

    def __setitem__(self, k, v):
        assert k in InfoDict.ALL_KEYS, f"Unknown info key: '{k}'"
        super().__setitem__(k, v)


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]
    ROLES = ["MMAI_USER", "MMAI_MODEL", "StupidAI", "BattleAI"]

    def __init__(
        self,
        mapname,
        seed=None,  # not used currently
        render_mode="ansi",
        max_steps=500,
        render_each_step=False,
        vcmi_loglevel_global="error",  # vcmi loglevel
        vcmi_loglevel_ai="error",  # vcmi loglevel
        vcmienv_loglevel="WARN",  # python loglevel
        consecutive_error_reward_factor=-1,  # unused
        attacker="MMAI_USER",  # MMAI_USER / MMAI_MODEL / StupidAI / BattleAI
        defender="StupidAI",  # MMAI_USER / MMAI_MODEL / StupidAI / BattleAI
        attacker_model=None,  # MPPO zip model (if attacker=MMAI_MODEL)
        defender_model=None,  # MPPO zip model (if defender=MMAI_MODEL)
        sparse_info=False,
        allow_invalid_actions=False,
        actions_log_file=None,  # DEBUG
        user_timeout=0,  # seconds
        vcmi_timeout=5,  # seconds
        boot_timeout=0,  # seconds
        hexattr_filter=None,
        reward_dmg_factor=5,
        reward_clip_tanh_army_frac=1,  # max action reward relative to starting army value
        reward_army_value_ref=0,  # scale rewards relative to starting army value (0=no scaling)
        step_reward_fixed=0,  # fixed reward
        step_reward_mult=1,
        term_reward_mult=1,  # at term step, reward = diff in total army values
    ):
        assert vcmi_loglevel_global in VcmiEnv.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in VcmiEnv.VCMI_LOGLEVELS
        assert attacker in VcmiEnv.ROLES
        assert defender in VcmiEnv.ROLES

        assert attacker == "MMAI_USER" or defender == "MMAI_USER", "an MMAI_USER role is required"

        self.logger = log.get_logger("VcmiEnv", vcmienv_loglevel)
        self.connector = PyConnector(vcmienv_loglevel, user_timeout, vcmi_timeout, boot_timeout)

        result = self.connector.start(
            os.getcwd(),
            mapname,
            vcmi_loglevel_global,
            vcmi_loglevel_ai,
            attacker,
            defender,
            attacker_model or "",
            defender_model or ""
        )

        # NOTE: removing action=0 (retreat) which is used for resetting...
        #       => start from 1 and reduce total actions by 1
        #          XXX: there seems to be a bug as start=1 causes this error:
        #          index 1322 is out of bounds for dimension 2 with size 1322
        #       => just start from 0, reduce max by 1, and manually add +1
        self.action_offset = 1
        self.action_space = gym.spaces.Discrete(N_ACTIONS - self.action_offset)

        if hexattr_filter:
            assert isinstance(hexattr_filter, list)
            assert len(hexattr_filter) == len(list(set(hexattr_filter)))
            assert len(hexattr_filter) <= STATE_SIZE_X
            assert all(isinstance(x, int) and x < N_HEX_ATTRS for x in hexattr_filter)
            self.observation_space = gym.spaces.Box(
                # id, state, stack qty, side, type
                shape=(STATE_SIZE_Z, STATE_SIZE_Y, 15 * len(hexattr_filter)),
                low=NV_MIN,
                high=NV_MAX,
                dtype=DTYPE
            )
        else:
            self.observation_space = gym.spaces.Box(
                shape=(STATE_SIZE_Z, STATE_SIZE_Y, STATE_SIZE_X),
                low=NV_MIN,
                high=NV_MAX,
                dtype=DTYPE
            )

        self.actfile = None

        # <params>
        self.render_mode = render_mode
        self.sparse_info = sparse_info
        self.allow_invalid_actions = allow_invalid_actions
        self.max_steps = max_steps
        self.render_each_step = render_each_step
        self.mapname = mapname
        self.attacker = attacker
        self.defender = defender
        self.attacker_model = attacker_model
        self.defender_model = defender_model
        self.actions_log_file = actions_log_file
        self.hexattr_filter = hexattr_filter
        self.reward_clip_tanh_army_frac = reward_clip_tanh_army_frac
        self.reward_army_value_ref = reward_army_value_ref
        self.reward_dmg_factor = reward_dmg_factor
        self.step_reward_fixed = step_reward_fixed
        self.step_reward_mult = step_reward_mult
        self.term_reward_mult = term_reward_mult
        # </params>

        # required to init vars
        self._reset_vars(result)

    @tracelog
    def step(self, action):
        action = self._transform_action(action)

        if self.terminated or self.truncated:
            raise Exception("Reset needed")

        if self.actfile:
            self.actfile.write(f"{action}\n")

        res = self.connector.act(action)
        # assert res.errmask == 0 or self.allow_invalid_actions, "Errmask != 0: %d" % res.errmask
        if res.errmask != 0 and not self.allow_invalid_actions:
            # MQRDQN sometimes (very rarely) hits an invalid action
            # Could be a corner case for just after learning starts
            self.logger.warn("errmask=%d" % res.errmask)

        analysis = self.analyzer.analyze(action, res)
        term = res.is_battle_over
        rew, rew_unclipped = self.calc_reward(analysis, res)

        obs = VcmiEnv.maybe_filter_hexattrs(res.state, self.hexattr_filter)
        trunc = self.analyzer.actions_count >= self.max_steps

        self._update_vars_after_step(res, rew, rew_unclipped, term, trunc, analysis)
        self._maybe_render(analysis)

        info = VcmiEnv.build_info(
            res,
            term,
            trunc,
            analysis,
            self.reward_clip_abs_total,
            self.reward_clip_abs_max,
            self.sparse_info
        )

        return obs, rew, term, trunc, info

    @tracelog
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        result = self.connector.reset()
        self._reset_vars(result)

        if self.render_each_step:
            print(self.render())

        if self.actions_log_file:
            if self.actfile:
                self.actfile.close()
            self.actfile = open(self.actions_log_file, "w")

        obs = VcmiEnv.maybe_filter_hexattrs(self.result.state, self.hexattr_filter)
        return obs, {"side": self.result.side}

    @tracelog
    def render(self):
        if self.render_mode == "ansi":
            return (
                "%s\n"
                "Step:      %-5s\n"
                "Reward:    %-5s (total: %s)\n"
                "Net value: %-5s (total: %s)"
            ) % (
                self.connector.render_ansi(),
                self.analyzer.actions_count,
                self.reward,
                self.reward_total,
                self.net_value_last,
                self.analyzer.net_value
            )

        elif self.render_mode == "rgb_array":
            gym.logger.warn("Rendering RGB arrays not yet implemented for VcmiEnv")
        elif self.render_mode is None:
            gym.logger.warn("Cannot render with no render mode set")

    @tracelog
    def close(self):
        self.logger.info("Closing env...")
        if self.actfile:
            self.actfile.close()

        self.connector.shutdown()
        self.logger.info("Env closed")
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()

    @tracelog
    def action_masks(self):
        return self.result.actmask[self.action_offset:]

    # Print all info about a hex
    # To test:
    # import vcmi_gym
    # env = vcmi_gym.VcmiEnv("ai/generated/A03.vmap")
    # obs = env.step(589)[0]
    # env.hexreport(obs, 14, 3)
    def hexreport(self, obs, x, y):
        iy = y-1
        ix0 = (x-1) * N_HEX_ATTRS
        ix1 = (x) * N_HEX_ATTRS

        print("Obs range: [%d][%d:%d]" % (iy, ix0, ix1))
        part = obs[0][iy][ix0:ix1]

        def round_and_strip(number, decimals=2):
            rounded_number = round(number, decimals)
            return str(rounded_number).rstrip('0').rstrip('.') if '.' in str(rounded_number) else str(rounded_number)

        def vreport(i, name, vmin, vmax, voffset=0):
            norm = part[i]
            unnorm = round_and_strip(norm * (vmax - vmin) + vmin, 3)
            print("[%d] %s: %s" % (i, name, unnorm))

        def vstackreport(i, name, vmin, vmax, voffset=0):
            vreport(i, f"{name} #1", 0, 1)
            vreport(i+1, f"{name} #2", 0, 1)
            vreport(i+2, f"{name} #3", 0, 1)
            vreport(i+3, f"{name} #4", 0, 1)
            vreport(i+4, f"{name} #5", 0, 1)
            vreport(i+5, f"{name} #6", 0, 1)
            vreport(i+6, f"{name} #7", 0, 1)

        vreport(0, "y (0-based)", 0, 10)
        vreport(1, "x (0-based)", 0, 14)
        vreport(2, "state", 1, 4)
        vreport(3, "StackAttr::Quantity", -1, 5000, 1)
        vreport(4, "StackAttr::Attack", -1, 100, 1)
        vreport(5, "StackAttr::Defense", -1, 100, 1)
        vreport(6, "StackAttr::Shots", -1, 32, 1)
        vreport(7, "StackAttr::DmgMin", -1, 100, 1)
        vreport(8, "StackAttr::DmgMax", -1, 100, 1)
        vreport(9, "StackAttr::HP", -1, 1500, 1)
        vreport(10, "StackAttr::HPLeft", -1, 1500, 1)
        vreport(11, "StackAttr::Speed", -1, 30, 1)
        vreport(12, "StackAttr::Waited", -1, 1, 1)
        vreport(13, "StackAttr::QueuePos", -1, 14, 1)
        vreport(14, "StackAttr::RetaliationsLeft", -1, 3, 1)
        vreport(15, "StackAttr::Side", -1, 1, 1)
        vreport(16, "StackAttr::Slot", -1, 6, 1)
        vreport(17, "StackAttr::CreatureType", -1, 150, 1)
        vreport(18, "StackAttr::AIValue", -1, 40000, 1)
        vreport(19, "StackAttr::IsActive", -1, 1, 1)
        vreport(20, "rangedDmgModifier", 0, 1)
        vstackreport(21, "Reachability by friendly stack", 0, 1)
        vstackreport(28, "Reachability by enemy stack", 0, 1)
        vstackreport(35, "Neighbouring friendly stack", 0, 1)
        vstackreport(42, "Neighbouring enemy stack", 0, 1)
        vstackreport(49, "Potential enemy attacking stack", 0, 1)
        assert N_HEX_ATTRS == 56

    # just play a pre-recorded actions from vcmi actions.txt
    # useful when "resuming" battle after reproducing & fixing a bug
    def replay_actions_txt(self, actions_txt="vcmi_gym/envs/v0/vcmi/actions.txt"):
        with open(actions_txt, "r") as f:
            actions_str = f.read()

        for a in actions_str.split():
            print("Replaying %s" % a)
            obs, rew, term, trunc, info = self.step(int(a) - 1)
            if rew < 0:
                print(self.render())
                raise Exception("error replaying: %s" % a)

        print(self.render())

    #
    # private
    #

    def _update_vars_after_step(self, res, rew, rew_unclipped, term, trunc, analysis):
        reward_clip_abs = abs(rew - rew_unclipped)
        self.result = res
        self.reward = rew
        self.reward_total += rew
        self.reward_clip_abs_total += reward_clip_abs
        self.reward_clip_abs_max = max(reward_clip_abs, self.reward_clip_abs_max)
        self.net_value_last = analysis.net_value
        self.terminated = term
        self.truncated = trunc

    def _reset_vars(self, res):
        self.result = res
        self.reward = 0
        self.reward_total = 0
        self.reward_clip_abs_total = 0
        self.reward_clip_abs_max = 0
        self.net_value_last = 0
        self.terminated = False
        self.truncated = False

        # TODO: handle unequal starting army values
        #       e.g. smaller starting army = larger rew for that side?

        # Tanh clipping:
        # Used to clip max reward for a single action
        # The clip value is relative to the avg starting army value
        avg_army_value = (res.side0_army_value + res.side1_army_value) / 2
        self.reward_clip_tanh_value = avg_army_value * self.reward_clip_tanh_army_frac

        # Reward scaling factor:
        # Used to equalize avg rewards for different starting army values
        # E.g. 1K starting armies will have similar avg rewards as 100K armies
        # This scaling is applied last, after all other reward modifiers
        if self.reward_army_value_ref:
            self.reward_scaling_factor = self.reward_army_value_ref / avg_army_value
        else:
            self.reward_scaling_factor = 1

        # Vars updated after other events
        self.analyzer = Analyzer(N_ACTIONS - self.action_offset)

    def _maybe_render(self, analysis):
        if self.render_each_step:
            print(self.render())

    def _transform_action(self, action):
        # see note for action_space
        max_action = self.action_space.n - self.action_offset
        assert action >= 0 and action <= max_action
        return action + self.action_offset

    #
    # A note on the static methods
    # Initially simply instance methods, I wanted to rip off the
    # reference to "self" as it introduced bugs related to the env state
    # (eg. some instance vars being used in calculations were not yet updated)
    # This approach is justified for training-critical methods only
    # (ie. no need to abstract out `render`, for example)

    @staticmethod
    def maybe_filter_hexattrs(state, hexattr_filter):
        if not hexattr_filter:
            return state

        # original obs is (1, 11, 15*N_HEX_ATTRS)
        return state.reshape(11, 15, N_HEX_ATTRS)[..., hexattr_filter].reshape(1, 11, 15 * len(hexattr_filter))

    #
    # NOTE:
    # info is read only after env termination
    # One-time values will be lost, put only only cumulatives/totals/etc.
    #
    @staticmethod
    def build_info(res, term, trunc, analysis, rewclip_total, rewclip_max, sparse_info):
        # Performance optimization
        if not (term or trunc) and sparse_info:
            return {"side": res.side}

        # XXX: do not use constructor args (bypasses validations)
        info = InfoDict()
        info["side"] = res.side
        info["net_value"] = analysis.net_value_ep
        info["is_success"] = res.is_victorious
        info["action_type_counters"] = analysis.action_type_counters_ep
        # info["reward_clip_abs_total"] = rewclip_total
        # info["reward_clip_abs_max"] = rewclip_max

        # Return regular dict (wrappers insert arbitary keys)
        return dict(info)

    def calc_reward(self, analysis, res):
        if res.errmask > 0:
            return -100, -100

        rew = analysis.net_value + self.reward_dmg_factor * analysis.net_dmg
        rew *= self.step_reward_mult

        if res.is_battle_over:
            vdiff = res.side0_army_value - res.side1_army_value
            vdiff = -vdiff if res.side == 1 else vdiff
            rew += (self.term_reward_mult * vdiff)

        rew += self.step_reward_fixed

        # Visualize on https://www.desmos.com/calculator
        # In expression 1 type: s = 5000
        # In expression 2 type: s * tanh(x/s)
        # NOTE: this seems useless
        if self.reward_clip_tanh_value:
            clipped = self.reward_clip_tanh_value * np.tanh(rew / self.reward_clip_tanh_value)
        else:
            clipped = rew

        rew *= self.reward_scaling_factor
        clipped *= self.reward_scaling_factor

        return clipped, rew
