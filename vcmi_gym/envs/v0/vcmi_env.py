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

from .util import log
from .util.analyzer import Analyzer, ActionType
from .util.pyconnector import (
    PyConnector,
    STATE_SIZE_X,
    STATE_SIZE_Y,
    STATE_SIZE_Z,
    N_ACTIONS,
    NV_MAX,
    NV_MIN,
)


# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ZERO = DTYPE(0)
ONE = DTYPE(1)

TRACE = True
MAXLEN = 80

ARMY_VALUE_REF = 600_000


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
        actions_log_file=None,  # DEBUG
        user_timeout=0,  # seconds
        vcmi_timeout=5,  # seconds
        boot_timeout=0,  # seconds
        reward_clip_mod=None,  # clip at +/- this value
    ):
        assert vcmi_loglevel_global in VcmiEnv.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in VcmiEnv.VCMI_LOGLEVELS
        assert attacker in VcmiEnv.ROLES
        assert defender in VcmiEnv.ROLES

        assert attacker == "MMAI_USER" or defender == "MMAI_USER", "an MMAI_USER role is required"

        self.logger = log.get_logger("VcmiEnv", vcmienv_loglevel)
        self.connector = PyConnector(vcmienv_loglevel, user_timeout, vcmi_timeout, boot_timeout)

        result = self.connector.start(
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
        self.max_steps = max_steps
        self.render_each_step = render_each_step
        self.mapname = mapname
        self.attacker = attacker
        self.defender = defender
        self.attacker_model = attacker_model
        self.defender_model = defender_model
        self.actions_log_file = actions_log_file
        self.reward_clip_mod = reward_clip_mod
        # </params>

        # required to init vars
        self._reset_vars(result)

    @tracelog
    def step(self, action):
        action += self.action_offset  # see note for action_space

        if self.terminated or self.truncated:
            raise Exception("Reset needed")

        if self.actfile:
            self.actfile.write(f"{action}\n")

        res = self.connector.act(action)
        assert res.errmask == 0

        analysis = self.analyzer.analyze(action, res)
        rew, rew_unclipped = VcmiEnv.calc_reward(analysis, self.reward_scaling_factor, self.reward_clip_mod)
        obs = res.state
        term = res.is_battle_over
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

        return self.result.state, {"side": self.result.side}

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
        self.reward_scaling_factor = ARMY_VALUE_REF / (res.side0_army_value + res.side1_army_value)
        self.net_value_last = 0
        self.terminated = False
        self.truncated = False

        # Vars updated after other events
        self.analyzer = Analyzer(self.action_space.n)

    def _maybe_render(self, analysis):
        if self.render_each_step:
            print(self.render())

    #
    # A note on the static methods
    # Initially simply instance methods, I wanted to rip off the
    # reference to "self" as it introduced bugs related to the env state
    # (eg. some instance vars being used in calculations were not yet updated)
    # This approach is justified for training-critical methods only
    # (ie. no need to abstract out `render`, for example)

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

    @staticmethod
    def calc_reward(analysis, scaling_factor, clip_mod):
        rew = int(scaling_factor * (analysis.net_value + 5 * analysis.net_dmg))
        clipped = max(min(rew, clip_mod), -clip_mod) if clip_mod else rew
        return clipped, rew
