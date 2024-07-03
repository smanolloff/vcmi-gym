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

from ..util import log
from .analyzer import Analyzer, ActionType
from .decoder.decoder import Decoder
from .pyconnector import (
    PyConnector,
    STATE_VALUE_NA,
    STATE_SIZE,
    N_ACTIONS,
)

TRACE = True


def tracelog(func, maxlen=80):
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
    ROLES = ["MMAI_USER", "MMAI_MODEL", "MMAI_SCRIPT_SUMMONER", "StupidAI", "BattleAI"]

    CONNECTOR_CLASS = PyConnector

    # NOTE: removing action=0 to prevent agents from freely retreating for now
    ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)
    OBSERVATION_SPACE = gym.spaces.Box(
        low=STATE_VALUE_NA,
        high=1,
        shape=(STATE_SIZE,),
        dtype=np.float32
    )

    def __init__(
        self,
        mapname,
        seed=None,
        render_mode="ansi",
        max_steps=500,
        render_each_step=False,
        vcmi_loglevel_global="error",  # vcmi loglevel
        vcmi_loglevel_ai="error",  # vcmi loglevel
        vcmi_loglevel_stats="error",  # vcmi loglevel
        vcmienv_loglevel="WARN",  # python loglevel
        consecutive_error_reward_factor=-1,  # unused
        attacker="MMAI_USER",  # MMAI_USER / MMAI_MODEL / StupidAI / BattleAI
        defender="StupidAI",  # MMAI_USER / MMAI_MODEL / StupidAI / BattleAI
        attacker_model=None,  # MPPO zip model (if attacker=MMAI_MODEL)
        defender_model=None,  # MPPO zip model (if defender=MMAI_MODEL)
        vcmi_stats_mode="disabled",
        vcmi_stats_storage="-",
        vcmi_stats_persist_freq=100,
        vcmi_stats_sampling=0,
        vcmi_stats_score_var=0.4,
        true_rng=True,
        sparse_info=False,
        allow_invalid_actions=False,
        user_timeout=0,  # seconds - user input might take very long
        vcmi_timeout=0,  # seconds - VCMI occasionally writes stats DB to disk (see vcmi_stats_persist_freq)
        boot_timeout=0,  # seconds - needed as VCMI boot sometimes hangs with a memleak
        reward_dmg_factor=5,
        reward_clip_tanh_army_frac=1,  # max action reward relative to starting army value
        reward_army_value_ref=0,  # scale rewards relative to starting army value (0=no scaling)
        random_heroes=0,  # pick heroes at random each Nth combat (disabled if 0*)
        random_obstacles=0,  # place obstacles at random each Nth combat (disabled if 0*)
        town_chance=0,  # chance for an in-town (map must have towns with fort)
        warmachine_chance=0,  # chance for each of 3 war machines to appear in the army
        mana_min=0,  # hero mana at the start of combat is set at random in the [min, max] range
        mana_max=0,  # see mana_min
        swap_sides=0,  # swap combat sides at each Nth combat (disabled if 0*)
        step_reward_fixed=0,  # fixed reward
        step_reward_mult=1,
        term_reward_mult=1,  # at term step, reward = diff in total army values
        allow_retreat=False,  # whether to always mask the "0" action
    ):
        assert vcmi_loglevel_global in self.__class__.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in self.__class__.VCMI_LOGLEVELS
        assert attacker in self.__class__.ROLES
        assert defender in self.__class__.ROLES
        assert attacker == "MMAI_USER" or defender == "MMAI_USER", "an MMAI_USER role is required"

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
        self.reward_clip_tanh_army_frac = reward_clip_tanh_army_frac
        self.reward_army_value_ref = reward_army_value_ref
        self.reward_dmg_factor = reward_dmg_factor
        self.step_reward_fixed = step_reward_fixed
        self.step_reward_mult = step_reward_mult
        self.term_reward_mult = term_reward_mult
        self.allow_retreat = allow_retreat
        # </params>

        self.logger = log.get_logger("VcmiEnv", vcmienv_loglevel)

        self.connector = self.__class__.CONNECTOR_CLASS(
            vcmienv_loglevel,
            user_timeout,
            vcmi_timeout,
            boot_timeout,
            allow_retreat,
        )

        result = self.connector.start(
            mapname,
            seed or 0,
            random_heroes,
            random_obstacles,
            town_chance,
            warmachine_chance,
            mana_min,
            mana_max,
            swap_sides,
            vcmi_loglevel_global,
            vcmi_loglevel_ai,
            vcmi_loglevel_stats,
            attacker,
            defender,
            attacker_model or "",
            defender_model or "",
            vcmi_stats_mode,
            vcmi_stats_storage,
            vcmi_stats_persist_freq,
            vcmi_stats_sampling,
            vcmi_stats_score_var,
            true_rng,
        )

        self.action_space = self.__class__.ACTION_SPACE
        self.observation_space = self.__class__.OBSERVATION_SPACE

        # print("Action space: %s" % self.action_space)
        # print("Observation space: %s" % self.observation_space)

        # required to init vars
        self._reset_vars(result)

    @tracelog
    def step(self, action):
        if self.terminated or self.truncated:
            raise Exception("Reset needed")

        # Prevent VCMI exceptions (mid-battle retreats are not handled)
        if action == 0 and not self.allow_retreat:
            raise Exception("Retreat is not allowed")

        res = self.connector.act(action)
        if res.errcode > 0 and not self.allow_invalid_actions:
            self.logger.warn("errcode=%d" % res.errcode)

        analysis = self.analyzer.analyze(action, res)
        term = res.is_battle_over
        rew, rew_unclipped = self.calc_reward(analysis, res)

        obs = res.state
        trunc = self.analyzer.actions_count >= self.max_steps

        self._update_vars_after_step(res, rew, rew_unclipped, term, trunc, analysis)
        self._maybe_render(analysis)

        info = self.__class__.build_info(
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
            self.render()

        return self.result.state, {"side": self.result.side}

    @tracelog
    def render(self):
        if self.render_mode == "ansi":
            print((
                "%s\n"
                "Step:      %-5s\n"
                "Reward:    %-5s (total: %s)\n"
                "Net value: %-5s (total: %s)"
            ) % (
                self.connector.render_ansi(),
                self.analyzer.actions_count,
                round(self.reward, 2),
                round(self.reward_total, 2),
                self.net_value_last,
                self.analyzer.net_value
            ))

        elif self.render_mode == "rgb_array":
            gym.logger.warn("Rendering RGB arrays is not implemented")
        elif self.render_mode is None:
            gym.logger.warn("Cannot render with no render mode set")

    @tracelog
    def close(self):
        self.logger.info("Closing env...")

        self.connector.shutdown()
        self.logger.info("Env closed")
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()

    @tracelog
    def action_mask(self):
        return self.result.actmask

    # To use attnmask, code in pyconnector.py and BAI/v1/state.cpp
    # must be uncommented and both VCMI and connector must be recompiled.
    def attn_mask(self):
        # return self.result.attnmask
        raise Exception("attn_mask disabled for performance reasons")

    def decode(self):
        return self.__class__.decode_obs(self.result.state)

    @staticmethod
    def decode_obs(obs):
        raise NotImplementedError("v3 decoding is not implemented")
        # return Decoder.decode(obs)

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
        avg_army_value = (res.initial_side0_army_value + res.initial_side1_army_value) / 2
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
        self.analyzer = Analyzer(self.action_space.n)

    def _maybe_render(self, analysis):
        if self.render_each_step:
            self.render()

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
        if res.errcode > 0:
            return -100, -100

        rew = analysis.net_value + self.reward_dmg_factor * analysis.net_dmg
        rew *= self.step_reward_mult

        if res.is_battle_over:
            vdiff = res.current_side0_army_value - res.current_side1_army_value
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
