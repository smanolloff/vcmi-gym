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
from typing import Optional

from ..util import log
from .decoder.decoder import Decoder
from .decoder.other import HexAction

from .pyprocconnector import (
    PyProcConnector,
    STATE_VALUE_NA,
    STATE_SIZE,
    N_ACTIONS,
    STATE_SIZE_MISC,
    STATE_SIZE_STACKS,
    STATE_SIZE_HEXES,
    STATE_SEQUENCE
)

from .pythreadconnector import PyThreadConnector

TRACE = False


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

    ALL_KEYS = ["side"] + SCALAR_VALUES

    def __setitem__(self, k, v):
        assert k in InfoDict.ALL_KEYS, f"Unknown info key: '{k}'"
        super().__setitem__(k, v)


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]
    ROLES = ["attacker", "defender"]
    OPPONENTS = ["StupidAI", "BattleAI", "MMAI_SCRIPT_SUMMONER", "MMAI_MODEL", "MMAI_RANDOM", "OTHER_ENV"]

    ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)

    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=STATE_VALUE_NA, high=1, shape=(STATE_SIZE,), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=bool)
    })

    STATE_SIZE_MISC = STATE_SIZE_MISC
    STATE_SIZE_HEXES = STATE_SIZE_HEXES
    STATE_SIZE_STACKS = STATE_SIZE_STACKS
    STATE_SEQUENCE = STATE_SEQUENCE
    ENV_VERSION = 4

    def __init__(
        self,
        mapname: str = "gym/A1.vmap",
        seed: Optional[int] = None,
        render_mode: str = "ansi",
        max_steps: int = 500,
        render_each_step: bool = False,
        vcmi_loglevel_global: str = "error",
        vcmi_loglevel_ai: str = "error",
        vcmi_loglevel_stats: str = "error",
        vcmienv_loglevel: str = "WARN",
        role: str = "attacker",
        opponent: str = "StupidAI",
        opponent_model: Optional[str] = None,
        vcmi_stats_mode: str = "disabled",
        vcmi_stats_storage: str = "-",
        vcmi_stats_persist_freq: int = 100,
        true_rng: bool = False,
        allow_invalid_actions: bool = False,
        user_timeout: int = 0,
        vcmi_timeout: int = 0,
        boot_timeout: int = 0,
        random_heroes: int = 0,
        random_obstacles: int = 0,
        town_chance: int = 0,
        warmachine_chance: int = 0,
        random_terrain_chance: int = 0,
        tight_formation_chance: int = 0,
        battlefield_pattern: str = "",
        mana_min: int = 0,
        mana_max: int = 0,
        swap_sides: int = 0,
        allow_retreat: bool = False,
        step_reward_mult: int = 1,
        step_reward_fixed: int = 0,
        step_reward_frac: int = 0,
        reward_dmg_factor: int = 5,
        term_reward_mult: int = 1,
        reward_clip_tanh_army_frac: float = 1.0,
        reward_army_value_ref: int = 1000,
        reward_dynamic_scaling: bool = False,
        conntype: str = "proc",
        other_env: Optional["VcmiEnv"] = None,
        nostart: bool = False
    ):
        assert vcmi_loglevel_global in self.__class__.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in self.__class__.VCMI_LOGLEVELS
        assert role in self.__class__.ROLES
        assert opponent in self.__class__.OPPONENTS, f"{opponent} in {self.__class__.OPPONENTS}"
        assert conntype in ["thread", "proc"]

        self.action_space = self.__class__.ACTION_SPACE
        self.observation_space = self.__class__.OBSERVATION_SPACE

        # <params>
        self.render_mode = render_mode
        self.allow_invalid_actions = allow_invalid_actions
        self.max_steps = max_steps
        self.render_each_step = render_each_step
        self.mapname = mapname
        self.role = role
        self.opponent = opponent
        self.opponent_model = opponent_model
        self.reward_clip_tanh_army_frac = reward_clip_tanh_army_frac
        self.reward_army_value_ref = reward_army_value_ref
        self.reward_dynamic_scaling = reward_dynamic_scaling
        self.reward_dmg_factor = reward_dmg_factor
        self.step_reward_fixed = step_reward_fixed
        self.step_reward_frac = step_reward_frac
        self.step_reward_mult = step_reward_mult
        self.term_reward_mult = term_reward_mult
        self.allow_retreat = allow_retreat
        self.other_env = other_env
        # </params>

        self.logger = log.get_logger("VcmiEnv-v4", vcmienv_loglevel)

        connector_class = PyProcConnector

        if conntype == "thread":
            connector_class = PyThreadConnector

        if other_env is None:
            self.connector = connector_class(
                vcmienv_loglevel,
                100,  # maxlogs
                user_timeout,
                vcmi_timeout,
                boot_timeout,
                allow_retreat,
            )

            if opponent == "OTHER_ENV":
                opponent = "MMAI_MODEL"

            if role == "attacker":
                attacker = "MMAI_USER"
                defender = opponent
            else:
                attacker = opponent
                defender = "MMAI_USER"

            if attacker == "MMAI_MODEL":
                attacker_model = opponent_model
                defender_model = ""
            elif defender == "MMAI_MODEL":
                attacker_model = ""
                defender_model = opponent_model
            else:
                attacker_model = ""
                defender_model = ""

            self.connector.start(
                mapname,
                seed or 0,
                random_heroes,
                random_obstacles,
                town_chance,
                warmachine_chance,
                random_terrain_chance,
                tight_formation_chance,
                battlefield_pattern,
                mana_min,
                mana_max,
                swap_sides,
                vcmi_loglevel_global,
                vcmi_loglevel_ai,
                vcmi_loglevel_stats,
                attacker,
                defender,
                attacker_model,
                defender_model,
                vcmi_stats_mode,
                vcmi_stats_storage,
                vcmi_stats_persist_freq,
            )
        else:
            assert conntype == "thread"
            assert opponent == "OTHER_ENV"
            assert role != other_env.role, "both envs must have different roles"
            self.logger.info("Using external connector (from another env)")
            self.connector = other_env.connector

        if conntype == "thread":
            self.connector.connect_as(role)

        # print("Action space: %s" % self.action_space)
        # print("Observation space: %s" % self.observation_space)

        # required to init vars
        # self._reset_vars(self.connector.reset())
        self.reset()

    @tracelog
    def step(self, action, fallback=False):
        # print(".", end="", flush=True)
        if (self.terminated or self.truncated) and action != -1:
            raise Exception("Reset needed")

        # Prevent VCMI exceptions (mid-battle retreats are not handled)
        if action == 0 and not self.allow_retreat:
            if self.allow_invalid_actions:
                self.logger.warn("Attempted a retreat action (action=%d), but retreat is not allowed" % action)
                defend = self._compute_defend_action(self.result.state)
                self.logger.warn("Falling to defend action=%d)" % defend)
                return self.step(defend, fallback=True)
            else:
                self.logger.error("Attempted a retreat action (action=%d), but retreat is not allowed" % action)

        res = self.connector.step(action)

        if action in [-1, 0]:
            self._reset_vars(res)

        if res.errcode > 0:
            if self.allow_invalid_actions:
                self.logger.warn("Attempted an invalid action (action=%d, errcode=%d)" % (action, res.errcode))
                assert not fallback, "action=%d is a fallback action, but is also invalid (errcode=%d)" % (action, res.errcode)
                defend = self.defend_action()
                print(self.render())
                self.logger.warn("Falling back to defend action=%d" % defend)
                return self.step(defend, fallback=True)
            else:
                self.logger.error("Attempted an invalid action (action=%d, errcode=%d)" % (action, res.errcode))
                print(self.render())
                raise Exception("Invalid action given: %s" % action)

        term = res.is_battle_over
        rew, rew_unclipped = self.calc_reward(res, fallback)

        res.actmask[0] = False  # prevent retreats for now
        obs = {"observation": res.state, "action_mask": res.actmask}

        trunc = self.actions_total >= self.max_steps

        self._update_vars_after_step(obs, res, rew, rew_unclipped, term, trunc)
        self._maybe_render()

        info = self.__class__.build_info(res, term, trunc, self.actions_total, self.net_value)

        return obs, rew, term, trunc, info

    @tracelog
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        result = self.connector.reset()
        obs = {"observation": result.state, "action_mask": result.actmask}
        self._reset_vars(result, obs)
        if self.render_each_step:
            print(self.render())

        self.result.actmask[0] = False  # prevent retreats for now

        info = {"side": self.result.side}
        return obs, info

    @tracelog
    def render(self):
        if self.render_mode == "ansi":
            return (
                "%s\n"
                "Step:      %-5s\n"
                "Reward:    %-5s (total: %s)\n"
                "Net value: %-5s (total: %s)"
            ) % (
                self.connector.render(),
                self.actions_total,
                round(self.reward, 2),
                round(self.reward_total, 2),
                self.net_value,
                self.net_value_total
            )

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

    def __del__(self):
        self.close()

    def decode(self):
        return self.__class__.decode_obs(self.result)

    def defend_action(self):
        bf = self.decode()
        astack = None
        for stack in bf.stacks:
            if stack.QUEUE_POS == 0:
                astack = stack
                break

        if not astack:
            raise Exception("Could not find active stack")

        # Moving to self results in a defend action
        h = bf.get_hex(astack.Y_COORD, astack.X_COORD)
        return h.action(HexAction.MOVE)

    def random_action(self):
        if self.terminated or self.truncated:
            return None
        actions = np.where(self.obs["action_mask"])[0]
        assert actions.any(), "action mask allows no actions, but last result was not terminal"
        return np.random.choice(actions)

    @staticmethod
    def decode_obs(pyresult):
        return Decoder.decode(pyresult.state, pyresult.is_battle_over)

    #
    # private
    #

    def _update_vars_after_step(self, obs, res, rew, rew_unclipped, term, trunc):
        reward_clip_abs = abs(rew - rew_unclipped)
        self.actions_total += 1
        self.net_dmg = res.dmg_dealt - res.dmg_received
        self.net_dmg_total += self.net_dmg
        self.net_value = res.value_killed - res.value_lost
        self.net_value_total += self.net_value
        self.obs = obs
        self.result = res
        self.reward = rew
        self.reward_total += rew
        self.reward_clip_abs_total += reward_clip_abs
        self.reward_clip_abs_max = max(reward_clip_abs, self.reward_clip_abs_max)
        self.terminated = term
        self.truncated = trunc

    def _reset_vars(self, res, obs):
        self.actions_total = 0
        self.net_dmg = 0
        self.net_dmg_total = 0
        self.net_value = 0
        self.net_value_total = 0
        self.obs = obs
        self.result = res
        self.reward = 0
        self.reward_total = 0
        self.reward_clip_abs_total = 0
        self.reward_clip_abs_max = 0
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

        self._step_reward_calc = self.step_reward_fixed + self.step_reward_frac * avg_army_value

    def _maybe_render(self):
        if self.render_each_step:
            self.render()

    #
    # NOTE:
    # info is read only after env termination
    # One-time values will be lost, put only only cumulatives/totals/etc.
    #
    @staticmethod
    def build_info(res, term, trunc, actions_total, net_value):
        # Performance optimization
        if not (term or trunc):
            return {"side": res.side, "step": actions_total}

        # XXX: do not use constructor args (bypasses validations)
        info = InfoDict()
        info["side"] = res.side
        info["net_value"] = net_value
        info["is_success"] = res.is_victorious

        # Return regular dict (wrappers insert arbitary keys)
        return dict(info)

    def calc_reward(self, res, fallback=False):
        if res.errcode > 0:
            return -100, -100

        if self.reward_dynamic_scaling:
            avg_army_value = (res.current_side0_army_value + res.current_side1_army_value) / 2
            self.reward_scaling_factor = self.reward_army_value_ref / avg_army_value
            self._step_reward_calc = self.step_reward_fixed + self.step_reward_frac * avg_army_value

        rew = self.net_value + self.reward_dmg_factor * self.net_dmg
        rew *= self.step_reward_mult

        if res.is_battle_over:
            vdiff = res.current_side0_army_value - res.current_side1_army_value
            vdiff = -vdiff if res.side == 1 else vdiff
            rew += (self.term_reward_mult * vdiff)

        rew += (self._step_reward_calc * (3 + int(fallback)))

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
