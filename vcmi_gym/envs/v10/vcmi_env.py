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
from typing import Optional, NamedTuple

from ..util import log
from .decoder.decoder import Decoder
from .decoder.other import HexAction

from .pyprocconnector import (
    PyProcConnector,
    STATE_VALUE_NA,
    STATE_SIZE,
    STATE_SIZE_HEXES,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    N_ACTIONS,
    HEX_ACT_MAP
)

from .pythreadconnector import PyThreadConnector

TRACE = os.getenv("VCMIGYM_DEBUG", "0") == "1"


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


# NOTE:
# Typical episode returns for step_fixed=0, dmg_mult=0, term_mult=0 (no errors):
# (-60, 60)
# * -/+ 60 if dmg_mult=1
# * -/+ 50 if term_mult=1
class RewardConfig(NamedTuple):
    err_exclusive: float
    step_fixed: float
    dmg_mult: float
    term_mult: float


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]
    ROLES = ["attacker", "defender"]
    OPPONENTS = ["StupidAI", "BattleAI", "MMAI_SCRIPT_SUMMONER", "MMAI_MODEL", "MMAI_RANDOM", "OTHER_ENV"]

    STATE_SIZE = STATE_SIZE
    STATE_SIZE_HEXES = STATE_SIZE_HEXES
    STATE_SIZE_GLOBAL = STATE_SIZE_GLOBAL
    STATE_SIZE_ONE_PLAYER = STATE_SIZE_ONE_PLAYER

    ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)
    REWARD_SPACE = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)

    obs_space = gym.spaces.Box(low=STATE_VALUE_NA, high=1, shape=(STATE_SIZE,), dtype=np.float32)
    actmask_space = gym.spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=bool)

    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": obs_space,
        "action_mask": actmask_space,
        "transitions": gym.spaces.Dict({
            "observations": gym.spaces.Sequence(obs_space, stack=True),
            "action_masks": gym.spaces.Sequence(actmask_space, stack=True),
            "actions": gym.spaces.Sequence(ACTION_SPACE, stack=True),
            "rewards": gym.spaces.Sequence(REWARD_SPACE, stack=True),
        })
    })

    ENV_VERSION = 10

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
        reward_err_exclusive: float = -10,
        reward_step_fixed: float = -1,
        reward_dmg_mult: float = 1,
        reward_term_mult: float = 1,
        conntype: str = "thread",
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
        self.allow_retreat = allow_retreat
        self.other_env = other_env
        # </params>

        self.reward_cfg = RewardConfig(
            err_exclusive=float(reward_err_exclusive),
            step_fixed=float(reward_step_fixed),
            dmg_mult=float(reward_dmg_mult),
            term_mult=float(reward_term_mult),
        )

        self.logger = log.get_logger("VcmiEnv-v10", vcmienv_loglevel)

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
                defend = self.defend_action()
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

        intbfs = [Decoder.decode(s, only_global=True) for s in res.intstates]
        term = intbfs[-1].global_stats.BATTLE_WINNER.v is not None
        intrews = self.__class__.calc_rewards(self.reward_cfg, res.errcode, res.intstates, intbfs)
        rew = sum(intrews[1:])
        res.intmasks[:, 0] = False  # prevent retreats for now
        obs = self.__class__.build_obs(res, intrews)
        trunc = self.steps_this_episode >= self.max_steps

        self._update_vars_after_step(action, obs, res, rew, term, trunc, intbfs)
        self._maybe_render()

        info = self.__class__.build_info(res, term, trunc, intbfs[-1], self.steps_this_episode)

        return obs, rew, term, trunc, info

    @tracelog
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        result = self.connector.reset()

        intbfs = [Decoder.decode(s, only_global=True) for s in result.intstates]

        # Typically, there's reward before our first turn
        # (rew is not even returned by this call)
        # However, for sample collection purposes, we want the rewards even here
        # (for reward prediction training)
        # intrews = np.zeros(result.intstates.shape[0], dtype=np.float32)
        # intrews[0] = float("nan")

        intbfs = [Decoder.decode(s, only_global=True) for s in result.intstates]
        intrews = self.__class__.calc_rewards(self.reward_cfg, 0, result.intstates, intbfs)

        result.intmasks[:, 0] = False  # prevent retreats for now
        obs = self.__class__.build_obs(result, intrews)

        self._reset_vars(result, obs, intbfs)
        if self.render_each_step:
            print(self.render())

        info = {"side": intbfs[-1].global_stats.BATTLE_SIDE.v}
        return obs, info

    @tracelog
    def render(self):
        if self.render_mode == "ansi":
            bf = self.intbfs[-1]
            return (
                "%s\n"
                "Step:          %-5s\n"
                "Reward:        %-5s (total: %s)\n"
                "Net value (%%): %-5s (total: %s)"
            ) % (
                self.connector.render(),
                self.steps_this_episode,
                round(self.reward, 2),
                round(self.reward_total, 2),
                bf.enemy_stats.VALUE_LOST_NOW_REL.v - bf.my_stats.VALUE_LOST_NOW_REL.v,
                bf.enemy_stats.VALUE_LOST_ACC_REL0.v - bf.my_stats.VALUE_LOST_ACC_REL0.v
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

    @staticmethod
    def action_text(action, obs=None, bf=None):
        if action == -1:
            return ""

        if bf is None:
            bf = VcmiEnv.decode_obs(obs)

        res = "Action %d: " % action
        if action < 2:
            res += "Wait" if action else "Retreat"
        else:
            hex = bf.get_hex((action - 2) // len(HEX_ACT_MAP))

            act = list(HEX_ACT_MAP)[(action - 2) % len(HEX_ACT_MAP)]

            if hex.stack and hex.stack.QUEUE.v[0] == 1 and act == "MOVE" and not hex.IS_REAR.v:
                act = "Defend"

            res += "%s (y=%s x=%s)" % (act, hex.Y_COORD.v, hex.X_COORD.v)
        return res

    def render_transitions(self, add_regular_render=True):
        def prepare(obs, action, reward):
            import re
            bf = Decoder.decode(obs)
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            rewtxt = "" if reward is None else "Reward: %s" % round(reward, 2)
            render = {}
            render["bf_lines"] = bf.render_battlefield()[0][:-1]
            render["bf_len"] = [len(l) for l in render["bf_lines"]]
            render["bf_printlen"] = [len(ansi_escape.sub('', l)) for l in render["bf_lines"]]
            render["bf_maxlen"] = max(render["bf_len"])
            render["bf_maxprintlen"] = max(render["bf_printlen"])
            render["bf_lines"].insert(0, rewtxt.ljust(render["bf_maxprintlen"]))
            render["bf_printlen"].insert(0, len(render["bf_lines"][0]))
            render["bf_lines"] = [l + " "*(render["bf_maxprintlen"] - pl) for l, pl in zip(render["bf_lines"], render["bf_printlen"])]
            render["bf_lines"].append(VcmiEnv.action_text(action, bf=bf).rjust(render["bf_maxprintlen"]))
            return render["bf_lines"]

        trans = self.obs["transitions"]
        bfields = [prepare(s, a, None) for s, a, r in zip(trans["observations"][:1], trans["actions"][:1], trans["rewards"][:1])]
        bfields += [prepare(s, a, r) for s, a, r in zip(trans["observations"][1:], trans["actions"][1:], trans["rewards"][1:])]

        # for i in range(len(bfields)):
        print("")
        print("\n".join([(" → ".join(rowlines)) for rowlines in zip(*bfields)]))
        print("")

        if add_regular_render:
            print(self.render())

    def __del__(self):
        self.close()

    def decode(self):
        return self.__class__.decode_obs(self.result.state)

    def defend_action(self, bf=None):
        if bf is None:
            bf = Decoder.decode(self.obs["transitions"]["observations"][-1])

        ahex = None
        for hex in [h for row in bf.hexes for h in row]:
            if hex.stack and hex.stack.QUEUE.v[0] == 1 and not hex.IS_REAR.v:
                ahex = hex
                break

        if not ahex:
            raise Exception("Could not find active hex")

        return hex.action(HexAction.MOVE)

    def random_action(self):
        if self.terminated or self.truncated:
            return None
        actions = np.where(self.obs["action_mask"])[0]
        if not actions.any():
            print("?!?!!?!?!??!?!!?!?")
            self.render_transitions()
            print(self.render())
            raise Exception("action mask allows no actions, but last result was not terminal")
        return np.random.choice(actions)

    @staticmethod
    def build_obs(pyresult, intrews):
        return {
            "observation": pyresult.intstates[-1],
            "action_mask": pyresult.intmasks[-1],
            "transitions": {
                "observations": pyresult.intstates,
                "action_masks": pyresult.intmasks,
                "actions": pyresult.intactions,
                "rewards": intrews
            }
        }

    @staticmethod
    def decode_obs(state):
        return Decoder.decode(state)

    @staticmethod
    def calc_rewards(reward_cfg, errcode, intstates, intbfs):
        # Since a "fixed step reward" is ill-defined in a transition context
        # => set `step_fixed=0` for all but the first transition
        # i.e. immediately after we act
        mid_rewcfg = reward_cfg._replace(step_fixed=0)

        def calcrew(err, state, rewcfg, bf):
            return VcmiEnv.calc_reward(err, bf, rewcfg)

        initial = [float("nan")]

        # NOTE: `first` is [] when calculating rewards after reset
        first = [calcrew(errcode, s, reward_cfg, bf) for s, bf in zip(intstates[1:2], intbfs[1:2])]

        # NOTE: `rest` is [] when there were no enemy actions after ours
        rest = [calcrew(0, s, mid_rewcfg, bf) for s, bf in zip(intstates[2:], intbfs[2:])]

        return initial + first + rest

    #
    # private
    #

    def _update_vars_after_step(self, action, obs, res, rew, term, trunc, intbfs):
        self.last_action = action
        self.steps_this_episode += 1
        self.obs = obs
        self.result = res
        self.reward = rew
        self.reward_total += rew
        self.terminated = term
        self.truncated = trunc
        self.intbfs = intbfs

    def _reset_vars(self, res, obs, intbfs):
        self.last_action = None
        self.steps_this_episode = 0
        self.obs = obs
        self.result = res
        self.reward = 0
        self.reward_total = 0
        self.reward_clip_abs_total = 0
        self.reward_clip_abs_max = 0
        self.terminated = False
        self.truncated = False
        self.intbfs = intbfs

    def _maybe_render(self):
        if self.render_each_step:
            self.render()

    #
    # NOTE:
    # info is read only after env termination
    # One-time values will be lost, put only only cumulatives/totals/etc.
    #
    @staticmethod
    def build_info(res, term, trunc, bf, steps_this_episode):
        # Performance optimization
        if not (term or trunc):
            return {"side": bf.global_stats.BATTLE_SIDE.v, "step": steps_this_episode}

        # XXX: do not use constructor args (bypasses validations)
        info = InfoDict()
        info["side"] = bf.global_stats.BATTLE_SIDE.v
        info["net_value"] = bf.enemy_stats.VALUE_LOST_ACC_REL0.v - bf.my_stats.VALUE_LOST_ACC_REL0.v
        info["is_success"] = bf.is_battle_won or False  # can be None if truncated

        # Return regular dict (wrappers insert arbitary keys)
        return dict(info)

    @staticmethod
    def calc_reward(errcode, bf, cfg: RewardConfig):
        if errcode > 0:
            return cfg.err_exclusive

        net_value = bf.enemy_stats.VALUE_LOST_NOW_REL.v - bf.my_stats.VALUE_LOST_NOW_REL.v
        net_dmg = bf.enemy_stats.DMG_RECEIVED_NOW_REL.v - bf.my_stats.DMG_RECEIVED_NOW_REL.v
        net_value_acc = bf.enemy_stats.VALUE_LOST_ACC_REL0.v - bf.my_stats.VALUE_LOST_ACC_REL0.v
        ended = bf.global_stats.BATTLE_WINNER.v is not None
        term_rew = net_value_acc if ended else 0

        # print(f"net_value: {net_value}, net_dmg: {net_dmg}, step_fixed: {cfg.step_fixed}")
        # print("REWARD: net_value=%d (%d - %d)" % (net_value, bf.enemy_stats.VALUE_LOST_REL.v, bf.my_stats.VALUE_LOST_REL.v))

        return (
            cfg.step_fixed
            + net_value
            + net_dmg * cfg.dmg_mult
            + term_rew * cfg.term_mult
        )
