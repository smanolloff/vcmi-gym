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
from .decoder.decoder import Decoder, Battlefield
from .decoder.other import HexAction

from .pyconnector import (
    PyConnector,
    STATE_VALUE_NA,
    STATE_SIZE,
    STATE_SIZE_HEXES,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    N_ACTIONS,
    HEX_ACT_MAP,
    LINK_TYPES
)

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


# NOTE:
# Typical episode returns for step_fixed=0, dmg_mult=0, term_mult=0 (no errors):
# (-60, 60)
# * -/+ 60 if dmg_mult=1
# * -/+ 50 if term_mult=1
class RewardConfig(NamedTuple):
    err_exclusive: float
    step_fixed: float
    # step_round_mult: float
    # round_fixed: float
    # round_round_mult: float
    dmg_mult: float
    term_mult: float
    relval_mult: float


class RewardValues(NamedTuple):
    step_fixed: float = 0.0
    # step_round_mult: float = 0.0
    # round_fixed: float = 0.0
    # round_round_mult: float = 0.0
    dmg_mult: float = 0.0
    term_mult: float = 0.0
    relval_mult: float = 0.0


class EdgeIndexSpace(gym.spaces.Space):
    def __init__(self, num_nodes):
        super().__init__(shape=None, dtype=np.int64)
        assert num_nodes > 0
        self.max_index = num_nodes - 1

    def sample(self, num_edges=10):
        return np.random.uniform(low=0, high=self.max_index, size=(2, num_edges)).astype(np.int64)


class EdgeAttrsSpace(gym.spaces.Space):
    def __init__(self, attrs_space):
        super().__init__(shape=None, dtype=attrs_space.dtype)
        self.attrs_space = attrs_space

    def sample(self, num_edges=10):
        return np.stack([self.attrs_space.sample() for _ in range(num_edges)])


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]
    ROLES = ["attacker", "defender"]
    OPPONENTS = ["StupidAI", "BattleAI", "MMAI_BATTLEAI", "MMAI_MODEL", "MMAI_RANDOM", "OTHER_ENV"]

    STATE_SIZE = STATE_SIZE
    STATE_SIZE_HEXES = STATE_SIZE_HEXES
    STATE_SIZE_GLOBAL = STATE_SIZE_GLOBAL
    STATE_SIZE_ONE_PLAYER = STATE_SIZE_ONE_PLAYER

    ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)
    REWARD_SPACE = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)

    obs_space = gym.spaces.Box(low=STATE_VALUE_NA, high=1, shape=(STATE_SIZE,), dtype=np.float32)
    actmask_space = gym.spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=bool)
    links_space = gym.spaces.Dict({
        k: gym.spaces.Dict({
            "index": EdgeIndexSpace(num_nodes=165),
            "attrs": EdgeAttrsSpace(attrs_space=gym.spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32))
        })
        for k in LINK_TYPES.keys()
    })

    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": obs_space,
        "action_mask": actmask_space,
        "links": links_space
    })

    ENV_VERSION = 13

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
        vcmienv_logtag: str = "VcmiEnv-v13",
        role: str = "attacker",
        opponent: str = "StupidAI",
        opponent_model: Optional[str] = None,
        opponent_allow_mlbot: bool = True,      # only if opponent is MMAI_MODEL or OTHER_ENV (i.e. MMAI_USER)
        vcmi_stats_mode: str = "disabled",
        vcmi_stats_storage: str = "-",
        vcmi_stats_persist_freq: int = 100,
        allow_invalid_actions: bool = False,
        user_timeout: int = 0,
        vcmi_timeout: int = 0,
        boot_timeout: int = 0,
        random_heroes: int = 0,
        random_obstacles: int = 0,
        town_chance: int = 0,
        warmachine_chance: int = 0,
        random_terrain_chance: int = 0,
        random_stack_chance: int = 0,
        tight_formation_chance: int = 0,
        vip_chance: int = 0,
        opponent_vip_chance: int = 0,
        battlefield_pattern: str = "",
        mana_min: int = 0,
        mana_max: int = 0,
        swap_sides: int = 0,
        allow_retreat: bool = False,
        reward_err_exclusive: float = -10,
        # Applied on every step:
        reward_step_fixed: float = -1,          # reward = value

        # These require BATTLE_ROUND in obs (to add in future env version)
        # reward_step_round_mult: float = -1,     # reward = value * current_round
        # # Applied on first step of every round:
        # reward_round_fixed: float = -1,         # reward = value
        # reward_round_round_mult: float = -1,    # reward = value * current_round

        reward_dmg_mult: float = 1,
        reward_term_mult: float = 1,
        reward_relval_mult: float = 1,

        # If this is a secondary env in a dual-env scenario,
        # the "main" env is to be provided here.
        main_env: Optional["VcmiEnv"] = None
    ):
        """
        In a dual-env setup, env.connect() must be called manually for each env after init.
        Example:

            def env1_loop(env0, cond):
                env1 = VcmiEnv(opponent="OTHER_ENV", other_env=env0)
                env1.connect()
                # env1 can now be used as a regular env
                env1.reset()
                obs, rew, term, trunc, info = env1.step(env1.random_action())
                # ...

            def main():
                env0 = VcmiEnv(opponent="OTHER_ENV")
                env1_thread = Thread(target=env1_loop, args=(env0, cond), daemon=True)
                env1_thread.start()
                env0.connect()
                # env0 can now be used as a regular env
                env0.reset()
                obs, rew, term, trunc, info = env0.step(env0.random_action())
                # ...
        """

        assert vcmi_loglevel_global in self.__class__.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in self.__class__.VCMI_LOGLEVELS
        assert role in self.__class__.ROLES
        assert opponent in self.__class__.OPPONENTS, f"{opponent} in {self.__class__.OPPONENTS}"

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
        # </params>

        # accessed externally for vector env creation
        self.vcmienv_loglevel = vcmienv_loglevel
        self.vcmienv_logtag = vcmienv_logtag

        self.reward_cfg = RewardConfig(
            err_exclusive=float(reward_err_exclusive),
            step_fixed=float(reward_step_fixed),
            # step_round_mult=float(reward_step_round_mult),
            # round_fixed=float(reward_round_fixed),
            # round_round_mult=float(reward_round_round_mult),
            dmg_mult=float(reward_dmg_mult),
            term_mult=float(reward_term_mult),
            relval_mult=float(reward_relval_mult),
        )

        self.logger = log.get_logger(vcmienv_logtag, vcmienv_loglevel)
        self.logger.debug("Initializing...")
        self.main_env = main_env

        if opponent == "OTHER_ENV":
            opp = "MMAI_USER"
        else:
            opp = opponent

        if role == "attacker":
            attacker = "MMAI_USER"
            defender = opp
            # When mlbot is allowed for a MMAI_USER or MMAI_MODEL ai, it will
            # acts automatically if VIP-shooter army is detected.
            # We never want that for the main VcmiEnv player
            # => make sure mlbot is NOT allowed for our side
            attacker_allow_mlbot = False
            defender_allow_mlbot = opponent_allow_mlbot
            attacker_vip_chance = vip_chance
            defender_vip_chance = opponent_vip_chance
        else:
            attacker = opp
            defender = "MMAI_USER"
            attacker_allow_mlbot = opponent_allow_mlbot
            defender_allow_mlbot = False
            attacker_vip_chance = opponent_vip_chance
            defender_vip_chance = vip_chance

        if attacker == "MMAI_MODEL":
            attacker_model = opponent_model
            defender_model = ""
        elif defender == "MMAI_MODEL":
            attacker_model = ""
            defender_model = opponent_model
        else:
            attacker_model = ""
            defender_model = ""

        if main_env is not None:
            self.opponent = "OTHER_ENV"
            self.role = "defender" if main_env.role == "attacker" else "attacker"

        self.connector = PyConnector(
            loglevel=vcmienv_loglevel,
            logtag=vcmienv_logtag,
            maxlogs=100,
            user_timeout=user_timeout,
            vcmi_timeout=vcmi_timeout,
            boot_timeout=boot_timeout,
            allow_retreat=allow_retreat,
        )

        self.connector.start(
            mapname,
            seed or 0,
            random_heroes,
            random_obstacles,
            town_chance,
            warmachine_chance,
            random_stack_chance,
            tight_formation_chance,
            random_terrain_chance,
            attacker_vip_chance,
            defender_vip_chance,
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
            attacker_allow_mlbot,
            defender_allow_mlbot,
            vcmi_stats_mode,
            vcmi_stats_storage,
            vcmi_stats_persist_freq,
            main_connector=main_env.connector if main_env else None
        )

        if opponent == "OTHER_ENV":
            self.logger.warn("Dual-env setup detected -- will NOT connect automatically.")
        else:
            self.connector.connect_as(role)
            self.reset()  # needed to init vars

    @tracelog
    def connect(self):
        assert self.opponent == "OTHER_ENV"
        self.connector.connect_as(self.role)
        self.reset()  # needed to init vars

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

        # if action in [-1, 0]:
        #     self._reset_vars(res)

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

        bf = Decoder.decode(res.state, only_global=True)
        term = bf.global_stats.BATTLE_WINNER.v is not None
        trunc = self.steps_this_episode >= self.max_steps
        rewvals = VcmiEnv.calc_reward(res.errcode, term, trunc, bf, self.bf, self.reward_cfg)
        rew = sum(rewvals)
        res.mask[0] = False  # prevent retreats for now
        obs = self.__class__.build_obs(res)

        self._update_vars_after_step(action, obs, res, rew, term, trunc, bf, rewvals)
        self._maybe_render()

        info = self.__class__.build_info(res, term, trunc, bf, self.steps_this_episode, self.rewvals_total)

        return obs, rew, term, trunc, info

    @tracelog
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        result = self.connector.reset()
        bf = Decoder.decode(result.state, only_global=True)
        result.mask[0] = False  # prevent retreats for now
        obs = self.__class__.build_obs(result)

        self._reset_vars(result, obs, bf)
        if self.render_each_step:
            print(self.render())

        info = {"side": bf.global_stats.BATTLE_SIDE.v}
        return obs, info

    @tracelog
    def render(self):
        if self.render_mode == "ansi":
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
                self.bf.enemy_stats.VALUE_LOST_NOW_REL.v - self.bf.my_stats.VALUE_LOST_NOW_REL.v,
                self.bf.enemy_stats.VALUE_LOST_ACC_REL0.v - self.bf.my_stats.VALUE_LOST_ACC_REL0.v
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
        actions = np.where(self.obs["action_mask"][2:])[0]
        act_remainders = actions % 14
        grouped_actions = []
        for r in range(14):
            inds = actions[act_remainders == r]
            if len(inds):
                grouped_actions.append(inds + 2)

        if self.obs["action_mask"][1]:  # WAIT
            grouped_actions.append([1])

        # Choose a group (MOVE, AMOVE_TL, AMOVE_TR, ... etc)
        chosen_group = np.random.choice(range(len(grouped_actions)))

        # Choose an action from the group
        chosen_action = np.random.choice(grouped_actions[chosen_group])

        return chosen_action

    @staticmethod
    def build_obs(pyresult):
        return {
            "observation": pyresult.state,
            "action_mask": pyresult.mask,
            "links": dict(pyresult.links_dict),
        }

    @staticmethod
    def decode_obs(state):
        return Decoder.decode(state)

    #
    # private
    #

    def _update_vars_after_step(self, action, obs, res, rew, term, trunc, bf, rewvals):
        self.last_action = action
        self.steps_this_episode += 1
        self.obs = obs
        self.result = res
        self.reward = rew
        self.reward_total += rew
        self.rewvals = rewvals
        self.rewvals_total = RewardValues(*map(sum, zip(self.rewvals_total, rewvals)))  # element-wise sum
        self.terminated = term
        self.truncated = trunc
        self.bf = bf

    def _reset_vars(self, res, obs, bf):
        self.last_action = None
        self.steps_this_episode = 0
        self.obs = obs
        self.result = res
        self.reward = 0
        self.reward_total = 0
        self.rewvals = RewardValues()
        self.rewvals_total = RewardValues()
        self.reward_clip_abs_total = 0
        self.reward_clip_abs_max = 0
        self.terminated = False
        self.truncated = False
        self.bf = bf

    def _maybe_render(self):
        if self.render_each_step:
            self.render()

    #
    # NOTE:
    # info is read only after env termination
    # One-time values will be lost, put only only cumulatives/totals/etc.
    #
    @staticmethod
    def build_info(res, term, trunc, bf, steps_this_episode, rewvals_total):
        # Performance optimization
        if not (term or trunc):
            return dict(side=bf.global_stats.BATTLE_SIDE.v, step=steps_this_episode)

        return dict(
            side=bf.global_stats.BATTLE_SIDE.v,
            step=steps_this_episode,

            # round=bf.global_stats.BATTLE_ROUND.v,
            net_value=bf.enemy_stats.VALUE_LOST_ACC_REL0.v - bf.my_stats.VALUE_LOST_ACC_REL0.v,
            is_success=bf.is_battle_won or False,  # can be None if truncated
            reward_step_fixed=rewvals_total.step_fixed,
            # reward_step_round_mult=rewvals_total.step_round_mult,
            # reward_round_fixed=rewvals_total.round_fixed,
            # reward_round_round_mult=rewvals_total.round_round_mult,
            reward_dmg_mult=rewvals_total.dmg_mult,
            reward_term_mult=rewvals_total.term_mult,
            reward_relval_mult=rewvals_total.relval_mult,
        )

    @staticmethod
    def calc_reward(errcode, term, trunc, bf: Battlefield, bf_old: Battlefield, cfg: RewardConfig):
        if errcode > 0:
            return cfg.err_exclusive

        if trunc:
            # Trunc is bad: means model runs away forever
            # => punish as if entire army was lost
            my_dmg_received = bf.my_stats.ARMY_HP_NOW_REL.v
            my_value_lost = bf.my_stats.ARMY_VALUE_NOW_REL.v
            my_value_lost_acc = bf.my_stats.ARMY_VALUE_NOW_REL0.v
        else:
            my_dmg_received = bf.my_stats.DMG_RECEIVED_NOW_REL.v
            my_value_lost = bf.my_stats.VALUE_LOST_NOW_REL.v
            my_value_lost_acc = bf.my_stats.VALUE_LOST_ACC_REL0.v

        net_dmg = bf.enemy_stats.DMG_RECEIVED_NOW_REL.v - my_dmg_received
        net_value = bf.enemy_stats.VALUE_LOST_NOW_REL.v - my_value_lost
        net_value_acc = bf.enemy_stats.VALUE_LOST_ACC_REL0.v - my_value_lost_acc

        # is_new_round = battle_round > bf_old.global_stats.BATTLE_ROUND.v
        # battle_round = bf.global_stats.BATTLE_ROUND.v

        done = term or trunc

        return RewardValues(
            step_fixed=cfg.step_fixed,
            # step_round_mult=cfg.step_round_mult * battle_round,
            # round_fixed=is_new_round * cfg.round_fixed,
            # round_round_mult=is_new_round * cfg.round_round_mult * battle_round,
            dmg_mult=net_dmg * cfg.dmg_mult,
            term_mult=done * net_value_acc * cfg.term_mult,
            relval_mult=net_value * cfg.relval_mult,
        )
