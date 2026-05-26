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
import random
from typing import Optional, NamedTuple

from ..util import log
from .decode import make_node_namedtuple_class, dump

from .pyconnector import (
    PyConnector,
    # STATE_VALUE_NA,
    # N_ACTIONS,
    MAX_ROUNDS,
    NODE_TYPES,
    COMBAT_RESULTS,
    ACTION_TYPES
)

TRACE = os.getenv("VCMIGYM_DEBUG", "0") == "1"

Global = make_node_namedtuple_class("Global", NODE_TYPES["Global"])
Player = make_node_namedtuple_class("Player", NODE_TYPES["Player"])
Unit = make_node_namedtuple_class("Unit", NODE_TYPES["Unit"])
Hex = make_node_namedtuple_class("Hex", NODE_TYPES["Hex"])
Action = make_node_namedtuple_class("Action", NODE_TYPES["Action"])

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
#
class RewardConfig(NamedTuple):
    err_exclusive: float
    step_fixed: float
    dmg_mult: float
    term_mult: float
    relval_mult: float

    # Progressive rewards is a per-step penalty which increases each round
    # until it reaches a specific value (cap), after which it stops growing.
    #
    # Use https://www.desmos.com/calculator to visualize:
    #
    #       Equation:
    #           \min\left(a\left(\max\left(\operatorname{floor}\left(x\right),b\right)-b\right)^{c},d\right)
    #
    #       Then add sliders for a, b & c
    #       NOTE: Actual rewards use the negative value of this result
    #
    # For example, with {a=0.1 b=9 c=2 d=15}, reward is:
    #
    # round <=9: 0
    # round 10: -0.1
    # round 11: -0.4
    # round 12: -0.9
    # round 13: -1.6
    # round 14: -2.5
    # round 15: -3.6
    # round 16: -4.9
    # round 17: -6.4
    # round 18: -8.1
    # round 19: -10.0
    # round 20: -12.1
    # round 21: -14.4
    # round >=22: -15
    # ...
    #
    # Cap=15 above is suitable in conjunction with reward_term_mult=0.01
    # (i.e. even if net_value_lost=500‰, net_dmg_received=500‰ + some acc value)
    #   totals to 1500‰ * reward_term_mult = 15
    # => cap our prog to 15 as well
    prog_base: float        # a
    prog_trigger: int       # b
    prog_exponent: float    # c
    prog_limit: int         # d


class RewardValues(NamedTuple):
    step_fixed: float = 0.0
    prog: float = 0.0
    dmg_mult: float = 0.0
    term_mult: float = 0.0
    relval_mult: float = 0.0


# class EdgeIndexSpace(gym.spaces.Space):
#     def __init__(self, num_nodes):
#         super().__init__(shape=None, dtype=np.int64)
#         assert num_nodes > 0
#         self.max_index = num_nodes - 1

#     def sample(self, num_edges=10):
#         return np.random.uniform(low=0, high=self.max_index, size=(2, num_edges)).astype(np.int64)


# class EdgeAttrsSpace(gym.spaces.Space):
#     def __init__(self, attrs_space):
#         super().__init__(shape=None, dtype=attrs_space.dtype)
#         self.attrs_space = attrs_space

#     def sample(self, num_edges=10):
#         return np.stack([self.attrs_space.sample() for _ in range(num_edges)])


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]
    ROLES = ["attacker", "defender"]
    OPPONENTS = ["StupidAI", "BattleAI", "MMAI_BATTLEAI", "MMAI_MODEL", "MMAI_RANDOM", "OTHER_ENV"]

    # ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)
    # REWARD_SPACE = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)

    # obs_space = gym.spaces.Box(low=STATE_VALUE_NA, high=1, shape=(STATE_SIZE,), dtype=np.float32)
    # actmask_space = gym.spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=bool)
    # links_space = gym.spaces.Dict({
    #     name: gym.spaces.Dict({
    #         "index": EdgeIndexSpace(num_nodes=165),
    #         "attrs": EdgeAttrsSpace(attrs_space=gym.spaces.Box(low=0, high=1, shape=(size,), dtype=np.float32))
    #     })
    #     for name, size in LINK_ATTR_SIZES.items()
    # })

    # OBSERVATION_SPACE = gym.spaces.Dict({
    #     "observation": obs_space,
    #     "action_mask": actmask_space,
    #     "links": links_space
    # })

    ENV_VERSION = 15

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
        vcmienv_logtag: str = "VcmiEnv-v15",
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
        random_primary_skills: int = 0,
        swap_sides: int = 0,
        allow_retreat: bool = False,
        reward_err_exclusive: float = -10,
        # Applied on every step:
        reward_step_fixed: float = -1,          # reward = value

        # These require BATTLE_ROUND in obs (to add in future env version)
        # See RewardConfig for more info
        reward_prog_base: float = 0.1,
        reward_prog_trigger: int = 9,
        reward_prog_exponent: float = 2,
        reward_prog_limit: float = 15,

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

        # self.action_space = self.__class__.ACTION_SPACE
        # self.observation_space = self.__class__.OBSERVATION_SPACE

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
            prog_base=float(reward_prog_base),
            prog_trigger=float(reward_prog_trigger),
            prog_exponent=float(reward_prog_exponent),
            prog_limit=float(reward_prog_limit),
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
            main_connector=main_env.connector if main_env else None,
            mapname=mapname,
            seed=(seed or 0),
            randomHeroes=random_heroes,
            randomObstacles=random_obstacles,
            townChance=town_chance,
            warmachineChance=warmachine_chance,
            randomStackChance=random_stack_chance,
            tightFormationChance=tight_formation_chance,
            randomTerrainChance=random_terrain_chance,
            leftVipChance=attacker_vip_chance,
            rightVipChance=defender_vip_chance,
            battlefieldPattern=battlefield_pattern,
            manaMin=mana_min,
            manaMax=mana_max,
            randomPrimarySkills=random_primary_skills,
            swapSides=swap_sides,
            loglevelGlobal=vcmi_loglevel_global,
            loglevelAI=vcmi_loglevel_ai,
            loglevelStats=vcmi_loglevel_stats,
            red=attacker,
            blue=defender,
            redModel=attacker_model,
            blueModel=defender_model,
            redAllowMlBot=attacker_allow_mlbot,
            blueAllowMlBot=defender_allow_mlbot,
            statsMode=vcmi_stats_mode,
            statsStorage=vcmi_stats_storage,
            statsPersistFreq=vcmi_stats_persist_freq,
        )

        if opponent == "OTHER_ENV":
            self.logger.info("Dual-env setup detected -- will NOT connect automatically.")
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

        obs = self.__class__.build_obs(res)
        gnode = Global.decode_one(obs["nodes"]["Global"][0])
        term = gnode.BATTLE_WINNER != COMBAT_RESULTS["NONE"]
        trunc = gnode.BATTLE_ROUND >= MAX_ROUNDS  # vcmi should have retreated
        rewvals = VcmiEnv.calc_reward(term, trunc, obs, self.pnodes_start, self.reward_cfg)
        rew = sum(rewvals)

        self._update_vars_after_step(action, obs, res, rew, term, trunc, rewvals)
        self._maybe_render()

        info = self.__class__.build_info(obs, term, trunc, self.side, self.pnodes_start, self.steps_this_episode, self.rewvals_total)

        return obs, rew, term, trunc, info

    @tracelog
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        result = self.connector.reset()
        obs = self.__class__.build_obs(result)

        self._reset_vars(result, obs)
        if self.render_each_step:
            print(self.render())

        info = {
            "side": self.side,
            "round": Global.decode_one(obs["nodes"]["Global"][0]).BATTLE_ROUND
        }

        return obs, info

    @tracelog
    def render(self):
        if self.render_mode == "ansi":
            return (
                "%s\n"
                "Step:          %-5s\n"
                "Reward:        %-5s (total: %s)\n"
                # "Net value (%%): %-5s (total: %s)"
            ) % (
                self.connector.render(),
                self.steps_this_episode,
                round(self.reward, 2),
                round(self.reward_total, 2),
                # self.bf.enemy_stats.VALUE_LOST_NOW_REL.v - self.bf.my_stats.VALUE_LOST_NOW_REL.v,
                # self.bf.enemy_stats.VALUE_LOST_ACC_REL0.v - self.bf.my_stats.VALUE_LOST_ACC_REL0.v
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

    # @staticmethod
    # def action_text(action, obs=None, bf=None):
    #     if action == -1:
    #         return ""

    #     if bf is None:
    #         bf = VcmiEnv.decode_obs(obs)

    #     res = "Action %d: " % action
    #     if action < 2:
    #         res += "Wait" if action else "Retreat"
    #     else:
    #         hex = bf.get_hex((action - 2) // len(HEX_ACT_MAP))

    #         act = list(HEX_ACT_MAP)[(action - 2) % len(HEX_ACT_MAP)]

    #         if hex.stack and hex.stack.QUEUE.v[0] == 1 and act == "MOVE" and not hex.IS_REAR.v:
    #             act = "Defend"

    #         res += "%s (y=%s x=%s)" % (act, hex.Y_COORD.v, hex.X_COORD.v)
    #     return res

    def __del__(self):
        self.close()

    def decode(self):
        return self.__class__.decode_obs(self.result.state)

    def defend_action(self, bf=None):
        active_ids = self.result.get_active_action_ids()
        active_actions = Action.decode_many(self.obs["nodes"]["Action"][active_ids])

        return next(
            action_id
            for action_id, action in zip(active_ids, active_actions)
            if action.ACTION_TYPE == ACTION_TYPES["DEFEND"]
        )

    def random_action(self):
        if self.terminated or self.truncated:
            return None

        return random.choice(self.result.get_active_action_ids())

    @staticmethod
    def build_obs(res):
        return {
            "nodes": res.get_nodes(),
            "edges": res.get_edges()
        }

    @staticmethod
    def decode_obs(state):
        return Decoder.decode(state)

    #
    # private
    #

    def _update_vars_after_step(self, action, obs, res, rew, term, trunc, rewvals):
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

    def _reset_vars(self, res, obs):
        gnode = Global.decode_one(obs["nodes"]["Global"][0])
        pnodes = Player.decode_many(obs["nodes"]["Player"])

        assert pnodes[0].BATTLE_SIDE == 0, "player node are not ordered by battle side"

        self.gnode_start = Global.decode_one(obs["nodes"]["Global"][0])
        self.pnodes_start = Player.decode_many(obs["nodes"]["Player"])
        self.side = int(pnodes[1].IS_ACTIVE)
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

    def _maybe_render(self):
        if self.render_each_step:
            self.render()

    #
    # NOTE:
    # info is read only after env termination
    # One-time values will be lost, put only only cumulatives/totals/etc.
    #
    @staticmethod
    def build_info(obs, term, trunc, side, pnodes_start, steps_this_episode, rewvals_total):
        gnode = Global.decode_one(obs["nodes"]["Global"][0])

        # Performance optimization
        if not (term or trunc):
            return dict(
                side=side,
                round=gnode.BATTLE_ROUND,
                step=steps_this_episode
            )

        pnodes = Player.decode_many(obs["nodes"]["Player"])

        if pnodes[0].IS_ACTIVE:
            me, enemy = pnodes
            me0, enemy0 = pnodes_start
        else:
            enemy, me = pnodes
            enemy0, me0 = pnodes_start

        my_rel0_diff = me0.ARMY_VALUE_NOW_REL0 - me.ARMY_VALUE_NOW_REL0
        enemy_rel0_diff = enemy0.ARMY_VALUE_NOW_REL0 - enemy.ARMY_VALUE_NOW_REL0
        efficiency = enemy_rel0_diff - my_rel0_diff

        return dict(
            side=side,
            round=gnode.BATTLE_ROUND,
            step=steps_this_episode,
            net_value=efficiency,
            is_success=gnode.BATTLE_WINNER == side,
            reward_step_fixed=rewvals_total.step_fixed,
            reward_prog=rewvals_total.prog,
            reward_dmg_mult=rewvals_total.dmg_mult,
            reward_term_mult=rewvals_total.term_mult,
            reward_relval_mult=rewvals_total.relval_mult,
        )

    @staticmethod
    def calc_reward(term, trunc, obs, pnodes_start, cfg: RewardConfig):
        gnode = Global.decode_one(obs["nodes"]["Global"][0])
        pnodes = Player.decode_many(obs["nodes"]["Player"])

        if pnodes[0].IS_ACTIVE:
            me, enemy = pnodes
            me0, enemy0 = pnodes_start
        else:
            enemy, me = pnodes
            enemy0, me0 = pnodes_start

        if trunc:
            # Trunc is bad: means model runs away forever
            # => punish as if entire army was lost
            my_dmg_received = me.ARMY_HP_NOW_REL
            my_value_lost = me.ARMY_VALUE_NOW_REL
        else:
            my_dmg_received = me.DMG_RECEIVED_NOW_REL
            my_value_lost = me.VALUE_LOST_NOW_REL

        net_dmg = enemy.DMG_RECEIVED_NOW_REL - my_dmg_received
        net_value = enemy.VALUE_LOST_NOW_REL - my_value_lost

        # See note in RewardConfig
        a = cfg.prog_base
        b = cfg.prog_trigger
        c = cfg.prog_exponent
        d = cfg.prog_limit
        x = gnode.BATTLE_ROUND
        prog = -min(a*(max(b, int(x)) - b)**c, d)

        done = term or trunc

        # Idea:
        # If I win with final rel0=100 (enemy rel0=0 ofc), term reward depends
        # on the starting conditions:
        #  a) if enemy started at an advantage (e.g. initial rel0 were 300 vs 700)
        #       => term_reward = 500
        #  b) if I started at an advantage(e.g. initial rel0 were 700 vs 300)
        #       => term_reward=-300
        #       i.e. I won, but lost way more than the opponent
        my_rel0_diff = me0.ARMY_VALUE_NOW_REL0 - me.ARMY_VALUE_NOW_REL0
        enemy_rel0_diff = enemy0.ARMY_VALUE_NOW_REL0 - enemy.ARMY_VALUE_NOW_REL0
        efficiency = enemy_rel0_diff - my_rel0_diff

        return RewardValues(
            step_fixed=cfg.step_fixed,
            prog=prog,
            dmg_mult=net_dmg * cfg.dmg_mult,
            term_mult=done * cfg.term_mult * (enemy_rel0_diff - my_rel0_diff),
            relval_mult=net_value * cfg.relval_mult,
        )
