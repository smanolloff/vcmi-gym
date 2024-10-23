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
from .analyzer import Analyzer, ActionType
from .decoder.decoder import Decoder

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
    ROLES = ["attacker", "defender"]
    OPPONENTS = ["StupidAI", "BattleAI", "MMAI_SCRIPT_SUMMONER", "MMAI_MODEL", "OTHER_ENV"]

    # NOTE: removing action=0 to prevent agents from freely retreating for now
    ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)
    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=STATE_VALUE_NA, high=1, shape=(STATE_SIZE,), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=bool)
    })

    STATE_SIZE_MISC = STATE_SIZE_MISC
    STATE_SIZE_HEXES = STATE_SIZE_HEXES
    STATE_SIZE_STACKS = STATE_SIZE_STACKS
    STATE_SEQUENCE = STATE_SEQUENCE

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
        sparse_info: bool = False,
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
        other_env: Optional["VcmiEnv"] = None
    ):
        # Initialization code here

        """
        Initialize the VCMI gym environment.

        Required parameters:

        * mapname (str)
            Path to the VCMI map. Relative paths are resolved with respect
            to the "maps/" directory.
            Example: "gym/A1.vmap"

        Optional parameters:

        * render_mode (str)
            Gym render mode. Only "ansi" is supported.
            Values: "ansi"
            Default: "ansi"

        * seed (int)
            VCMI `--seed` option. Speficies the RNG seed to use.
            A value of `0` will generate a random seed.
            Values: 0 < seed < 2^31
            Default: 0

        * max_steps (int)
            Max episode length in steps. A value of `0` means no limit.
            Values: 0 < max_steps
            Default: 500

        * render_each_step (bool)
            Automatically invoke render() after each step.
            Default: False

        * vcmi_loglevel_global (str)
            VCMI `--loglevel-global` option.
            Values: "trace" | "debug" | "info" | "warn" | "error"
            Default: "error"

        * vcmi_loglevel_ai (str)
            VCMI `--loglevel-ai` option.
            Values: "trace" | "debug" | "info" | "warn" | "error"
            Default: "error"

        * vcmi_loglevel_stats (str)
            VCMI `--loglevel-stats` option.
            Values: "trace" | "debug" | "info" | "warn" | "error"
            Default: "error"

        * vcmienv_loglevel (str)
            The gym environment's log level.
            Values: "DEBUG" | "INFO" | "WARN" | "ERROR"
            Default: "WARN"

        * role (str)
            VCMI battle perspective of this VcmiEnv.
            Values: "attacker" | "defender"
            Default: "attacker"

        * opponent (str)
            Opponent to play against.
            - If "StupidAI", "BattleAI" or "MMAI_SCRIPT_SUMMONER" is given,
            VCMI will automatically act on behalf of the attacking player using
            scripted bot logic.
            - If "MMAI_MODEL" is given, a pre-trained model will be loaded from
            a file specified by the `opponent_model` parameter and will
            automatically act on behalf of the opposing player using the
            model's predictions.
            - If "OTHER_ENV" is given, the opponent will be another VcmiEnv.
            For this to work, the two environments must be connected via the
            `other_env` argument.
            Values: "StupidAI" | "BattleAI" | "MMAI_SCRIPT_SUMMONER" | "MMAI_MODEL" | "OTHER_ENV"
            Default: "StupidAI"

        * opponent_model (str)
            Path to a pre-trained torch JIT model file.
            Ignored unless `opponent` is "MMAI_MODEL".
            Default: None

        * vcmi_stats_mode (str)
            VCMI `--stats-mode` option. Specifies whether per-hero statistics
            should be collected after each battle (wins, losses). Used for
            creating and rebalancing training maps.
            Values: "disabled" | "red" | "blue"
            Default: "disabled"

        * vcmi_stats_storage (str)
            VCMI `--stats-storage` option. Specifies the location of the
            sqlite3 database where statistics will be read from and written to.
            A value of "-" means that an in-memory database will be created and
            used instead.
            Ignored if `vcmi_stats_mode` is "disabled".
            Default: "-"

        * vcmi_stats_persist_freq (int)
            VCMI `--stats-persist-freq` option. Specifies how often (in number
            of battles) to write data to the database.
            A value of `0` means never write to the database.

        * true_rng (bool)
            Deprecated; do not use.

        * sparse_info (bool)
            Whether to return a minimalistic `info` dict on non-terminal steps.
            Default: True

        * allow_invalid_actions (bool)
            Whether to mute logged warnings when an attempted action is invalid.
            Default: False

        * user_timeout (int)
            Timeout in seconds while waiting for user input.
            A value of 0 means no timeout.
            Default: 0

        * vcmi_timeout (int)
            Timeout in seconds while waiting for a response from VCMI.
            A value of 0 means no timeout.
            Default: 0

        * boot_timeout (int)
            Same as `vcmi_timeout`, but for during the first observation just
            after the environment has started.

        * random_heroes (int)
            VCMI `--random-heroes` option. Specifies how often (in number of
            battles) should heroes in the battle be changed, effectively
            changing the army compositions.
            If `0`, all battles will use the same two heroes (and armies).
            If `1`, each battle will use a different pair of heroes from a
            randomized hero pool. When all possible hero pairs (combinations)
            are exhausted, the pool is randomized again and the process is
            repeated. The map must contain at least 4 heroes for this to
            have an effect.
            Default: 0

        * random_obstacles (int)
            VCMI `--random-obstacles` option. Specifies how often (in number of
            battles) should battlefield obstacles be changed.
            If `0`, all battles will use the same obstacle layout.
            If `1`, each battle will use random obstacle layout.
            Default: 0

        * town_chance (int)
            VCMI `--town-chance` option. Specifies the chance (in percents) to
            have a town battle.
            If `0`, no town battles will be fought.
            If `100`, all battles will be fought in a town. The map must
            contain towns with forts for this to have an effect.
            Default: 0

        * warmachine_chance (int)
            VCMI `--warmachine-chance` option. Specifies the percentage chance
            to add one war machine to a hero army. This is checked separately
            for each war machine type (tent, cart or ballista) and each hero.
            For example: a value of `50` means there is a 50% chance to add a
            ballista, a 50% chance to add a tent and a 50% chance to add a cart
            to each hero in the battle.
            If `0`, no war machines will be included in any army.
            If `100`, all 3 war machines will be included in each army.
            Default: 0

        * random_terrain_chance (int)
            VCMI `--random-terrain-chance` option. Specifies the percentage
            chance to replace the battlefield's original terrain by a randomly
            selected land-based terrain (i.e. excludes boats).
            Default: 0

        * tight_formation_chance (int)
            VCMI `--tight-formation-chance` option. Specifies the percentage
            chance to set a tight formation for a hero's army.
            Default: 0

        * battlefield_pattern (str)
            VCMI `--battlefield-pattern` option, used to filter the
            battlefield on which battles will be fought.
            Default: ""

        * mana_min (int)
            VCMI `--mana-min` option. At the start of a battle, heroes will be
            given a random amount of mana no less than this value.
            Cannot be greater than `mana_max`.
            Default: 0

        * mana_max (int)
            VCMI `--mana-max` option. At the start of a battle, heroes will be
            given a random amount of mana no greater than this value.
            Default: 0

        * swap_sides (int)
            VCMI `--swap-sides` option. Specifies how often (in number of
            battles) the battle perspective is swapped, effectively swapping
            the values of `attacker` and `defender`.
            If `0`, no swaps will occur.
            If `1`, perspective will be swapped on each battle.
            Default: 0

        * allow_retreat (bool)
            Whether to always mask the "0" action in order to prevent agents
            from retreating.
            Default: False

        * step_reward_mult (int)
            Reward calculation parameter, denoted `a` in the formula below:
                R0 = a * (b + c + d*D + Vk) + σ*e*Ve
            where:
            - `R0` is the reward before any global modificators are applied
              (see `reward_clip_tanh_army_frac` and `reward_army_value_ref`)
            - `a`, `b`, `c`, `d` and `e` are configurable parameters
            - `D` is the net damage dealt vs damage received
            - `V` is the net value of units killed vs units lost
            - `Ve` is the difference in value of our army vs the enemy army
            - `σ` is a term which evaluates to 1 at battle end, 0 otherwise
            Default: 1

        * step_reward_fixed (int)
            Reward calculation parameter, denoted `b` in the formula in `step_reward_mult`.
            This is usually a negative value representing a punishment for
            agents who keep running away from the enemy troops to avoid damage.
            Default: 0

        * step_reward_frac (int)
            Reward calculation parameter, denoted `c` in the formula in `step_reward_mult`.
            This is similar to `step_reward_fixed`, but is given as a fraction
            of the mean of the two starting total army values.
            Example: -0.001, meaning 0.1% of the starting army values (-100 for a 100K army)
            Default: 0

        * reward_dmg_factor (int)
            Reward calculation parameter, denoted `d` in the formula in `step_reward_mult`.
            Default: 5

        * term_reward_mult (int)
            Reward calculation parameter, denoted `e` in the formula in `step_reward_mult`.
            Default: 1

        * reward_clip_tanh_army_frac (float)
            Reward modification parameter, denoted `t` in the formula below:
                T = t * Vm
                R1 = T * np.tanh(R0 / T)
            where:
            - `t` is a configurable parameter
            - `Vm` is the mean of the two starting total army values
            - `R1` is the reward after applying this modificator
            - `R0` is the calculated reward as per the formula in `reward_dmg_factor`
            Used if soft clipping of rewards is desired.
            Default: 1.0

        * reward_army_value_ref (int)
            Reward modification parameter, denoted `f` in the formula below:
                R2 = R1 * f * Vm
            where:
            - `R2` is the reward after applying this modificator
            - `R1` is the calculated reward as per the formula in `reward_clip_tanh_army_frac`
            - `f` is a configurable parameter
            - `Vm` is the mean of the two starting total army values
            Used to scale rewards based on the total army values by
            making all rewards relative to the starting army value.
            For example: consider these two VCMI battles:
              (A) armies with total starting army value = 1K (early game army)
              (B) armies with total starting army value = 100K (late game army)
            Without scaling, the rewards in battle A would be 100 times smaller
            than the rewards in battle B.
            Specifying an army ref of 10K, A and B's rewards will be multiplied
            by 10 and 0.1, effectively negating this discrepancy and ensuring
            the RL agent perceives early-game and late-game battles as equally
            significant.
            Default: 1000

        * reward_dynamic_scaling (bool)
            Whether to scale rewards based on current (instead of starting)
            total army values.
            Default: False

        * conntype (str)
            Experimental; do not use.
            Uses threadconnector instead of procconnector and VCMI is run in a
            separate thread in the same process. This has major limitations:
            env cannot be shutdown, nor there can be more than 1 envs ever
            created in a single process (except for the special case of
            dual-training where a single connector is shared). However, it
            significantly improves performance as well as allows for a dual-env
            setup in which another model can be loaded if VCMI is compiled
            without libtorch support.

        * other_env (VcmiEnv)
            Experimental; do not use.
            Another VcmiEnv to connect to.
            Used for dual-model training in a multi-threaded setup where
            the connector is "threadconnector". Here is how it works:
            1. The first VcmiEnv is created with the following init args:
                `opponent="OTHER_ENV", conntype="thread", other_env=None`.
            2. The second VcmiEnv is created with the following init args:
                `opponent="OTHER_ENV", conntype="thread", other_env=<the first env>`.
            The two environments must be created in different threads as they
            will block each other (see connector diagrams in the web docs).
            All init arguments for the second env will be ignored in this case,
            as there is only one instance of VCMI, started by the first env.
        """
        assert vcmi_loglevel_global in self.__class__.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in self.__class__.VCMI_LOGLEVELS
        assert role in self.__class__.ROLES
        assert opponent in self.__class__.OPPONENTS, f"{opponent} in {self.__class__.OPPONENTS}"
        assert conntype in ["thread", "proc"]

        self.action_space = self.__class__.ACTION_SPACE
        self.observation_space = self.__class__.OBSERVATION_SPACE

        # <params>
        self.render_mode = render_mode
        self.sparse_info = sparse_info
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
        self._reset_vars(self.connector.reset())

    @tracelog
    def step(self, action):
        if (self.terminated or self.truncated) and action != -1:
            raise Exception("Reset needed")

        # Prevent VCMI exceptions (mid-battle retreats are not handled)
        if action == 0 and not self.allow_retreat:
            raise Exception("Retreat is not allowed")

        res = self.connector.step(action)

        if action in [-1, 0]:
            self._reset_vars(res)

        if res.errcode > 0 and not self.allow_invalid_actions:
            self.logger.warn("Attempted an invalid action (errcode=%d)" % res.errcode)

        analysis = self.analyzer.analyze(action, res)
        term = res.is_battle_over
        rew, rew_unclipped = self.calc_reward(analysis, res)

        res.actmask[0] = False  # prevent retreats for now
        obs = {"observation": res.state, "action_mask": res.actmask}

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

        obs = {"observation": self.result.state, "action_mask": self.result.actmask}
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
                self.analyzer.actions_count,
                round(self.reward, 2),
                round(self.reward_total, 2),
                self.net_value_last,
                self.analyzer.net_value
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

    # To use attnmask, code in pyconnector.py and BAI/v1/state.cpp
    # must be uncommented and both VCMI and connector must be recompiled.
    def attn_mask(self):
        # return self.result.attnmask
        raise Exception("attn_mask disabled for performance reasons")

    def decode(self):
        return self.__class__.decode_obs(self.result.state)

    @staticmethod
    def decode_obs(obs):
        return Decoder.decode(obs["observation"])

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

        self._step_reward_calc = self.step_reward_fixed + self.step_reward_frac * avg_army_value

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

        if self.reward_dynamic_scaling:
            avg_army_value = (res.current_side0_army_value + res.current_side1_army_value) / 2
            self.reward_scaling_factor = self.reward_army_value_ref / avg_army_value
            self._step_reward_calc = self.step_reward_fixed + self.step_reward_frac * avg_army_value

        rew = analysis.net_value + self.reward_dmg_factor * analysis.net_dmg
        rew *= self.step_reward_mult

        if res.is_battle_over:
            vdiff = res.current_side0_army_value - res.current_side1_army_value
            vdiff = -vdiff if res.side == 1 else vdiff
            rew += (self.term_reward_mult * vdiff)

        rew += self._step_reward_calc

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
