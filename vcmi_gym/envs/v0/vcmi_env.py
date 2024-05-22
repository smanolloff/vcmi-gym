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
from collections import namedtuple
from types import SimpleNamespace

from .util import log
from .util.analyzer import Analyzer, ActionType
from .util.pyconnector import (
    PyConnector,
    STATE_VALUE_NA,
    STATE_SIZE_DEFAULT_ONE_HEX,
    STATE_SIZE_FLOAT_ONE_HEX,
    STATE_ENCODING_DEFAULT,
    STATE_ENCODING_FLOAT,
    ATTRMAP_DEFAULT,
    ATTRMAP_FLOAT,
    HEXACTMAP,
    HEXSTATEMAP,
    MELEEDISTMAP,
    SHOOTDISTMAP,
    DMGMODMAP,
    SIDEMAP,
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
    N_ACTIONS,
)


# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ZERO = DTYPE(0)
ONE = DTYPE(1)

TRACE = True
MAXLEN = 80


# NOTE: removing action=0 (retreat) which is used for resetting.
#       => start from 1 and reduce total actions by 1
#          XXX: there seems to be a bug as start=1 causes this error with SB3:
#          index 1322 is out of bounds for dimension 2 with size 1322
#       => just start from 0, reduce max by 1, and manually add +1
ACTION_OFFSET = 1


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


class Hex(namedtuple("Hex", ATTRMAP_DEFAULT.keys())):
    def __repr__(self):
        return f'Hex(y={self.HEX_Y_COORD} x={self.HEX_X_COORD})'

    def dump(self, compact=True):
        maxlen = 0
        lines = []
        for field in self._fields:
            value = getattr(self, field)
            maxlen = max(maxlen, len(field))

            if field.startswith("HEX_ACTION_MASK_FOR_"):
                names = list(HEXACTMAP.keys())
                indexes = np.where(value)[0]
                value = None if not indexes else ", ".join([names[i] for i in indexes])
            elif field.startswith("HEX_STATE"):
                value = None if not value else list(HEXSTATEMAP.keys())[value]
            elif field.startswith("HEX_MELEE_DISTANCE_FROM"):
                value = None if not value else list(MELEEDISTMAP.keys())[value]
            elif field.startswith("HEX_SHOOT_DISTANCE_FROM"):
                value = None if not value else list(SHOOTDISTMAP.keys())[value]
            elif field.startswith("HEX_MELEEABLE_BY"):
                value = None if not value else list(DMGMODMAP.keys())[value]

            if value is None and compact:
                continue
            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if value is None else value) for (field, value) in lines]))

    def action(self, hexaction):
        if isinstance(hexaction, str):
            hexaction = HEXACTMAP.get(hexaction, None)

        if hexaction not in HEXACTMAP.values():
            return None

        if not self.HEX_ACTION_MASK_FOR_ACT_STACK[hexaction]:
            print("Action not possible for this hex")
            return None

        n = 15*self.HEX_Y_COORD + self.HEX_X_COORD
        return N_NONHEX_ACTIONS - ACTION_OFFSET + n*N_HEX_ACTIONS + hexaction

    def actions(self):
        return [k for k, v in HexAction.__dict__.items() if self.HEX_ACTION_MASK_FOR_ACT_STACK[v]]


HexAction = SimpleNamespace(**HEXACTMAP)
HexState = SimpleNamespace(**HEXSTATEMAP)
MeleeDistance = SimpleNamespace(**MELEEDISTMAP)
ShootDistance = SimpleNamespace(**SHOOTDISTMAP)
DmgMod = SimpleNamespace(**DMGMODMAP)
Side = SimpleNamespace(**SIDEMAP)


class Battlefield(list):
    def __repr__(self):
        return "Battlefield(11x15)"

    def get(self, y_or_n, x=None):
        if x:
            y = y_or_n
        else:
            y = y_or_n // 15
            x = y_or_n % 15

        if y >= 0 and y < len(self) and x >= 0 and x < len(self[y]):
            return self[y][x]
        else:
            print("Invalid hex (y=%s x=%s)" % (y, x))


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
        vcmi_loglevel_stats="error",  # vcmi loglevel
        vcmienv_loglevel="WARN",  # python loglevel
        encoding_type="default",  # default / float
        consecutive_error_reward_factor=-1,  # unused
        attacker="MMAI_USER",  # MMAI_USER / MMAI_MODEL / StupidAI / BattleAI
        defender="StupidAI",  # MMAI_USER / MMAI_MODEL / StupidAI / BattleAI
        attacker_model=None,  # MPPO zip model (if attacker=MMAI_MODEL)
        defender_model=None,  # MPPO zip model (if defender=MMAI_MODEL)
        vcmi_stats_mode="disabled",
        vcmi_stats_storage="-",
        vcmi_stats_persist_freq="100",
        vcmi_stats_sampling=0,
        vcmi_stats_score_var=0.4,
        sparse_info=False,
        allow_invalid_actions=False,
        actions_log_file=None,  # DEBUG
        user_timeout=600,  # seconds - user input
        vcmi_timeout=5,  # seconds
        boot_timeout=60,  # seconds - needed as VCMI boot sometimes hangs with a memleak
        reward_dmg_factor=5,
        reward_clip_tanh_army_frac=1,  # max action reward relative to starting army value
        reward_army_value_ref=0,  # scale rewards relative to starting army value (0=no scaling)
        swap_sides=0,  # swap combat sides at each Nth combat (disabled if 0*)
        random_heroes=0,  # pick heroes at random each Nth combat (disabled if 0*)
        random_obstacles=0,  # place obstacles at random each Nth combat (disabled if 0*)
        step_reward_fixed=0,  # fixed reward
        step_reward_mult=1,
        term_reward_mult=1,  # at term step, reward = diff in total army values
    ):
        assert vcmi_loglevel_global in VcmiEnv.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in VcmiEnv.VCMI_LOGLEVELS
        assert attacker in VcmiEnv.ROLES
        assert defender in VcmiEnv.ROLES
        assert attacker == "MMAI_USER" or defender == "MMAI_USER", "an MMAI_USER role is required"

        if encoding_type == STATE_ENCODING_DEFAULT:
            hex_size = STATE_SIZE_DEFAULT_ONE_HEX
            self.attribute_mapping = ATTRMAP_DEFAULT
        elif encoding_type == STATE_ENCODING_FLOAT:
            hex_size = STATE_SIZE_FLOAT_ONE_HEX
            self.attribute_mapping = ATTRMAP_FLOAT
        else:
            raise Exception("Expected encoding_type: expected one of %s, got: %s" % (
                [STATE_ENCODING_DEFAULT, STATE_ENCODING_FLOAT], encoding_type)
            )

        self.logger = log.get_logger("VcmiEnv", vcmienv_loglevel)
        self.connector = PyConnector(vcmienv_loglevel, user_timeout, vcmi_timeout, boot_timeout)
        self.encoding_type = encoding_type

        result = self.connector.start(
            encoding_type,
            mapname,
            random_heroes,
            random_obstacles,
            swap_sides,
            vcmi_loglevel_global,
            vcmi_loglevel_ai,
            # vcmi_loglevel_stats,
            attacker,
            defender,
            attacker_model or "",
            defender_model or "",
            # vcmi_stats_mode,
            # vcmi_stats_storage,
            # vcmi_stats_persist_freq,
            # vcmi_stats_sampling,
            # vcmi_stats_score_var,
        )

        self.action_space = gym.spaces.Discrete(N_ACTIONS - ACTION_OFFSET)
        self.observation_space = gym.spaces.Box(
            low=STATE_VALUE_NA,
            high=1,
            shape=(11, 15, hex_size),
            dtype=np.float32
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

        obs = res.state
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
            self.render()

        if self.actions_log_file:
            if self.actfile:
                self.actfile.close()
            self.actfile = open(self.actions_log_file, "w")

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
        if self.actfile:
            self.actfile.close()

        self.connector.shutdown()
        self.logger.info("Env closed")
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()

    @tracelog
    def action_mask(self):
        return self.result.actmask[ACTION_OFFSET:]

    def attn_masks(self):
        return self.result.attnmasks

    def decode(self):
        return VcmiEnv.decode_obs(self.result.state)

    @staticmethod
    def decode_obs(obs):
        assert len(obs.shape) == 3
        assert obs.shape[0] == 11
        assert obs.shape[1] == 15
        assert obs.shape[2] in [STATE_SIZE_DEFAULT_ONE_HEX, STATE_SIZE_FLOAT_ONE_HEX]

        attrmap = ATTRMAP_FLOAT if obs.shape[2] == STATE_SIZE_FLOAT_ONE_HEX else ATTRMAP_DEFAULT

        res = Battlefield()
        for y in range(11):
            row = []
            for x in range(15):
                row.append(VcmiEnv.decode_hex(obs[y][x], attrmap))
            res.append(row)

        return res

    @staticmethod
    def decode_hex(hexdata, attrmap):
        res = {}

        for attr, (enctype, offset, n, vmax) in attrmap.items():
            attrdata = hexdata[offset:][:n]

            if attrdata[0] == STATE_VALUE_NA:
                res[attr] = None
                continue

            match enctype:
                case "NUMERIC":
                    res[attr] = attrdata.argmin()
                case "NUMERIC_SQRT":
                    value_sqrt = attrdata.argmin()
                    value_min = round(value_sqrt ** 2)
                    value_max = round((value_sqrt+1) ** 2)
                    assert value_max <= vmax, f"internal error: {value_max} > {vmax}"
                    res[attr] = value_min, value_max
                case "BINARY":
                    bits = attrdata
                    res[attr] = bits.astype(int)
                case "CATEGORICAL":
                    res[attr] = attrdata.argmax()

                case "FLOATING":
                    assert n == 1, f"internal error: {n} != 1"
                    res[attr] = round(attrdata[0] * vmax)
                case "CATEGORICAL":
                    res[attr] = attrdata.argmax()
                case _:
                    raise Exception(f"Unexpected encoding type: {enctype}")

        return Hex(**res)

    # just play a pre-recorded actions from vcmi actions.txt
    # useful when "resuming" battle after reproducing & fixing a bug
    def replay_actions_txt(self, actions_txt="vcmi_gym/envs/v0/vcmi/actions.txt"):
        with open(actions_txt, "r") as f:
            actions_str = f.read()

        for a in actions_str.split():
            print("Replaying %s" % a)
            obs, rew, term, trunc, info = self.step(int(a) - 1)
            if rew < 0:
                self.render()
                raise Exception("error replaying: %s" % a)

        self.render()

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
        self.analyzer = Analyzer(N_ACTIONS - ACTION_OFFSET)

    def _maybe_render(self, analysis):
        if self.render_each_step:
            self.render()

    def _transform_action(self, action):
        # see note for action_space
        max_action = self.action_space.n - ACTION_OFFSET
        assert action >= 0 and action <= max_action
        return action + ACTION_OFFSET

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
