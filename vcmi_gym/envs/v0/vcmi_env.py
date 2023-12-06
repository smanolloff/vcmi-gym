import numpy as np
import gymnasium as gym

from .util import log
from .util.analyzer import Analyzer, ActionType
from .util.pyconnector import PyConnector, ERRNAMES, STATE_SIZE, N_ACTIONS


# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ZERO = DTYPE(0)
ONE = DTYPE(1)


class InfoDict(dict):
    SCALAR_VALUES = [
        # "errors",  # redundant for MPPO (no errors)
        "net_value",
        "is_success",
        # "valid_actions",  # redundant for MPPO (=rollout/ep_len_mean)
    ]

    D1_ARRAY_VALUES = {
        # "error_counters": ERRNAMES,  # redundant for MPPO (no errors)
        "action_type_counters": [at.name for at in ActionType],
        # "action_type_valid_counters": [at.name for at in ActionType],  # redundant for MPPO (no errors)
    }

    D2_ARRAY_VALUES = [
        "action_coords_counters",
        # "action_coords_valid_counters",  # redundant for MPPO (no errors)
    ]

    ALL_KEYS = SCALAR_VALUES + list(D1_ARRAY_VALUES.keys()) + D2_ARRAY_VALUES

    def __setitem__(self, k, v):
        assert k in InfoDict.ALL_KEYS, f"Unknown info key: '{k}'"
        super().__setitem__(k, v)


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]

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
        consecutive_error_reward_factor=-1,
        sparse_info=False
    ):
        assert vcmi_loglevel_global in VcmiEnv.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in VcmiEnv.VCMI_LOGLEVELS

        self.logger = log.get_logger("VcmiEnv", vcmienv_loglevel)
        self.connector = PyConnector(vcmienv_loglevel)

        result = self.connector.start(
            mapname,
            vcmi_loglevel_global,
            vcmi_loglevel_ai
        )

        # NOTE: removing action=0 (retreat) which is used for resetting...
        #       => start from 1 and reduce total actions by 1
        #          XXX: there seems to be a bug as start=1 causes this error:
        #          index 1322 is out of bounds for dimension 2 with size 1322
        #       => just start from 0, reduce max by 1, and manually add +1
        self.action_offset = 1
        self.action_space = gym.spaces.Discrete(N_ACTIONS - self.action_offset)
        self.observation_space = gym.spaces.Box(shape=(STATE_SIZE,), low=0, high=1, dtype=DTYPE)

        # <params>
        self.render_mode = render_mode
        self.sparse_info = sparse_info
        self.max_steps = max_steps
        self.render_each_step = render_each_step
        self.consecutive_error_reward_factor = consecutive_error_reward_factor
        # </params>

        # required to init vars
        self._reset_vars(result)

    def step(self, action):
        action += self.action_offset  # see note for action_space

        if self.terminated or self.truncated:
            raise Exception("Reset needed")

        res = self.connector.act(action)
        analysis = self.analyzer.analyze(action, res)
        rew = VcmiEnv.calc_reward(analysis, self.consecutive_error_reward_factor)
        obs = res.state
        term = res.is_battle_over
        trunc = self.analyzer.actions_count >= self.max_steps
        info = VcmiEnv.build_info(res, term, trunc, analysis, self.sparse_info)

        self._update_vars_after_step(res, rew, term, trunc, analysis)
        self._maybe_render(analysis)

        return obs, rew, term, trunc, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_vars(res=None)
        self.result = self.connector.reset()

        if self.render_each_step:
            print(self.render())

        return self.result.state, {}

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

    def close(self):
        self.connector.shutdown()
        self.logger.info("Env closed")
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()

    def action_masks(self):
        return self.result.actmask[self.action_offset:]

    #
    # private
    #

    def _update_vars_after_step(self, res, rew, term, trunc, analysis):
        self.result = res
        self.reward = rew
        self.reward_total += rew
        self.net_value_last = analysis.net_value
        self.terminated = term
        self.truncated = trunc
        self.last_action_was_valid = (analysis.errors_count == 0)

    def _reset_vars(self, res=None):
        self.result = res
        self.reward = 0
        self.reward_total = 0
        self.net_value_last = 0
        self.terminated = False
        self.truncated = False
        self.last_action_was_valid = True

        # Vars updated after other events
        self.n_renders_skipped = 0
        self.analyzer = Analyzer(self.action_space.n)

    def _maybe_render(self, analysis):
        if self.render_each_step:
            if analysis.errors_count == 0:
                print(self.render())
                self.n_renders_skipped = 0
            else:
                self.n_renders_skipped += 1

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
    def build_info(res, term, trunc, analysis, sparse_info):
        # Performance optimization
        if not (term or trunc) and sparse_info:
            return {}

        info = InfoDict()

        # info["errors"] = analysis.errors_count_ep  # redundant
        info["net_value"] = analysis.net_value_ep
        info["is_success"] = res.is_victorious
        # info["valid_actions"] = analysis.actions_valid_count_ep  # redundant

        # info["error_counters"] = analysis.error_counters_ep  # redundant
        info["action_type_counters"] = analysis.action_type_counters_ep
        # info["action_type_valid_counters"] = analysis.action_type_valid_counters_ep  # redundant
        info["action_coords_counters"] = analysis.action_coords_counters_ep
        # info["action_coords_valid_counters"] = analysis.action_coords_valid_counters_ep  # redundant

        # Return regular dict (wrappers insert arbitary keys)
        return dict(info)

    @staticmethod
    def calc_reward(analysis, consecutive_error_reward_factor):
        # Penalize with ever-increasing amounts to prevent
        # it from just making errors forever
        if analysis.errors_count > 0:
            return analysis.errors_consecutive_count * consecutive_error_reward_factor

        # XXX: pikemen value=80, life=10
        return analysis.net_value + 5 * analysis.net_dmg
