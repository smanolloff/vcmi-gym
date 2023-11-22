import numpy as np
import gymnasium as gym

from .util import log
from .util.analyzer import Analyzer
from .util.pyconnector import PyConnector, ERRNAMES, STATE_SIZE, N_ACTIONS

# the numpy data type (pytorch works best with float32)
DTYPE = np.float32
ZERO = DTYPE(0)
ONE = DTYPE(1)


class VcmiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    VCMI_LOGLEVELS = ["trace", "debug", "info", "warn", "error"]

    INFO_ACTION_TYPE_KEY = ["action_type"]
    INFO_ACTION_HEX_KEY = ["action_hex"]
    INFO_KEYS = (
        ERRNAMES +
        INFO_ACTION_TYPE_KEY +
        INFO_ACTION_HEX_KEY +
        ["errors", "net_value", "is_success"]
    )

    #
    # A note on the static methods
    # Initially simply instance methods, I wanted to rip off the
    # reference to "self" as it introduced bugs related to the env state
    # (eg. some instance vars being used in calculations were not yet updated)
    # This approach is justified for training-critical methods only
    # (ie. no need to abstract out `render`, for example)

    @staticmethod
    def build_info(res, analysis):
        info = {name: count for (name, count) in zip(ERRNAMES, analysis.error_counters_ep)}
        info["errors"] = analysis.errors_count_ep
        info["net_value"] = analysis.net_value_ep
        info["is_success"] = res.is_victorious
        return info

    @staticmethod
    def calc_reward(analysis, consecutive_error_reward_factor):
        # Penalize with ever-increasing amounts to prevent
        # it from just making errors forever
        if analysis.errors_count > 0:
            return analysis.errors_consecutive_count * consecutive_error_reward_factor

        # XXX: pikemen value=80, life=10
        return analysis.net_value + 5 * analysis.net_dmg

    def __init__(
        self,
        mapname,
        seed=None,  # not used currently
        render_mode="ansi",
        render_each_step=False,
        vcmi_loglevel_global="error",  # vcmi loglevel
        vcmi_loglevel_ai="error",  # vcmi loglevel
        vcmienv_loglevel="WARN",  # python loglevel
        consecutive_error_reward_factor=-1
    ):
        assert vcmi_loglevel_global in VcmiEnv.VCMI_LOGLEVELS
        assert vcmi_loglevel_ai in VcmiEnv.VCMI_LOGLEVELS

        self.logger = log.get_logger("VcmiEnv", vcmienv_loglevel)
        self.connector = PyConnector(
            vcmienv_loglevel,
            mapname,
            vcmi_loglevel_global,
            vcmi_loglevel_ai
        )

        (result, self.errflags) = self.connector.start()

        # NOTE: removing action=0 (retreat) which is used for resetting...
        #       => start from 1 and reduce total actions by 1
        #          XXX: there seems to be a bug as start=1 causes this error:
        #          index 1322 is out of bounds for dimension 2 with size 1322
        #       => just start from 0, reduce max by 1, and manually add +1
        self.action_offset = 1
        self.action_space = gym.spaces.Discrete(N_ACTIONS - self.action_offset)
        self.observation_space = gym.spaces.Box(shape=(STATE_SIZE,), low=-1, high=1, dtype=DTYPE)

        # <params>
        self.render_mode = render_mode
        self.render_each_step = render_each_step
        self.consecutive_error_reward_factor = consecutive_error_reward_factor
        # </params>

        # required to init vars
        self._reset_vars(result)

    def step(self, action):
        action += self.action_offset  # see note for action_space

        if self.terminated:
            raise Exception("Reset needed")

        res = self.connector.act(action)
        analysis = self.analyzer.analyze(action, res)
        rew = VcmiEnv.calc_reward(analysis, self.consecutive_error_reward_factor)
        obs = res.state
        term = res.is_battle_over
        info = VcmiEnv.build_info(res, analysis)

        self._update_vars_after_step(analysis, res, rew, term)
        self._maybe_render(analysis)

        return obs, rew, term, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_vars(res=None)
        self.result = self.connector.reset()
        return self.result.state, {}

    def render(self):
        if self.render_mode == "ansi":
            footer = (
                f"Reward: {self.reward}\n"
                f"Total reward: {self.reward_total}\n"
                f"Total value: {self.analyzer.net_value}\n"
                f"n_steps: {self.analyzer.actions_count}"
            )

            return "%s\n%s" % (self.connector.render_ansi(), footer)

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

    #
    # private
    #

    def _update_vars_after_step(self, analysis, res, rew, term):
        self.result = res
        self.reward = rew
        self.reward_total += rew
        self.terminated = term
        self.last_action_was_valid = (analysis.errors_count == 0)

    def _reset_vars(self, res=None):
        self.result = res
        self.reward = 0
        self.reward_total = 0
        self.terminated = False
        self.last_action_was_valid = True

        # Vars updated after other events
        self.n_renders_skipped = 0
        self.analyzer = Analyzer(self.action_space.n, self.errflags)

    def _maybe_render(self, analysis):
        if self.render_each_step:
            if analysis.errors_count == 0:
                print("%s\nSkipped renders: %s" % (self.render(), self.n_renders_skipped))
                self.n_renders_skipped = 0
            else:
                self.n_renders_skipped += 1
