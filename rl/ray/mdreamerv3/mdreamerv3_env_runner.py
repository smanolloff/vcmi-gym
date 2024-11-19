
# from typing import Collection, List, Optional, Tuple, Union
from typing import List
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree

from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.core import COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME

from ray.rllib.utils.numpy import convert_to_numpy

import ray.rllib.utils.spaces.space_utils as space_utils


class MDreamerV3_EnvRunner(DreamerV3EnvRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # XXX (simo): on_episode_end() cb
        self._callbacks = self.config.callbacks_class()

    def _sample_timesteps(
        self,
        num_timesteps: int,
        explore: bool = True,
        random_actions: bool = False,
        force_reset: bool = False,
    ) -> List[SingleAgentEpisode]:
        """Helper method to run n timesteps.

        See docstring of self.sample() for more details.
        """
        done_episodes_to_return = []

        # Get initial states for all `batch_size_B` rows in the forward batch.
        initial_states = tree.map_structure(
            lambda s: np.repeat(s, self.num_envs, axis=0),
            convert_to_numpy(self.module.get_initial_state()),
        )

        # Have to reset the env (on all vector sub-envs).
        if force_reset or self._needs_initial_reset:
            obs, _ = self.env.reset()
            # XXX (simo): +1-0
            obs = space_utils.unbatch(obs)
            self._needs_initial_reset = False

            self._episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]

            # Set initial obs and states in the episodes.
            for i in range(self.num_envs):
                self._episodes[i].add_env_reset(observation=obs[i])
                self._states[i] = None

        # Don't reset existing envs; continue in already started episodes.
        else:
            # Pick up stored observations and states from previous timesteps.
            # XXX (simo): +1-0?
            # obs = space_utils.batch([eps.observations[-1] for eps in self._episodes])
            obs = np.stack([eps.observations[-1] for eps in self._episodes])

        # Loop through env for n timesteps.
        ts = 0
        while ts < num_timesteps:
            # Act randomly.
            if random_actions:
                # XXX (simo): +1-1
                # actions = self.env.action_space.sample()
                actions = [np.random.choice(np.where(o["action_mask"])[0]) for o in obs]
            # Compute an action using our RLModule.
            else:
                is_first = np.zeros((self.num_envs,))
                for i, eps in enumerate(self._episodes):
                    if self._states[i] is None:
                        is_first[i] = 1.0
                        self._states[i] = {k: s[i] for k, s in initial_states.items()}

                to_module = {
                    Columns.STATE_IN: tree.map_structure(
                        lambda s: self.convert_to_tensor(s), space_utils.batch(self._states)
                    ),
                    # XXX (simo): +1-1
                    # Columns.OBS: self.convert_to_tensor(obs),
                    Columns.OBS: tree.map_structure(lambda s: self.convert_to_tensor(s), space_utils.batch(obs)),
                    "is_first": self.convert_to_tensor(is_first),
                }
                # Explore or not.
                if explore:
                    outs = self.module.forward_exploration(to_module)
                else:
                    outs = self.module.forward_inference(to_module)

                # Model outputs one-hot actions (if discrete). Convert to int actions
                # as well.
                actions = convert_to_numpy(outs[Columns.ACTIONS])
                if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                    actions = np.argmax(actions, axis=-1)
                self._states = space_utils.unbatch(convert_to_numpy(outs[Columns.STATE_OUT]))

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            ts += self.num_envs

            # XXX (simo): +1-0
            obs = space_utils.unbatch(obs)

            for i in range(self.num_envs):
                # The last entry in self.observations[i] is already the reset
                # obs of the new episode.
                if terminateds[i] or truncateds[i]:
                    # Finish the episode with the actual terminal observation stored in
                    # the info dict.
                    self._episodes[i].add_env_step(
                        observation=infos["final_observation"][i],
                        action=actions[i],
                        reward=rewards[i],
                        terminated=terminateds[i],
                        truncated=truncateds[i],
                        # XXX (simo): add info
                        infos=infos["final_info"][i]
                    )
                    self._states[i] = None

                    # XXX (simo): on_episode_end() cb
                    self._callbacks.on_episode_end(
                        episode=self._episodes[i],
                        env_runner=self,
                        metrics_logger=self.metrics,
                        env=self.env,
                        rl_module=self.module,
                        env_index=i,
                    )

                    done_episodes_to_return.append(self._episodes[i])
                    # Create a new episode object.
                    self._episodes[i] = SingleAgentEpisode(observations=[obs[i]])
                else:
                    self._episodes[i].add_env_step(
                        observation=obs[i],
                        action=actions[i],
                        reward=rewards[i],
                        # XXX (simo): no need for infos on non-terminal steps
                    )

        # Return done episodes ...
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        # ... and all ongoing episode chunks. Also, make sure, we return
        # a copy and start new chunks so that callers of this function
        # don't alter our ongoing and returned Episode objects.
        ongoing_episodes = self._episodes
        self._episodes = [eps.cut() for eps in self._episodes]
        for eps in ongoing_episodes:
            self._ongoing_episodes_for_metrics[eps.id_].append(eps)

        self._increase_sampled_metrics(ts)

        return done_episodes_to_return + ongoing_episodes

    def _sample_episodes(
        self,
        num_episodes: int,
        explore: bool = True,
        random_actions: bool = False,
    ) -> List[SingleAgentEpisode]:
        """Helper method to run n episodes.

        See docstring of `self.sample()` for more details.
        """
        done_episodes_to_return = []

        obs, _ = self.env.reset()
        # XXX (simo): +1-0
        obs = space_utils.unbatch(obs)
        episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]

        # Multiply states n times according to our vector env batch size (num_envs).
        states = tree.map_structure(
            lambda s: np.repeat(s, self.num_envs, axis=0),
            convert_to_numpy(self.module.get_initial_state()),
        )
        is_first = np.ones((self.num_envs,))

        for i in range(self.num_envs):
            episodes[i].add_env_reset(observation=obs[i])

        eps = 0
        while eps < num_episodes:
            if random_actions:
                # XXX (simo): +1-1
                # actions = self.env.action_space.sample()
                actions = [np.random.choice(np.where(o["action_mask"])[0]) for o in obs]
            else:
                batch = {
                    Columns.STATE_IN: tree.map_structure(
                        lambda s: self.convert_to_tensor(s), states
                    ),
                    # XXX (simo): +1-1
                    # Columns.OBS: self.convert_to_tensor(obs),
                    Columns.OBS: tree.map_structure(lambda s: self.convert_to_tensor(s), space_utils.batch(obs)),
                    "is_first": self.convert_to_tensor(is_first),
                }

                if explore:
                    outs = self.module.forward_exploration(batch)
                else:
                    outs = self.module.forward_inference(batch)

                actions = convert_to_numpy(outs[Columns.ACTIONS])
                if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                    actions = np.argmax(actions, axis=-1)
                states = convert_to_numpy(outs[Columns.STATE_OUT])

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

            # XXX (simo): +1-0
            obs = space_utils.unbatch(obs)

            for i in range(self.num_envs):
                # The last entry in self.observations[i] is already the reset
                # obs of the new episode.
                if terminateds[i] or truncateds[i]:
                    eps += 1

                    episodes[i].add_env_step(
                        observation=infos["final_observation"][i],
                        action=actions[i],
                        reward=rewards[i],
                        terminated=terminateds[i],
                        truncated=truncateds[i],
                        # XXX (simo): add info
                        infos=infos["final_info"][i]
                    )
                    done_episodes_to_return.append(episodes[i])

                    # Also early-out if we reach the number of episodes within this
                    # for-loop.
                    if eps == num_episodes:
                        break

                    # Reset h-states to the model's initial ones b/c we are starting a
                    # new episode.
                    for k, v in convert_to_numpy(
                        self.module.get_initial_state()
                    ).items():
                        states[k][i] = v
                    is_first[i] = True

                    # XXX (simo): on_episode_end() cb
                    self._callbacks.on_episode_end(
                        episode=episodes[i],
                        env_runner=self,
                        metrics_logger=self.metrics,
                        env=self.env,
                        rl_module=self.module,
                        env_index=i,
                    )

                    episodes[i] = SingleAgentEpisode(observations=[obs[i]])
                else:
                    episodes[i].add_env_step(
                        observation=obs[i],
                        action=actions[i],
                        reward=rewards[i],
                    )
                    is_first[i] = False

        self._done_episodes_for_metrics.extend(done_episodes_to_return)

        # If user calls sample(num_timesteps=..) after this, we must reset again
        # at the beginning.
        self._needs_initial_reset = True

        ts = sum(map(len, done_episodes_to_return))
        self._increase_sampled_metrics(ts)

        return done_episodes_to_return

    # XXX (simo): enable eval env runner (where env_steps are occasionally synced)
    def set_state(self, state):
        if self.module is None:
            assert self.config.share_module_between_env_runner_and_learner
            return

        if NUM_ENV_STEPS_SAMPLED_LIFETIME in state:
            self.metrics.set_value(
                key=NUM_ENV_STEPS_SAMPLED_LIFETIME,
                value=state[NUM_ENV_STEPS_SAMPLED_LIFETIME],
                reduce="sum",
            )
        else:
            self.module.set_state(state[COMPONENT_RL_MODULE][DEFAULT_MODULE_ID])

    def sample(self, *args, **kwargs):
        res = super().sample(*args, **kwargs)
        # XXX (simo): remove the (obsolete) incomplete_episode list from result:
        if len(res) == 2 and isinstance(res[0], list) and res[1] == []:
            return res[0]
        else:
            return res

    def ping(self) -> str:
        # Unwrap to bypass OrderEnforcing
        # Also, render() should be thread-safe (connector uses a lock)
        return str(bool(self.env.envs[0].unwrapped.render()))
