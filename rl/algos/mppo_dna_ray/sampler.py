import ray
import numpy as np
import torch
from collections import deque


@ray.remote
class Sampler:
    def __init__(self, agent_id, agent_creator, venv_creator, num_steps, device):
        self.agent = agent_creator(agent_id)
        self.venv = venv_creator(agent_id)
        self.next_obs, _ = self.env.reset()
        self.num_steps = num_steps
        self.device = device

        # 1 is num_envs
        # (everything must be batched, B=1 in this case)
        self.obs = torch.zeros((num_steps, 1) + self.env.obs_space["observation"].shape).to(device)
        self.actions = torch.zeros((num_steps, 1) + self.env.act_space.shape).to(device)
        self.logprobs = torch.zeros((num_steps, 1)).to(device)
        self.rewards = torch.zeros((num_steps, 1)).to(device)
        self.dones = torch.zeros((num_steps, 1)).to(device)
        self.values = torch.zeros((num_steps, 1)).to(device)
        self.masks = torch.zeros((num_steps, 1, self.env.act_space.n), dtype=torch.bool).to(device)

    def set_weights(self, state_dict):
        self.agent.load_state_dict(state_dict, strict=True)

    def sample(self):
        timesteps = 0
        seconds = 0
        episodes = 0
        ep_net_value_queue = deque(maxlen=100)
        ep_is_success_queue = deque(maxlen=100)

        for step in range(0, self.num_steps):
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done
            self.masks[step] = self.next_mask

            with torch.no_grad():
                action, logprob, _, _ = self.agent.NN_policy.get_action(self.next_obs, self.next_mask)
                value = self.agent.NN_value.get_value(self.next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            self.next_obs, reward, terminations, truncations, infos = self.venv.step(action.cpu().numpy())
            self.next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward, device=self.device).view(-1)
            self.next_obs = torch.as_tensor(self.next_obs, device=self.device)
            self.next_done = torch.as_tensor(self.next_done, device=self.device, dtype=torch.float32)
            self.next_mask = torch.as_tensor(np.array(self.venv.unwrapped.call("action_mask")), device=self.device)

            for final_info, has_final_info in zip(infos.get("final_info", []), infos.get("_final_info", [])):
                # XXXXX:
                # the `state` must not be stored in self.agent, but here in the runner
                # (and are only relevant for the current sample cycle)

                # "final_info" must be None if "has_final_info" is False
                if has_final_info:
                    assert final_info is not None, "has_final_info=True, but final_info=None"
                    self.agent.state.ep_net_value_queue.append(final_info["net_value"])
                    self.agent.state.ep_is_success_queue.append(final_info["is_success"])
                    self.agent.state.current_episode += 1
                    self.agent.state.global_episode += 1

            self.agent.state.current_vstep += 1
            self.agent.state.current_timestep += self.num_envs
            self.agent.state.global_timestep += self.num_envs
            self.agent.state.global_second = global_start_second + agent.state.current_second

        # print("SAMPLE TIME: %.2f" % (time.time() - tstart))
        # tstart = time.time()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.NN_value.get_value(
                next_obs,
                attn_mask=next_attnmask if attn else None
            ).reshape(1, -1)

            advantages, _ = compute_advantages(
                rewards, dones, values, next_done, next_value, args.gamma, args.gae_lambda_policy
            )
            _, returns = compute_advantages(rewards, dones, values, next_done, next_value, args.gamma, args.gae_lambda_value)

        # flatten the batch
        b_obs = obs.flatten(end_dim=1)
        b_logprobs = logprobs.flatten(end_dim=1)
        b_actions = actions.flatten(end_dim=1)
        b_masks = masks.flatten(end_dim=1)
        b_advantages = advantages.flatten(end_dim=1)
        b_returns = returns.flatten(end_dim=1)
        b_values = values.flatten(end_dim=1)


        return traj
