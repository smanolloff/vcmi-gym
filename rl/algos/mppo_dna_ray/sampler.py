import numpy as np
import torch
import time
from collections import deque


def safe_mean(array_like) -> float:
    return np.nan if len(array_like) == 0 else float(np.mean(array_like))


def compute_advantages(rewards, dones, values, next_done, next_value, gamma, gae_lambda):
    total_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(total_steps)):
        if t == total_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


class Sampler:
    def __init__(self, sampler_id, NN_creator, venv_creator, num_steps, gamma, gae_lambda_policy, gae_lambda_value, device_name):
        print("[sampler.%d] Initializing ..." % sampler_id)
        self.sampler_id = sampler_id
        self.device = torch.device(device_name)
        self.NN_value = NN_creator(self.device)
        self.NN_value.eval()
        self.NN_policy = NN_creator(self.device)
        self.NN_policy.eval()
        self.venv = venv_creator()
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda_policy = gae_lambda_policy
        self.gae_lambda_value = gae_lambda_value

        # 1 is num_envs
        # (everything must be batched, B=1 in this case)
        assert self.venv.num_envs == 1
        self.obs_space = self.venv.call("observation_space")[0]
        self.act_space = self.venv.call("action_space")[0]
        self.obs = torch.zeros((num_steps, 1) + self.obs_space.shape).to(self.device)
        self.actions = torch.zeros((num_steps, 1) + self.act_space.shape, dtype=torch.int64).to(self.device)
        self.logprobs = torch.zeros((num_steps, 1)).to(self.device)
        self.rewards = torch.zeros((num_steps, 1)).to(self.device)
        self.dones = torch.zeros((num_steps, 1)).to(self.device)
        self.values = torch.zeros((num_steps, 1)).to(self.device)
        self.masks = torch.zeros((num_steps, 1, self.act_space.n), dtype=torch.bool).to(self.device)

        next_obs, _ = self.venv.reset()
        self.next_obs = torch.as_tensor(next_obs, device=self.device)
        self.next_done = torch.zeros(1, device=self.device)
        self.next_mask = torch.as_tensor(np.array(self.venv.unwrapped.call("action_mask")), device=self.device)

    def set_weights(self, value_state_dict, policy_state_dict):
        self.NN_value.load_state_dict(value_state_dict, strict=True)
        self.NN_policy.load_state_dict(policy_state_dict, strict=True)
        self.NN_value.eval()
        self.NN_policy.eval()

    def sample(self):
        # print("[sampler.%d] Sampling ..." % self.sampler_id)
        started_at = time.time()
        seconds = 0
        episodes = 0
        ep_net_value_queue = deque(maxlen=1000)
        ep_is_success_queue = deque(maxlen=1000)

        for step in range(0, self.num_steps):
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done
            self.masks[step] = self.next_mask

            with torch.no_grad():
                action, logprob, _, _ = self.NN_policy.get_action(self.next_obs, self.next_mask)
                value = self.NN_value.get_value(self.next_obs)
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
                    ep_net_value_queue.append(final_info["net_value"])
                    ep_is_success_queue.append(final_info["is_success"])
                    episodes += 1

        seconds = int(started_at - time.time())

        # bootstrap value if not done
        with torch.no_grad():
            self.next_value = self.NN_value.get_value(self.next_obs).reshape(1, -1)
            advantages, _ = compute_advantages(
                self.rewards,
                self.dones,
                self.values,
                self.next_done,
                self.next_value,
                self.gamma,
                self.gae_lambda_policy
            )
            _, returns = compute_advantages(
                self.rewards,
                self.dones,
                self.values,
                self.next_done,
                self.next_value,
                self.gamma,
                self.gae_lambda_value
            )

        # flatten the batch
        res = (
            self.obs.flatten(end_dim=1),
            self.logprobs.flatten(end_dim=1),
            self.actions.flatten(end_dim=1),
            self.masks.flatten(end_dim=1),
            advantages.flatten(end_dim=1),
            returns.flatten(end_dim=1),
            self.values.flatten(end_dim=1),
        )

        stats = (
            seconds,
            episodes,
            safe_mean(ep_net_value_queue),
            safe_mean(ep_is_success_queue),
            safe_mean(self.venv.return_queue),
            safe_mean(self.venv.length_queue),
        )

        return res, stats

    def shutdown(self):
        print("[sampler.%d] Shutting down ..." % self.sampler_id)
        self.venv.close()
