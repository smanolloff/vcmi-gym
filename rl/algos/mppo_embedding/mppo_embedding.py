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
# This file contains a modified version of CleanRL's PPO implementation:
# https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py
import os
import sys
import random
import logging
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import warnings

from ...encoder.autoencoder import Autoencoder
from .. import common


ENVS = []  # debug


def render():
    print(ENVS[0].render())


@dataclass
class ScheduleArgs:
    # const / lin_decay / exp_decay
    mode: str = "const"
    start: float = 2.5e-4
    end: float = 0
    rate: float = 10


@dataclass
class EnvArgs:
    max_steps: int = 500
    reward_dmg_factor: int = 5
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
    step_reward_fixed: int = 0
    step_reward_frac: float = 0
    step_reward_mult: int = 1
    term_reward_mult: int = 0
    user_timeout: int = 30
    vcmi_timeout: int = 30
    boot_timeout: int = 30
    conntype: str = "proc"
    random_heroes: int = 1
    random_obstacles: int = 1
    town_chance: int = 0
    warmachine_chance: int = 0
    random_terrain_chance: int = 0
    tight_formation_chance: int = 0
    battlefield_pattern: str = ""
    mana_min: int = 0
    mana_max: int = 0
    swap_sides: int = 0
    reward_clip_tanh_army_frac: int = 1
    reward_army_value_ref: int = 0
    reward_dynamic_scaling: bool = False
    true_rng: bool = True  # DEPRECATED
    deprecated_args: list[dict] = field(default_factory=lambda: [
        "true_rng"
    ])

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class NetworkArgs:
    autoencoder_config_file: str
    action_embedding_dim: int
    body: list[dict] = field(default_factory=list)


@dataclass
class State:
    seed: int = -1
    resumes: int = 0
    global_timestep: int = 0
    current_timestep: int = 0
    current_vstep: int = 0
    current_rollout: int = 0
    global_second: int = 0
    current_second: int = 0
    global_episode: int = 0
    current_episode: int = 0

    ep_length_queue: deque = field(default_factory=lambda: deque(maxlen=100))

    ep_rew_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_rew_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_rew_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    ep_net_value_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_net_value_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_net_value_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    ep_is_success_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_is_success_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_is_success_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class Args:
    run_id: str
    group_id: str
    run_name: Optional[str] = None
    trial_id: Optional[str] = None
    wandb_project: Optional[str] = None
    resume: bool = False
    overwrite: list = field(default_factory=list)
    notes: Optional[str] = None
    tags: Optional[list] = field(default_factory=list)
    loglevel: int = logging.DEBUG

    agent_load_file: Optional[str] = None
    vsteps_total: int = 0
    seconds_total: int = 0
    rollouts_per_mapchange: int = 0
    rollouts_per_log: int = 1
    rollouts_per_table_log: int = 10
    success_rate_target: Optional[float] = None
    ep_rew_mean_target: Optional[float] = None
    quit_on_target: bool = False
    mapside: str = "attacker"
    permasave_every: int = 7200  # seconds; no retention
    save_every: int = 3600  # seconds; retention (see max_saves)
    max_saves: int = 3
    out_dir_template: str = "data/{group_id}/{run_id}"

    opponent_load_file: Optional[str] = None
    opponent_sbm_probs: list = field(default_factory=lambda: [1, 0, 0])
    lr_schedule: ScheduleArgs = field(default_factory=lambda: ScheduleArgs())
    weight_decay: float = 0.0
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    target_kl: float = None
    num_minibatches: int = 4
    num_steps: int = 128
    stats_buffer_size: int = 100
    update_epochs: int = 4
    num_envs: int = 1
    envmaps: list = field(default_factory=lambda: ["gym/generated/4096/4x1024.vmap"])

    logparams: dict = field(default_factory=dict)
    cfg_file: Optional[str] = None
    seed: int = 42
    skip_wandb_init: bool = False
    skip_wandb_log_code: bool = False

    env: EnvArgs = field(default_factory=lambda: EnvArgs())
    env_version: int = 0
    env_wrappers: list = field(default_factory=list)
    network: NetworkArgs = field(default_factory=lambda: NetworkArgs())

    def __post_init__(self):
        if not isinstance(self.env, EnvArgs):
            self.env = EnvArgs(**self.env)
        if not isinstance(self.lr_schedule, ScheduleArgs):
            self.lr_schedule = ScheduleArgs(**self.lr_schedule)
        if not isinstance(self.network, NetworkArgs):
            self.network = NetworkArgs(**self.network)

        common.coerce_dataclass_ints(self)


class AgentNN(nn.Module):
    @staticmethod
    def build_layer(spec):
        kwargs = dict(spec)  # copy
        t = kwargs.pop("t")
        layer_cls = getattr(torch.nn, t, None) or globals()[t]
        return layer_cls(**kwargs)

    def __init__(self, network, input_dim, num_actions):
        super().__init__()

        with open(network.autoencoder_config_file, "r") as f:
            autoencoder_config = json.load(f)
            autoencoder = Autoencoder(input_dim=input_dim, layer_sizes=autoencoder_config["train"]["layer_sizes"])

        self.encoder = autoencoder.encoder

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.action_embedding = nn.Embedding(num_actions, network.action_embedding_dim)

        # Must use separate networks, as the actor net has different
        # input dimensions
        self.actor_net = torch.nn.Sequential()
        self.value_net = torch.nn.Sequential()

        for spec in network.body:
            self.actor_net.append(AgentNN.build_layer(spec))
            self.value_net.append(AgentNN.build_layer(spec))

        self.actor_head = nn.LazyLinear(1)
        self.value_head = nn.LazyLinear(1)

        # Init lazy layers
        with torch.no_grad():
            self.get_action_and_value(torch.randn([1, input_dim]), torch.ones([1, num_actions], dtype=torch.bool))

    # Notation used:
    #   B = batch
    #   N = num_actions
    #   AE = embedded action dim
    #   O = obs dim
    #   OE = encoded obs dim

    def get_action_logits(self, b_encobs, b_mask):
        b = b_encobs.shape[0]
        n = b_mask.shape[1]

        action_embs = self.action_embedding.weight
        # => (N, AE)

        b_action_embs = action_embs.unsqueeze(0).expand(b, *action_embs.shape)
        # => (B, N, AE)

        b_encobs_expanded = b_encobs.unsqueeze(1).expand(-1, n, -1)
        # => (B, N, OE)

        b_combined = torch.cat([b_encobs_expanded, b_action_embs], dim=2)
        # => (B, N, OE+AE)

        # Compute logits for all actions
        b, num_actions = b_mask.shape
        b_features = self.actor_net(b_combined.view(b * num_actions, -1))
        # => (B*N, OE+AE)

        b_logits = self.actor_head(b_features).view(b, num_actions)
        # => (B, N)

        # Apply action mask
        # XXX: use 1e9 instead of float.inf for stability
        b_logits = b_logits + (b_mask.float() - 1) * 1e9

        # (B, N)
        return b_logits

    def get_value(self, b_obs):
        return self.value_head(self.value_net(self.encoder(b_obs)))

    def get_action_and_value(self, b_obs, b_mask, b_action=None, deterministic=False):
        b_encobs = self.encoder(b_obs)
        b_action_logits = self.get_action_logits(b_encobs, b_mask)
        dist = torch.distributions.Categorical(logits=b_action_logits)
        if b_action is None:
            if deterministic:
                b_action = torch.argmax(dist.probs, dim=1)
            else:
                b_action = dist.sample()

        b_value = self.value_head(self.value_net(b_encobs))

        return b_action, dist.log_prob(b_action), dist.entropy(), b_value

    # Inference (deterministic)
    def predict(self, b_obs, b_mask):
        with torch.no_grad():
            b_obs = torch.as_tensor(b_obs)
            b_mask = torch.as_tensor(b_mask)

            # Return unbatched action if input was unbatched
            if len(b_mask.shape) == 1:
                b_obs = b_obs.unsqueeze(dim=0)
                b_mask = b_mask.unsqueeze(dim=0)
                b_env_action, _, _, _ = self.get_action_and_value(b_obs, b_mask, deterministic=True)
                return b_env_action[0].cpu().item()
            else:
                b_env_action, _, _, _ = self.get_action_and_value(b_obs, b_mask, deterministic=True)
                return b_env_action.cpu().numpy()


class Agent(nn.Module):
    """
    Store a "clean" version of the agent: create a fresh one and copy the attrs
    manually (for nn's and optimizers - copy their states).
    This prevents issues if the agent contains wandb hooks at save time.
    """
    @staticmethod
    def save(agent, agent_file):
        print("Saving agent to %s" % agent_file)
        if not os.path.isabs(agent_file):
            warnings.warn(
                f"path {agent_file} is not absolute!"
                " If VCMI is started in a thread, the current directory is changed."
                f" CWD: {os.getcwd()}"
            )

        attrs = ["args", "observation_space", "action_space", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN.load_state_dict(agent.NN.state_dict(), strict=True)
        clean_agent.optimizer.load_state_dict(agent.optimizer.state_dict())

        # no need to save the encoder
        clean_agent.NN.encoder = None

        torch.save(clean_agent, agent_file)

    @staticmethod
    def jsave(agent, jagent_file):
        print("Saving JIT agent to %s" % jagent_file)
        attrs = ["args", "observation_space", "action_space", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN.load_state_dict(agent.NN.state_dict(), strict=True)
        clean_agent.optimizer.load_state_dict(agent.optimizer.state_dict())
        jagent = JitAgent()
        jagent.env_version = clean_agent.env_version
        jagent.encoder = clean_agent.NN.encoder
        jagent.action_embedding = clean_agent.NN.action_embedding
        jagent.actor_net = clean_agent.NN.actor_net
        jagent.value_net = clean_agent.NN.value_net
        jagent.actor_head = clean_agent.NN.actor_head
        jagent.value_head = clean_agent.NN.value_head

        jagent_optimized = optimize_for_mobile(torch.jit.script(jagent), preserved_methods=["get_version", "predict", "get_value"])
        jagent_optimized._save_for_lite_interpreter(jagent_file)

    @staticmethod
    def load(agent_file, device="cpu"):
        print("Loading agent from %s (device: %s)" % (agent_file, device))
        agent = torch.load(agent_file, map_location=device, weights_only=False)

        # The encoder is static and is not saved
        # => build a new agent and use its encoder instead
        assert agent.NN.encoder is None
        attrs = ["args", "observation_space", "action_space", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        agent.NN.encoder = clean_agent.NN.encoder
        agent.NN.encoder.to(device)
        return agent

    def __init__(self, args, observation_space, action_space, state=None, device="cpu"):
        super().__init__()
        self.args = args
        self.env_version = args.env_version
        self.observation_space = observation_space  # needed for save/load
        self.action_space = action_space  # needed for save/load
        self.NN = AgentNN(args.network, observation_space["observation"].shape[0], action_space.n)
        self.NN.to(device)

        # Exclude encoder params (which don't require grad) from optimizer
        trainable_params = (p for p in self.NN.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(trainable_params, eps=1e-5)

        self.predict = self.NN.predict
        self.state = state or State()


class JitAgent(nn.Module):
    """ TorchScript version of Agent (inference only) """

    def __init__(self):
        super().__init__()
        # XXX: these are overwritten after object is initialized
        self.env_version = 0
        self.encoder = nn.Identity()
        self.action_embedding = nn.Identity()
        self.actor_net = nn.Identity()
        self.value_net = nn.Identity()
        self.actor_head = nn.Identity()
        self.value_head = nn.Identity()

    @torch.jit.export
    def forward(self, obs) -> torch.Tensor:
        b_obs = obs.unsqueeze(dim=0)

        # Code copied from AgentNN's get_action_and_value
        # (comments stripped)
        all_action_embs = self.action_embedding.weight
        num_actions = all_action_embs.shape[0]
        b_encobs = self.encoder(b_obs)
        b_encobs_expanded = b_encobs.unsqueeze(1).expand(-1, all_action_embs.shape[0], -1)
        b_combined = torch.cat([b_encobs_expanded, all_action_embs.unsqueeze(0)], dim=2)
        b_features = self.actor_net(b_combined.view(num_actions, -1))
        b_logits = self.actor_head(b_features).view(1, num_actions)

        return torch.cat(dim=1, tensors=(b_logits, self.critic_head(self.critic_net(b_encobs))))

    # Inference (deterministic)
    @torch.jit.export
    def predict(self, obs, mask, deterministic: bool = False) -> int:
        with torch.no_grad():
            b_obs = obs.unsqueeze(dim=0)
            b_mask = mask.unsqueeze(dim=0)

            # Code copied from AgentNN's get_action_and_value
            # (comments stripped)
            all_action_embs = self.action_embedding.weight
            b_encobs = self.encoder(b_obs)
            b_encobs_expanded = b_encobs.unsqueeze(1).expand(-1, all_action_embs.shape[0], -1)
            b_combined = torch.cat([b_encobs_expanded, all_action_embs.unsqueeze(0)], dim=2)
            b, num_actions = b_mask.shape
            b_features = self.actor_net(b_combined.view(b * num_actions, -1))
            b_logits = self.actor_head(b_features).view(b, num_actions)
            probs = self.categorical_masked(logits0=b_logits, mask=b_mask)

            if deterministic:
                action = torch.argmax(probs, dim=1)
            else:
                action = self.sample(probs, b_logits)

            return action.int().item()

    @torch.jit.export
    def get_value(self, obs) -> float:
        b_obs = obs.unsqueeze(dim=0)
        b_value = self.critic_head(self.critic_net(self.encoder(b_obs)))
        return b_value.float().item()

    @torch.jit.export
    def get_version(self) -> int:
        return self.env_version

    @torch.jit.export
    def categorical_masked(self, logits0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_value = torch.tensor(-((2 - 2**-23) * 2**127), dtype=logits0.dtype)

        # logits
        logits1 = torch.where(mask, logits0, mask_value)
        logits = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    @torch.jit.export
    def sample(self, probs: torch.Tensor, action_logits: torch.Tensor) -> torch.Tensor:
        num_events = action_logits.size()[-1]
        probs_2d = probs.reshape(-1, num_events)
        samples_2d = torch.multinomial(probs_2d, 1, True).T
        batch_shape = action_logits.size()[:-1]
        return samples_2d.reshape(batch_shape)


def main(args, agent_cls=Agent):
    LOG = logging.getLogger("mppo")
    LOG.setLevel(args.loglevel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert isinstance(args, Args)

    agent, args = common.maybe_resume(Agent, args, device=device)

    if args.seconds_total:
        assert not args.vsteps_total, "cannot have both vsteps_total and seconds_total"
        rollouts_total = 0
    else:
        rollouts_total = args.vsteps_total // args.num_steps

    # Re-initialize to prevent errors from newly introduced args when loading/resuming
    # TODO: handle removed args
    args = Args(**vars(args))

    if os.getenv("NO_WANDB") == "true":
        args.wandb_project = None

    printargs = asdict(args).copy()

    # Logger
    if not any(LOG.handlers):
        formatter = logging.Formatter(f"-- %(asctime)s %(levelname)s [{args.group_id}/{args.run_id}] %(message)s")
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = None
        loghandler = logging.StreamHandler()
        loghandler.setFormatter(formatter)
        LOG.addHandler(loghandler)

    LOG.info("Args: %s" % printargs)

    out_dir = args.out_dir_template.format(seed=args.seed, group_id=args.group_id, run_id=args.run_id)
    LOG.info("Out dir: %s" % out_dir)

    lr_schedule_fn = common.schedule_fn(args.lr_schedule)

    num_envs = len(args.envmaps)
    batch_size = int(num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)

    save_ts = None
    permasave_ts = None

    if args.agent_load_file and not agent:
        f = args.agent_load_file
        agent = Agent.load(f, device=device)
        agent.args = args
        agent.state.current_timestep = 0
        agent.state.current_vstep = 0
        agent.state.current_rollout = 0
        agent.state.current_second = 0
        agent.state.current_episode = 0

        # backup = "%s/loaded-%s" % (os.path.dirname(f), os.path.basename(f))
        # with open(f, 'rb') as fsrc:
        #     with open(backup, 'wb') as fdst:
        #         shutil.copyfileobj(fsrc, fdst)
        #         LOG.info("Wrote backup %s" % backup)

    common.validate_tags(args.tags)

    seed = args.seed

    # XXX: seed logic is buggy, do not use
    #      (this seed was never used to re-play trainings anyway)
    #      Just generate a random non-0 seed every time

    # if args.seed:
    #     seed = args.seed
    # elif agent and agent.state.seed:
    #     seed = agent.state.seed
    # else:

    # XXX: make sure the new seed is never 0
    # while seed == 0:
    #     seed = np.random.randint(2**31 - 1)

    wrappers = args.env_wrappers

    if args.env_version == 6:
        from vcmi_gym import VcmiEnv_v6 as VcmiEnv
    else:
        raise Exception("Unsupported env version: %d" % args.env_version)

    obs_space = VcmiEnv.OBSERVATION_SPACE
    act_space = VcmiEnv.ACTION_SPACE

    assert act_space.n == obs_space["action_mask"].shape[0]

    if agent is None:
        agent = Agent(args, obs_space, act_space, device=device)

    # TRY NOT TO MODIFY: seeding
    LOG.info("RNG master seed: %s" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    try:
        seeds = [np.random.randint(2**31) for i in range(args.num_envs)]
        envs = common.create_venv(VcmiEnv, args, seeds)
        [ENVS.append(e) for e in envs.unwrapped.envs]  # DEBUG

        agent.state.seed = seed

        # these are used by gym's RecordEpisodeStatistics wrapper
        envs.return_queue = agent.state.ep_rew_queue
        envs.length_queue = agent.state.ep_length_queue

        if args.wandb_project:
            import wandb
            common.setup_wandb(args, agent, __file__)

            # For wandb.log, commit=True by default
            # for wandb_log, commit=False by default
            def wandb_log(*args, **kwargs):
                wandb.log(*args, **dict({"commit": False}, **kwargs))
        else:
            def wandb_log(*args, **kwargs):
                pass

        common.log_params(args, wandb_log)

        if args.resume:
            agent.state.resumes += 1
            wandb_log({"global/resumes": agent.state.resumes})

        # print("Agent state: %s" % asdict(agent.state))

        assert act_space.shape == ()

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, num_envs) + obs_space["observation"].shape).to(device)
        actions = torch.zeros((args.num_steps, num_envs)).to(device)
        logprobs = torch.zeros((args.num_steps, num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, num_envs)).to(device)
        dones = torch.zeros((args.num_steps, num_envs)).to(device)
        values = torch.zeros((args.num_steps, num_envs)).to(device)

        masks = torch.zeros((args.num_steps, num_envs, act_space.n), dtype=torch.bool).to(device)

        # TRY NOT TO MODIFY: start the game
        next_obs_dict, _ = envs.reset(seed=agent.state.seed)  # XXX: seed has no effect here
        next_obs = torch.Tensor(next_obs_dict["observation"]).to(device)
        next_done = torch.zeros(num_envs).to(device)
        next_mask = torch.as_tensor(next_obs_dict["action_mask"]).to(device)

        progress = 0
        map_rollouts = 0
        start_time = time.time()
        global_start_second = agent.state.global_second

        while progress < 1:
            if args.vsteps_total:
                progress = agent.state.current_vstep / args.vsteps_total
            elif args.seconds_total:
                progress = agent.state.current_second / args.seconds_total
            else:
                progress = 0

            agent.optimizer.param_groups[0]["lr"] = lr_schedule_fn(progress)

            # XXX: eval during experience collection
            agent.eval()
            for step in range(0, args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                masks[step] = next_mask

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.NN.get_action_and_value(next_obs, next_mask)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs_dict, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                next_obs = torch.as_tensor(next_obs_dict["observation"], device=device)
                next_done = torch.as_tensor(next_done, device=device, dtype=torch.float32)
                next_mask = torch.as_tensor(next_obs_dict["action_mask"], device=device)

                # See notes/gym_vector.txt
                for final_info, has_final_info in zip(infos.get("final_info", []), infos.get("_final_info", [])):
                    # "final_info" must be None if "has_final_info" is False
                    if has_final_info:
                        assert final_info is not None, "has_final_info=True, but final_info=None"
                        agent.state.ep_net_value_queue.append(final_info["net_value"])
                        agent.state.ep_is_success_queue.append(final_info["is_success"])
                        agent.state.current_episode += 1
                        agent.state.global_episode += 1

                agent.state.current_vstep += 1
                agent.state.current_timestep += num_envs
                agent.state.global_timestep += num_envs
                agent.state.current_second = int(time.time() - start_time)
                agent.state.global_second = global_start_second + agent.state.current_second

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.NN.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.flatten(end_dim=1)
            b_logprobs = logprobs.flatten(end_dim=1)
            b_actions = actions.flatten(end_dim=1)
            b_masks = masks.flatten(end_dim=1)
            b_advantages = advantages.flatten(end_dim=1)
            b_returns = returns.flatten(end_dim=1)
            b_values = values.flatten(end_dim=1)

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []

            agent.train()
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.NN.get_action_and_value(b_obs[mb_inds], b_masks[mb_inds], b_action=b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        # XXX: SIMO: SB3 does not multiply by 0.5 here
                        #            (ie. SB3's vf_coef is essentially x2)
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.NN.parameters(), args.max_grad_norm)
                    agent.optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            ep_rew_mean = common.safe_mean(agent.state.ep_rew_queue)
            ep_value_mean = common.safe_mean(agent.state.ep_net_value_queue)
            ep_is_success_mean = common.safe_mean(agent.state.ep_is_success_queue)

            if envs.episode_count > 0:
                assert ep_rew_mean is not np.nan
                assert ep_value_mean is not np.nan
                assert ep_is_success_mean is not np.nan
                agent.state.rollout_rew_queue_100.append(ep_rew_mean)
                agent.state.rollout_rew_queue_1000.append(ep_rew_mean)
                agent.state.rollout_net_value_queue_100.append(ep_value_mean)
                agent.state.rollout_net_value_queue_1000.append(ep_value_mean)
                agent.state.rollout_is_success_queue_100.append(ep_is_success_mean)
                agent.state.rollout_is_success_queue_1000.append(ep_is_success_mean)

            wandb_log({"params/learning_rate": agent.optimizer.param_groups[0]["lr"]})
            wandb_log({"losses/total_loss": loss.item()})
            wandb_log({"losses/value_loss": v_loss.item()})
            wandb_log({"losses/policy_loss": pg_loss.item()})
            wandb_log({"losses/entropy": entropy_loss.item()})
            wandb_log({"losses/old_approx_kl": old_approx_kl.item()})
            wandb_log({"losses/approx_kl": approx_kl.item()})
            wandb_log({"losses/clipfrac": np.mean(clipfracs)})
            wandb_log({"losses/explained_variance": explained_var})
            wandb_log({"rollout/ep_count": envs.episode_count})
            wandb_log({"rollout/ep_len_mean": common.safe_mean(envs.length_queue)})

            if envs.episode_count > 0:
                assert ep_rew_mean is not np.nan
                assert ep_value_mean is not np.nan
                assert ep_is_success_mean is not np.nan
                agent.state.rollout_rew_queue_100.append(ep_rew_mean)
                agent.state.rollout_rew_queue_1000.append(ep_rew_mean)
                agent.state.rollout_net_value_queue_100.append(ep_value_mean)
                agent.state.rollout_net_value_queue_1000.append(ep_value_mean)
                agent.state.rollout_is_success_queue_100.append(ep_is_success_mean)
                agent.state.rollout_is_success_queue_1000.append(ep_is_success_mean)

                wandb_log({"rollout/ep_rew_mean": ep_rew_mean})
                wandb_log({"rollout/ep_value_mean": ep_value_mean})
                wandb_log({"rollout/ep_success_rate": ep_is_success_mean})

            wandb_log({"rollout100/ep_value_mean": common.safe_mean(agent.state.rollout_net_value_queue_100)})
            wandb_log({"rollout1000/ep_value_mean": common.safe_mean(agent.state.rollout_net_value_queue_1000)})
            wandb_log({"rollout100/ep_rew_mean": common.safe_mean(agent.state.rollout_rew_queue_100)})
            wandb_log({"rollout1000/ep_rew_mean": common.safe_mean(agent.state.rollout_rew_queue_1000)})
            wandb_log({"rollout100/ep_success_rate": common.safe_mean(agent.state.rollout_is_success_queue_100)})
            wandb_log({"rollout1000/ep_success_rate": common.safe_mean(agent.state.rollout_is_success_queue_1000)})
            wandb_log({"global/num_rollouts": agent.state.current_rollout})
            wandb_log({"global/num_timesteps": agent.state.current_timestep})
            wandb_log({"global/num_seconds": agent.state.current_second})
            wandb_log({"global/num_episode": agent.state.current_episode})

            envs.episode_count = 0

            if rollouts_total:
                wandb_log({"global/progress": progress})

            # XXX: maybe use a less volatile metric here (eg. 100 or 1000-average)
            if args.success_rate_target and ep_is_success_mean >= args.success_rate_target:
                LOG.info("Early stopping due to: success rate > %.2f (%.2f)" % (args.success_rate_target, ep_is_success_mean))

                if args.quit_on_target:
                    # XXX: break?
                    sys.exit(0)
                else:
                    raise Exception("Not implemented: map change on target")

            # XXX: maybe use a less volatile metric here (eg. 100 or 1000-average)
            if args.ep_rew_mean_target and ep_rew_mean >= args.ep_rew_mean_target:
                LOG.info("Early stopping due to: ep_rew_mean > %.2f (%.2f)" % (args.ep_rew_mean_target, ep_rew_mean))

                if args.quit_on_target:
                    # XXX: break?
                    sys.exit(0)
                else:
                    raise Exception("Not implemented: map change on target")

            if agent.state.current_rollout > 0 and agent.state.current_rollout % args.rollouts_per_log == 0:
                wandb_log({
                    "global/global_num_timesteps": agent.state.global_timestep,
                    "global/global_num_seconds": agent.state.global_second,
                    "global/global_num_episodes": agent.state.global_episode,
                }, commit=True)  # commit on final log line

                LOG.debug("rollout=%d vstep=%d rew=%.2f net_value=%.2f is_success=%.2f loss=%.2f" % (
                    agent.state.current_rollout,
                    agent.state.current_vstep,
                    ep_rew_mean,
                    ep_value_mean,
                    ep_is_success_mean,
                    loss.item()
                ))

            agent.state.current_rollout += 1
            save_ts, permasave_ts = common.maybe_save(save_ts, permasave_ts, args, agent, out_dir)

    finally:
        common.maybe_save(0, 10e9, args, agent, out_dir)
        if "envs" in locals():
            envs.close()

    # Needed by PBT to save model after iteration ends
    # XXX: limit returned mean reward to only the rollouts in this iteration
    # XXX: but no more than the last 300 rollouts (esp. if training vs BattleAI)
    ret_rew = common.safe_mean(list(agent.state.rollout_rew_queue_1000)[-min(300, agent.state.current_rollout):])
    ret_value = common.safe_mean(list(agent.state.rollout_net_value_queue_1000)[-min(300, agent.state.current_rollout):])

    wandb_log({
        "trial/ep_rew_mean": ret_rew,
        "trial/ep_value_mean": ret_value,
        "trial/num_rollouts": agent.state.current_rollout,
    }, commit=True)  # commit on final log line

    return (agent, ret_rew, ret_value)


def debug_args():
    return Args(
        "mppo-embedding-test",
        "mppo-embedding-test",
        loglevel=logging.DEBUG,
        run_name=None,
        trial_id=None,
        wandb_project=None,
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="data/mppo-test/mppo-test/agent-1735766664.pt",
        agent_load_file=None,
        vsteps_total=0,
        seconds_total=0,
        rollouts_per_mapchange=0,
        rollouts_per_log=1,
        rollouts_per_table_log=100000,
        success_rate_target=None,
        ep_rew_mean_target=None,
        quit_on_target=False,
        mapside="defender",
        save_every=10,  # greater than time.time()
        permasave_every=2000000000,  # greater than time.time()
        max_saves=1,
        out_dir_template="data/mppo-test/mppo-test",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.0001),
        # envmaps=["gym/generated/4096/4x1024.vmap"],
        envmaps=["gym/A1.vmap"],
        num_steps=128,
        # num_steps=4,
        gamma=0.8,
        gae_lambda=0.8,
        num_minibatches=4,
        # num_minibatches=16,
        # update_epochs=2,
        update_epochs=2,
        norm_adv=True,
        clip_coef=0.3,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=1.2,
        max_grad_norm=0.5,
        target_kl=None,
        logparams={},
        cfg_file=None,
        seed=42,
        skip_wandb_init=False,
        skip_wandb_log_code=False,
        env=EnvArgs(
            max_steps=500,
            reward_dmg_factor=5,
            vcmi_loglevel_global="error",
            vcmi_loglevel_ai="error",
            vcmienv_loglevel="WARN",
            step_reward_fixed=-100,
            step_reward_mult=1,
            term_reward_mult=0,
            random_heroes=0,
            random_obstacles=0,
            town_chance=0,
            warmachine_chance=0,
            mana_min=0,
            mana_max=0,
            swap_sides=0,
            reward_clip_tanh_army_frac=1,
            reward_army_value_ref=0,
            user_timeout=0,
            vcmi_timeout=0,
            boot_timeout=0,
            conntype="proc"
        ),
        env_wrappers=[],
        env_version=6,
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        network=dict(
            autoencoder_config_file="/Users/simo/Projects/vcmi-gym/data/autoencoder/cfzpdxub-config.json",
            action_embedding_dim=256,
            body=[
                dict(t="LazyLinear", out_features=256),
                dict(t="GELU"),
                dict(t="LazyLinear", out_features=1024),
                dict(t="GELU"),
                dict(t="LazyLinear", out_features=256),
                dict(t="GELU"),
            ]
        )
    )


if __name__ == "__main__":
    # To run from vcmi-gym root:
    # $ python -m rl.algos.mppo
    main(debug_args())
