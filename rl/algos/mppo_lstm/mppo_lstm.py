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
import sys
import random
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque
import functools
import operator

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
# import tyro

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
    encoding_type: str = ""  # DEPRECATED
    max_steps: int = 500
    reward_dmg_factor: int = 5
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
    sparse_info: bool = True
    step_reward_fixed: int = 0
    step_reward_mult: int = 1
    term_reward_mult: int = 0
    consecutive_error_reward_factor: Optional[int] = None
    user_timeout: int = 30
    vcmi_timeout: int = 30
    boot_timeout: int = 30
    random_heroes: int = 1
    random_obstacles: int = 1
    town_chance: int = 0
    warmachine_chance: int = 0
    mana_min: int = 0
    mana_max: int = 0
    swap_sides: int = 0
    reward_clip_tanh_army_frac: int = 1
    reward_army_value_ref: int = 0
    true_rng: bool = True

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class LstmArgs:
    input_shape: list[int] = field(default_factory=list)
    bidirectional: bool = False
    num_layers: int = 0
    hidden_size: int = 0
    proj_size: int = 0
    seq_len: int = 0


@dataclass
class NetworkArgs:
    features_extractor: list[dict] = field(default_factory=list)
    actor: dict = field(default_factory=dict)
    critic: dict = field(default_factory=dict)
    lstm: LstmArgs = field(default_factory=lambda: LstmArgs())

    def __post_init__(self):
        if not isinstance(self.lstm, LstmArgs):
            self.lstm = LstmArgs(**self.lstm)
        common.coerce_dataclass_ints(self)


@dataclass
class State:
    seed: int = -1
    resumes: int = 0
    map_swaps: int = 0  # DEPRECATED
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
    mapmask: str = ""  # DEPRECATED
    randomize_maps: bool = False  # DEPRECATED
    permasave_every: int = 7200  # seconds; no retention
    save_every: int = 3600  # seconds; retention (see max_saves)
    max_saves: int = 3
    out_dir_template: str = "data/{group_id}/{run_id}"

    opponent_load_file: Optional[str] = None
    opponent_sbm_probs: list = field(default_factory=lambda: [1, 0, 0])
    lr_schedule: ScheduleArgs = field(default_factory=lambda: ScheduleArgs())
    weight_decay: float = 0.0
    num_envs: int = 0  # DEPRECATED (envmaps determines number of envs now)
    envmaps: list = field(default_factory=lambda: ["gym/A1.vmap"])
    num_steps: int = 128
    gamma: float = 0.99
    stats_buffer_size: int = 100
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

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


class ChanFirst(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    def __init__(self, channels, activation="LeakyReLU"):
        super().__init__()
        self.block = nn.Sequential(
            common.layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)),
            getattr(nn, activation)(),
            common.layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
        )

    def forward(self, x):
        return x + self.block(x)


class AgentNN(nn.Module):
    @staticmethod
    def build_layer(spec):
        kwargs = dict(spec)  # copy
        t = kwargs.pop("t")
        layer_cls = getattr(torch.nn, t, None) or globals()[t]
        return layer_cls(**kwargs)

    def __init__(self, network, action_space, observation_space):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.lstm_seq_len = network.lstm.seq_len
        self.lstm_d = 1 + network.lstm.bidirectional
        self.lstm_num_layers = network.lstm.num_layers
        self.lstm_input_shape = network.lstm.input_shape
        self.lstm_input_size = functools.reduce(operator.mul, self.lstm_input_shape)
        self.lstm_proj_size = network.lstm.proj_size
        self.lstm_hidden_size = network.lstm.hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=self.lstm_d == 2,
            proj_size=self.lstm_proj_size
        )

        self.features_extractor = torch.nn.Sequential()
        for spec in network.features_extractor:
            layer = AgentNN.build_layer(spec)
            self.features_extractor.append(common.layer_init(layer))

        self.actor = common.layer_init(AgentNN.build_layer(network.actor), gain=0.01)
        self.critic = common.layer_init(AgentNN.build_layer(network.critic), gain=1.0)

    def get_value(self, xseq, hstate, cstate):
        hstate = hstate.transpose(0, 1)
        cstate = cstate.transpose(0, 1)
        memories, _ = self.lstm(xseq, (hstate, cstate))
        memory = memories[:, -1, :]
        features = self.features_extractor(memory)
        return self.critic(features)

    def get_action_and_value(
        self,
        xseq,
        hstate,  # batch-first
        cstate,  # batch-first
        mask,
        action=None,
        deterministic=False
    ):
        # torch's LSTM has A batch_first param, but it affects only input and
        # output tensors. The hidden and cell states are always batch_first=False
        #  => they are transposed manually here.
        #
        # i.e. shapes are:
        #  input = (B, S, *IN)
        #  output = (B, S, *OUT)
        #  hstate = (L, B, *OUT) => transpose(0,1) => (B, L, *OUT)
        #  cstate = (L, B, *OUT) => transpose(0,1) => (B, L, *OUT)
        #
        # Legend: B=batch_size, S=seq_len, L=num_layers

        hstate = hstate.transpose(0, 1)
        cstate = cstate.transpose(0, 1)
        memories, (hstate, cstate) = self.lstm(xseq, (hstate, cstate))
        hstate = hstate.transpose(0, 1)
        cstate = cstate.transpose(0, 1)

        # XXX: memories contains n_seq for each element in the sequence
        #      In contrast, hstate and cstate are the *final* states
        #      Not sure how to use intermediate memories => take the final one
        features = self.features_extractor(memories[:, -1, :])
        value = self.critic(features)
        action_logits = self.actor(features)
        dist = common.CategoricalMasked(logits=action_logits, mask=mask)
        if action is None:
            if deterministic:
                action = torch.argmax(dist.probs, dim=1)
            else:
                action = dist.sample()
        return hstate, cstate, action, dist.log_prob(action), dist.entropy(), value

    # Inference (deterministic)
    def predict(self, obs, mask):
        if obs.shape != self.observation_space.shape:
            # If observation is batched, only a single batch is supported
            assert obs.shape[1:] == self.observation_space.shape, "bad input shape: %s, expected: %s or %s" % (
                obs.shape, self.observation_space.shape, (1, *self.observation_space.shape))

            assert obs.shape[0] == 1, "batched input is supported only if B=1, got: %s" % obs.shape

            obs = obs[1:]
            assert obs.shape == self.observation_space.shape

        with torch.no_grad():
            obs = torch.as_tensor(obs, device='cpu')
            mask = torch.as_tensor(mask, device='cpu')

            obs_seq = getattr(self, "lstm_obs_seq", None)
            if obs_seq is None:
                print("Initializing new lstm_obs_seq...")
                self.lstm_obs_seq = torch.zeros((1, self.lstm_seq_len, *self.observation_space.shape))
                self.lstm_hstate = torch.zeros((1, self.lstm_d * self.lstm_num_layers, self.lstm_proj_size or self.lstm_hidden_size))
                self.lstm_cstate = torch.zeros((1, self.lstm_d * self.lstm_num_layers, self.lstm_hidden_size))

            # TODO: lstm states must be RESET at the end of an episode.
            #       Maybe expose an Agent#reset() method which does this
            #       (it is up to the user to invoke it when the env resets)
            #       VCMI's MMAI_MODEL getAction function can also check the
            #       SupplementaryData if this is a new game. This will require
            #       careful handling (e.g. AutoPlay re-inits the bot but the
            #       state will be inconsistent if turns were played manually
            #       in the meantime...)
            self.lstm_obs_seq = self.lstm_obs_seq.roll(-1, dims=1)
            self.lstm_obs_seq[0][-1] = obs
            self.lstm_hstate, self.lstm_cstate, b_env_action, _, _, _ = self.get_action_and_value(
                self.lstm_obs_seq,
                self.lstm_hstate,
                self.lstm_cstate,
                mask.unsqueeze(dim=0),
                deterministic=True
            )

            return b_env_action[0].cpu().item()


class Agent(nn.Module):
    """
    Store a "clean" version of the agent: create a fresh one and copy the attrs
    manually (for nn's and optimizers - copy their states).
    This prevents issues if the agent contains wandb hooks at save time.
    """
    @staticmethod
    def save(agent, agent_file):
        print("Saving agent to %s" % agent_file)
        attrs = ["args", "observation_space", "action_space", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN.load_state_dict(agent.NN.state_dict(), strict=True)
        clean_agent.optimizer.load_state_dict(agent.optimizer.state_dict())
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

        # v1, v2
        # jagent.features_extractor = clean_agent.NN.features_extractor

        # v3
        jagent.features_extractor1_stacks = clean_agent.NN.features_extractor1_stacks
        jagent.features_extractor1_hexes = clean_agent.NN.features_extractor1_hexes
        jagent.features_extractor2 = clean_agent.NN.features_extractor2

        # common
        jagent.actor = clean_agent.NN.actor
        jagent.critic = clean_agent.NN.critic
        torch.jit.save(torch.jit.script(jagent), jagent_file)

    @staticmethod
    def load(agent_file, device="cpu"):
        print("Loading agent from %s" % agent_file)
        return torch.load(agent_file, map_location=device)

    def __init__(self, args, observation_space, action_space, state=None, device="cpu"):
        super().__init__()
        self.args = args
        self.env_version = args.env_version
        self.observation_space = observation_space  # needed for save/load
        self.action_space = action_space  # needed for save/load
        self.NN = AgentNN(args.network, action_space, observation_space)
        self.NN.to(device)
        self.optimizer = torch.optim.AdamW(self.NN.parameters(), eps=1e-5)
        self.predict = self.NN.predict
        self.state = state or State()


class JitAgent(nn.Module):
    """ TorchScript version of Agent (inference only) """

    def __init__(self):
        super().__init__()
        # XXX: these are overwritten after object is initialized
        self.features_extractor = nn.Identity()
        self.actor = nn.Identity()
        self.critic = nn.Identity()
        self.env_version = 0

    # Inference (deterministic)
    @torch.jit.export
    def predict(self, obs, mask) -> int:
        with torch.no_grad():
            b_obs = torch.as_tensor(obs).cpu().unsqueeze(dim=0)
            b_mask = torch.as_tensor(mask).cpu().unsqueeze(dim=0)

            # v1, v2
            # features = self.features_extractor(b_obs)

            # v3
            stacks, hexes = b_obs.split([1960, 10725], dim=1)
            fstacks = self.features_extractor1_stacks(stacks)
            fhexes = self.features_extractor1_hexes(hexes)
            fcat = torch.cat((fstacks, fhexes), dim=1)
            features = self.features_extractor2(fcat)

            action_logits = self.actor(features)
            dist = common.SerializableCategoricalMasked(logits=action_logits, mask=b_mask)
            action = torch.argmax(dist.probs, dim=1)
            return action.int().item()

    @torch.jit.export
    def get_value(self, obs) -> float:
        with torch.no_grad():
            b_obs = torch.as_tensor(obs).cpu().unsqueeze(dim=0)
            features = self.features_extractor(b_obs)
            value = self.critic(features)
            return value.float().item()

    @torch.jit.export
    def get_version(self) -> int:
        return self.env_version


def group_obs_into_episodes(obs, dones, lstm_seq_len):
    num_steps, num_envs = obs.shape
    single_obs_shape = obs[0][0].shape
    episodes = []

    for env in range(num_envs):
        episode = torch.zeros([lstm_seq_len, *single_obs_shape], dtype=obs.dtype)
        i_ep = 0
        for step in range(num_steps):
            episode[i_ep] = obs[step, env]
            i_ep += 1
            if i_ep == lstm_seq_len or dones[step, env]:
                i_ep = 0
                episodes.append(episode)
                episode = torch.zeros([lstm_seq_len, *single_obs_shape], dtype=obs.dtype)

        # Add remaining steps as a final episode
        if i_ep > 0:
            episodes.append(episode)

    return torch.stack(episodes)


def main(args, agent_cls=Agent):
    LOG = logging.getLogger("mppo_conv")
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

    if args.seed >= 0:
        seed = args.seed
    elif agent and agent.state.seed >= 0:
        seed = agent.state.seed
    else:
        seed = np.random.randint(2**31)

    wrappers = args.env_wrappers

    if args.env_version == 1:
        from vcmi_gym import VcmiEnv_v1 as VcmiEnv
    elif args.env_version == 2:
        from vcmi_gym import VcmiEnv_v2 as VcmiEnv
    elif args.env_version == 3:
        from vcmi_gym import VcmiEnv_v3 as VcmiEnv
    else:
        raise Exception("Unsupported env version: %d" % args.env_version)

    obs_space = VcmiEnv.OBSERVATION_SPACE
    act_space = VcmiEnv.ACTION_SPACE

    if agent is None:
        agent = Agent(args, obs_space, act_space, device=device)

    # Legacy models with offset actions
    if agent.NN.actor.out_features == 2311:
        print("Using legacy model with 2311 actions")
        wrappers.append(dict(module="vcmi_gym", cls="LegacyActionSpaceWrapper"))
        n_actions = 2311
    else:
        print("Using new model with 2312 actions")
        n_actions = 2312

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    try:
        envs = common.create_venv(VcmiEnv, args, seed=np.random.randint(2**31))
        [ENVS.append(e) for e in envs.unwrapped.envs]  # DEBUG

        assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent.state.seed = seed

        # these are used by gym's RecordEpisodeStatistics wrapper
        envs.return_queue = agent.state.ep_rew_queue
        envs.length_queue = agent.state.ep_length_queue

        # XXX: the start=0 requirement is needed for SB3 compat
        assert act_space.shape == ()

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
        # obs = torch.zeros((args.num_steps, num_envs) + obs_space.shape, device=device)
        lstm = args.network.lstm
        lstm_d = 1 + lstm.bidirectional

        # XXX: storing hstates and cstates as batch-first
        #      see comments in get_action_and_value()
        lstm_obs_seqs = torch.zeros((args.num_steps, num_envs, lstm.seq_len, *obs_space.shape), device=device)
        lstm_hstates = torch.zeros((args.num_steps, num_envs, lstm_d * lstm.num_layers, lstm.proj_size or lstm.hidden_size), device=device)
        lstm_cstates = torch.zeros((args.num_steps, num_envs, lstm_d * lstm.num_layers, lstm.hidden_size), device=device)
        actions = torch.zeros((args.num_steps, num_envs, *act_space.shape), device=device)
        logprobs = torch.zeros((args.num_steps, num_envs), device=device)
        rewards = torch.zeros((args.num_steps, num_envs), device=device)
        dones = torch.zeros((args.num_steps, num_envs), dtype=torch.bool, device=device)
        values = torch.zeros((args.num_steps, num_envs), device=device)
        masks = torch.zeros((args.num_steps, num_envs, n_actions), dtype=torch.bool, device=device)

        # TRY NOT TO MODIFY: start the game
        next_obs, _ = envs.reset(seed=agent.state.seed)  # XXX: seed has no effect here
        next_obs = torch.as_tensor(next_obs, device=device)
        next_lstm_obs_seq = torch.zeros_like(lstm_obs_seqs[0])
        next_lstm_obs_seq[:, -1] = next_obs
        next_lstm_hstate = torch.zeros_like(lstm_hstates[0])
        next_lstm_cstate = torch.zeros_like(lstm_cstates[0])
        next_done = torch.zeros(num_envs, device=device)
        next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask")), device=device)

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
                lstm_obs_seqs[step] = next_lstm_obs_seq
                lstm_hstates[step] = next_lstm_hstate
                lstm_cstates[step] = next_lstm_cstate
                dones[step] = next_done
                masks[step] = next_mask

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    next_lstm_hstate, next_lstm_cstate, action, logprob, _, value = agent.NN.get_action_and_value(
                        next_lstm_obs_seq,
                        next_lstm_hstate,
                        next_lstm_cstate,
                        next_mask
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_obs = torch.as_tensor(next_obs, device=device)
                next_done = torch.as_tensor(np.logical_or(terminations, truncations), device=device)
                # next_done=true means next_obs is part of a new episode (venv has auto-reset)
                # (verified that this does not mutate lstm_obs_seqs[step])
                next_lstm_obs_seq[next_done] = torch.zeros_like(next_lstm_obs_seq[0])
                next_lstm_obs_seq = next_lstm_obs_seq.roll(-1, dims=1)

                # next_lstm_obs_seq is (E,N,*), next_obs is (E,*)
                #
                # next_lstm_obs_seq[:, -1] = next_obs expands to:
                # next_lstm_obs_seq[0][-1] = next_obs[0]
                # next_lstm_obs_seq[1][-1] = next_obs[1]
                # ...
                next_lstm_obs_seq[:, -1] = next_obs
                # Reset LSTM states on episode end
                next_lstm_hstate[next_done] = torch.zeros_like(next_lstm_hstate[0])
                next_lstm_cstate[next_done] = torch.zeros_like(next_lstm_cstate[0])

                rewards[step] = torch.as_tensor(reward, device=device).view(-1)
                next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask")), device=device)

                # XXX SIMO: SB3 does bootstrapping for truncated episodes here
                # https://github.com/DLR-RM/stable-baselines3/pull/658

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
                next_value = agent.NN.get_value(next_lstm_obs_seq, next_lstm_hstate, next_lstm_cstate).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].float()
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # If S=num_steps, E=num_envs
            # => flatten the batch (S,E,*) => (*)
            b_lstm_obs_seqs = lstm_obs_seqs.flatten(end_dim=1)
            b_lstm_hstates = lstm_hstates.flatten(end_dim=1)
            b_lstm_cstates = lstm_cstates.flatten(end_dim=1)
            b_logprobs = logprobs.flatten(end_dim=1)
            b_actions = actions.flatten(end_dim=1)
            b_masks = masks.flatten(end_dim=1)
            b_advantages = advantages.flatten(end_dim=1)
            b_returns = returns.flatten(end_dim=1)
            b_values = values.flatten(end_dim=1)

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []

            # XXX: train during optimization
            agent.train()
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, _, _, newlogprob, entropy, newvalue = agent.NN.get_action_and_value(
                        b_lstm_obs_seqs[mb_inds],
                        b_lstm_hstates[mb_inds],
                        b_lstm_cstates[mb_inds],
                        b_masks[mb_inds],
                        action=b_actions.long()[mb_inds]
                    )
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
    return agent, common.safe_mean(list(agent.state.rollout_rew_queue_1000)[-agent.state.current_rollout:])


def debug_args():
    return Args(
        "mppo_lstm-test",
        "mppo_lstm-test",
        loglevel=logging.DEBUG,
        run_name=None,
        trial_id=None,
        wandb_project=None,
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="data/mppo_lstm-test/mppo_lstm-test/agent-1718753136.pt",
        # agent_load_file="data/mppo_lstm-test/mppo_lstm-test/agent-1718752596.pt",
        agent_load_file=None,
        vsteps_total=0,
        seconds_total=0,
        rollouts_per_mapchange=0,
        rollouts_per_log=1,
        rollouts_per_table_log=100000,
        success_rate_target=None,
        ep_rew_mean_target=None,
        quit_on_target=False,
        mapside="attacker",
        # save_every=2000000000,  # greater than time.time()
        save_every=2,  # greater than time.time()
        permasave_every=2000000000,  # greater than time.time()
        max_saves=1,
        out_dir_template="data/mppo_lstm-test/mppo_lstm-test",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.001),
        envmaps=["gym/A1.vmap", "gym/A2.vmap"],
        # num_steps=64,
        num_steps=256,
        gamma=0.8,
        gae_lambda=0.8,
        num_minibatches=2,
        # num_minibatches=16,
        # update_epochs=2,
        update_epochs=10,
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
            consecutive_error_reward_factor=-1,
            sparse_info=True,
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
        ),
        env_wrappers=[],
        env_version=3,
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        network=dict(
            lstm=dict(
                input_shape=[12685],
                bidirectional=False,
                num_layers=1,
                hidden_size=512,
                proj_size=0,
                seq_len=5
            ),
            features_extractor=[
                dict(t="Linear", in_features=512, out_features=512),
                dict(t="LeakyReLU"),
            ],
            actor=dict(t="Linear", in_features=512, out_features=2312),
            critic=dict(t="Linear", in_features=512, out_features=1)
        )
    )


if __name__ == "__main__":
    # To run from vcmi-gym root:
    # $ python -m rl.algos.mppo
    main(debug_args())
