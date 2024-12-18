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

import torch
import glob
import gymnasium as gym
import os
import time
import re
import importlib
import pathlib
import numpy as np
import yaml
from functools import partial

import dataclasses
from torch.distributions.categorical import Categorical
from torch.distributions.utils import lazy_property


# https://boring-guy.sh/posts/masking-rl/
# combined with
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.2.1/sb3_contrib/common/maskable/distributions.py#L18
class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        assert mask is not None
        self.mask = mask
        self.mask_value = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype)
        logits = torch.where(self.mask, logits, self.mask_value)
        super().__init__(logits=logits)

    def entropy(self):
        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask, p_log_p, torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device))
        return -p_log_p.sum(-1)


class SerializableCategoricalMasked:
    """ TorchScript version of CategoricalMasked """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        assert mask is not None
        self.mask = mask

        self.mask_value = torch.tensor(-((2 - 2**-23) * 2**127), dtype=logits.dtype)
        logits = torch.where(self.mask, logits, self.mask_value)

        self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self._param = self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = (self._param.size()[:-1] if self._param.dim() > 1 else torch.Size([]))

        self._batch_shape = batch_shape
        self._event_shape = torch.Size([])

    def entropy(self):
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask, p_log_p, torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device))
        return -p_log_p.sum(-1)

    def sample(self):
        with torch.no_grad():
            probs_2d = self.probs.reshape(-1, self._num_events)
            samples_2d = torch.multinomial(probs_2d, 1, True).T
            _extended_shape = torch.Size(self._batch_shape + self._event_shape)
            return samples_2d.reshape(_extended_shape)

    def log_prob(self, value):
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    @lazy_property
    def logits(self):
        probs = self.probs
        eps = 2**-23
        ps_clamped = probs.clamp(min=eps, max=1 - eps)
        return torch.log(ps_clamped)

    @lazy_property
    def probs(self):
        return torch.nn.functional.softmax(self.logits, dim=-1)


def create_venv(env_cls, args, seeds):
    assert args.mapside in ["attacker", "defender"]

    # assert args.env.conntype == "proc" or args.num_envs == 1, (
    #     f"conntype='thread' not possible with args.num_envs={args.num_envs}"
    # )

    assert args.num_envs == len(seeds)

    assert len(args.opponent_sbm_probs) == 3
    if args.opponent_sbm_probs[2]:
        assert os.path.isfile(args.opponent_load_file)

    def env_creator(i):
        sbm = ["MMAI_SCRIPT_SUMMONER", "BattleAI", "MMAI_MODEL"]
        sbm_probs = torch.tensor(args.opponent_sbm_probs, dtype=torch.float)

        dist = Categorical(sbm_probs)
        opponent = sbm[dist.sample()]
        mapname = args.envmaps[i % len(args.envmaps)]

        if opponent == "MMAI_MODEL":
            assert args.opponent_load_file, "opponent_load_file is required for MMAI_MODEL"

        env_kwargs = dict(
            dataclasses.asdict(args.env),
            seed=np.random.randint(2**31),
            mapname=mapname,
        )

        if args.env_version >= 4:
            env_kwargs = dict(
                env_kwargs,
                role=args.mapside,
                opponent=opponent,
                opponent_model=args.opponent_load_file,
            )
        elif args.env_version == 3:
            env_kwargs = dict(
                env_kwargs,
                attacker=opponent,
                defender=opponent,
                attacker_model=args.opponent_load_file,
                defender_model=args.opponent_load_file,
            )
            env_kwargs[args.mapside] = "MMAI_USER"
            env_kwargs.pop("conntype", None)
            env_kwargs.pop("reward_dynamic_scaling", None)
        else:
            raise Exception("Unexpected env version: %s" % args.env_version)

        for a in env_kwargs.pop("deprecated_args", ["encoding_type"]):
            env_kwargs.pop(a, None)

        print("Env kwargs (env.%d): %s" % (i, env_kwargs))

        res = env_cls(**env_kwargs)

        for wrapper in args.env_wrappers:
            wrapper_mod = importlib.import_module(wrapper["module"])
            wrapper_cls = getattr(wrapper_mod, wrapper["cls"])
            res = wrapper_cls(res, **wrapper.get("kwargs", {}))

        return res

    funcs = [partial(env_creator, i) for i in range(args.num_envs)]

    if args.num_envs > 1:
        vec_env = gym.vector.AsyncVectorEnv(funcs)
    else:
        vec_env = gym.vector.SyncVectorEnv(funcs)

    vec_env = gym.wrappers.RecordEpisodeStatistics(vec_env, deque_size=args.stats_buffer_size)

    return vec_env


def maybe_save(t_save, t_permasave, args, agent, out_dir):
    now = time.time()

    # Used in cases of some sweeps with hundreds of short-lived agents
    if os.environ.get("NO_SAVE", None) == "true":
        return now, now

    if t_save is None or t_permasave is None:
        return now, now

    if t_save + args.save_every > now:
        return t_save, t_permasave

    os.makedirs(out_dir, exist_ok=True)
    agent_file = os.path.join(out_dir, "agent-%d.pt" % now)
    agent.__class__.save(agent, agent_file)
    t_save = now

    if args.wandb_project:
        import wandb
        assert wandb.run, "expected initialized wandb run"
        wandb.run.log_model(agent_file, name="agent.pt")

    if t_permasave + args.permasave_every <= now:
        permasave_file = os.path.join(out_dir, "agent-permasave-%d.pt" % now)
        agent.__class__.save(agent, permasave_file)
        t_permasave = now

    # save file retention (keep latest N saves)
    for pattern in ["agent-[0-9]*.pt", "nn-[0-9]*.pt"]:
        files = sorted(
            glob.glob(os.path.join(out_dir, pattern)),
            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()),
            reverse=True
        )

        for file in files[args.max_saves:]:
            print("Deleting %s" % file)
            os.remove(file)

    return t_save, t_permasave


def find_latest_save(group_id, run_id):
    datadir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    pattern = f"{datadir}/{group_id}/{run_id}/agent-[0-9]*.pt"
    files = glob.glob(pattern)
    assert len(files) > 0, f"No files found for: {pattern}"
    return max(files, key=os.path.getmtime)


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    initializable_layers = (
        torch.nn.Linear,
        torch.nn.Conv2d,
        # TODO: other layers? Conv1d?
    )

    if isinstance(layer, initializable_layers):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)

    for mod in list(layer.modules())[1:]:
        layer_init(mod, gain, bias_const)

    return layer


def safe_mean(array_like) -> float:
    return np.nan if len(array_like) == 0 else float(np.mean(array_like))


def log_params(args, wandb_log):
    logged = {}
    for (k, v) in args.logparams.items():
        value = args
        for part in v.split("."):
            assert hasattr(value, part), "No `%s` attribute for: %s" % (part, value)
            value = getattr(value, part)

        if value is None:
            value = 0
        else:
            assert isinstance(value, (int, float, bool)), "Unexpected value type: %s (%s)" % (value, type(value))

        wandb_log({k: float(value)})
        logged[k] = float(value)
    print("Params: %s" % logged)


def maybe_resume(agent_cls, args, device="cpu"):
    if not args.resume:
        print("Starting new run %s/%s" % (args.group_id, args.run_id))
        return None, args

    print("Resuming run %s/%s" % (args.group_id, args.run_id))

    # XXX: resume will overwrite all input args except run_id & group_id
    file = find_latest_save(args.group_id, args.run_id)
    agent = agent_cls.load(file, device=device)

    assert agent.args.group_id == args.group_id
    assert agent.args.run_id == args.run_id

    # XXX: both `args` and `agent.args` are of class Args, but...
    #      it is not the same class (the loaded Args is an *older snapshot*)
    #
    #      Re-initializing it to the *new* Args will:
    #      * allow to assign newly introduced fields
    #      * (FIXME) blow up for dropped fields
    #
    # # TMP fix for dropped fields
    # a = vars(agent.args)
    # a = {k: v for (k, v) in a.items() if k != "loss_weights"}
    # agent.args = args.__class__(**a)
    agent.args = args.__class__(**vars(agent.args))

    agent.args.overwrite = args.overwrite
    # agent.args.wandb_project = args.wandb_project

    # Overwrite even if None
    # This can happen in a bare resume where no config is given
    # The config at the original path may have changed completely
    # and a new upload of it now would cause confusion
    # => overwrite with None is good as the config wont be uploaded
    agent.args.cfg_file = args.cfg_file

    for argname in args.overwrite:
        parts = argname.split(".")
        if len(parts) == 1:
            print("Overwrite %s: %s -> %s" % (argname, getattr(agent.args, argname), getattr(args, argname)))
            setattr(agent.args, argname, getattr(args, argname))
        else:
            assert len(parts) == 2
            sub_loaded = getattr(agent.args, parts[0])
            sub_arg = getattr(args, parts[0])
            print("Overwrite %s: %s -> %s" % (argname, getattr(sub_loaded, parts[1]), getattr(sub_arg, parts[1])))
            setattr(sub_loaded, parts[1], getattr(sub_arg, parts[1]))

    args = agent.args
    args.resume = True
    args.agent_load_file = file

    # Useful if the file initially loaded from got rotated
    # Disabling (for PBT this is not needed)
    # with open(file, 'rb') as fsrc:
    #     backup = "%s/resumed-%s" % (os.path.dirname(file), os.path.basename(file))
    #     with open(backup, 'wb') as fdst:
    #         shutil.copyfileobj(fsrc, fdst)
    #         print("Wrote backup %s" % backup)

    return agent, args


def setup_wandb(args, agent, src_file, watch=True):
    import wandb

    if args.skip_wandb_init:
        wandb.run.tags = args.tags
    else:
        wandb.init(
            project=args.wandb_project,
            group=args.group_id,
            name=args.run_name or args.run_id,
            id=args.run_id,
            notes=args.notes,
            tags=args.tags,
            resume="must" if args.resume else "never",
            # resume="allow",  # XXX: reuse id for insta-failed runs
            config=dataclasses.asdict(args),
            sync_tensorboard=False,
            save_code=False,  # code saved manually below
            allow_val_change=args.resume,
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
        )

    # https://docs.wandb.ai/ref/python/run#log_code
    # XXX: "path" is relative to `root`
    #      but args.cfg_file is relative to vcmi-gym ROOT dir
    src_file = pathlib.Path(src_file)
    this_file = pathlib.Path(__file__)
    rl_root = this_file.parent.parent
    cfg_file = pathlib.Path(args.cfg_file) if args.cfg_file else None

    def code_include_fn(path):
        p = pathlib.Path(path).absolute()

        res = (
            p.samefile(this_file)
            or p.samefile(src_file)
            or p.samefile(rl_root / "wandb" / "requirements.txt")
            or p.samefile(rl_root / "wandb" / "requirements.lock")
            or (cfg_file and p.samefile(cfg_file))
        )

        # print("Should include %s: %s" % (path, res))
        return res

    if not args.skip_wandb_log_code:
        wandb.run.log_code(root=rl_root, include_fn=code_include_fn)

    # XXX: this will blow up with algos like MPPO-DNA which have many NNs
    #      However, no model is logged at all if using just `agent`
    #      => proper fix would be to accept a list of NNs and call wandb.watch
    #         on each of them
    if watch:
        return wandb.watch(agent.NN, log="all", log_graph=True, log_freq=1000)
    else:
        return None


def schedule_fn(schedule):
    assert schedule.mode in ["const", "lin_decay", "exp_decay"]

    if schedule.mode != "const":
        assert schedule.start > schedule.end
        assert schedule.end > 0
        assert schedule.rate > 0

    high = schedule.start
    low = schedule.end
    rate = schedule.rate

    if schedule.mode == "lin_decay":
        return lambda p: np.clip(high - (high - low) * (rate * p), low, high)
    elif schedule.mode == "exp_decay":
        return lambda p: low + (high - low) * np.exp(-rate * p)
    else:
        return lambda _: high


def validate_tags(tags):
    all_tags_file = os.path.join(os.path.dirname(__file__), "..", "wandb", "tags.yml")
    with open(all_tags_file, "r") as f:
        all_tags = yaml.safe_load(f)
    for tag in tags:
        assert tag in all_tags, f"Invalid tag: {tag}"


def coerce_dataclass_ints(dataclass_obj):
    for f in dataclasses.fields(dataclass_obj):
        v = getattr(dataclass_obj, f.name)
        if f.type == int and v is not None and not isinstance(v, int):
            assert isinstance(v, float)
            setattr(dataclass_obj, f.name, round(v))
