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
import threading
import glob
import random
import concurrent
import gymnasium as gym
import os
import time
import re
import shutil
import importlib
import numpy as np
import string
import yaml

from dataclasses import asdict
from torch.distributions.categorical import Categorical


# https://boring-guy.sh/posts/masking-rl/
# combined with
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.2.1/sb3_contrib/common/maskable/distributions.py#L18
class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        assert mask is not None
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        self.mask_value = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype)
        logits = torch.where(self.mask, logits, self.mask_value)
        super().__init__(logits=logits)

    def entropy(self):
        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask, p_log_p, torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device))
        return -p_log_p.sum(-1)


def create_venv(env_cls, args, map_swaps):
    all_maps = glob.glob("maps/%s" % args.mapmask)
    all_maps = [m.removeprefix("maps/") for m in all_maps]
    all_maps.sort()
    map_offset = None

    assert args.mapside in ["attacker", "defender", "both"]

    if args.mapside == "both":
        sides = ["attacker", "defender"]
    else:
        sides = [args.mapside]

    if args.num_envs == 1:
        n_maps = 1
    else:
        assert args.num_envs % len(sides) == 0
        assert args.num_envs <= len(all_maps) * len(sides)
        n_maps = args.num_envs // len(sides)

    if args.randomize_maps:
        maps = random.sample(all_maps, n_maps)
    else:
        i = (n_maps * map_swaps) % len(all_maps)
        new_i = (i + n_maps) % len(all_maps)
        map_offset = i

        if new_i > i:
            maps = all_maps[i:new_i]
        else:
            maps = all_maps[i:] + all_maps[:new_i]

        assert len(maps) == n_maps

    # pairs = [[("attacker", m), ("defender", m)] for m in maps]
    pairs = [[(s, m) for s in sides] for m in maps]
    pairs = [x for y in pairs for x in y]  # aka. pairs.flatten(1)...
    state = {"n": 0}
    lock = threading.RLock()

    sbm = ["StupidAI", "BattleAI", "MMAI_MODEL"]
    sbm_probs = torch.tensor(args.opponent_sbm_probs, dtype=torch.float)

    assert len(sbm_probs) == 3
    if sbm_probs[2] > 0:
        assert os.path.isfile(args.opponent_load_file)

    dist = Categorical(sbm_probs)
    opponent = sbm[dist.sample()]

    def env_creator():
        with lock:
            assert state["n"] < args.num_envs
            role, mapname = pairs[state["n"]]
            # logfile = f"/tmp/{run_id}-env{state['n']}-actions.log"
            logfile = None

            if opponent == "MMAI_MODEL":
                assert args.opponent_load_file, "opponent_load_file is required for MMAI_MODEL"

            env_kwargs = dict(
                asdict(args.env),
                mapname=mapname,
                attacker=opponent,
                defender=opponent,
                attacker_model=args.opponent_load_file,
                defender_model=args.opponent_load_file,
                actions_log_file=logfile
            )

            env_kwargs[role] = "MMAI_USER"
            print("Env kwargs (env.%d): %s" % (state["n"], env_kwargs))
            state["n"] += 1

        res = env_cls(**env_kwargs)

        for wrapper in args.env_wrappers:
            wrapper_mod = importlib.import_module(wrapper["module"])
            wrapper_cls = getattr(wrapper_mod, wrapper["cls"])
            res = wrapper_cls(res, **wrapper.get("kwargs", {}))

        return res

    if args.num_envs > 1:
        # I don't remember anymore, but there were issues if max_workers>8
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.num_envs, 8)) as executor:
            futures = [executor.submit(env_creator) for _ in range(args.num_envs)]
            results = [future.result() for future in futures]

        funcs = [lambda x=x: x for x in results]
        vec_env = gym.vector.SyncVectorEnv(funcs)
    else:
        vec_env = gym.vector.SyncVectorEnv([env_creator])

    vec_env = gym.wrappers.RecordEpisodeStatistics(vec_env, deque_size=args.stats_buffer_size)

    return vec_env, map_offset


def maybe_save(t_save, t_permasave, args, agent, out_dir):
    now = time.time()

    # Used in cases of some sweeps with hundreds of short-lived agents
    if os.environ.get("NO_SAVE", "false") == "true":
        return now, now

    if t_save is None or t_permasave is None:
        return now, now

    if t_save + args.save_every > now:
        return t_save, t_permasave

    os.makedirs(out_dir, exist_ok=True)
    agent_file = os.path.join(out_dir, "agent-%d.pt" % now)
    nn_file = os.path.join(out_dir, "nn-%d.pt" % now)
    save(agent, agent_file, nn_file=nn_file)
    t_save = now

    if t_permasave + args.permasave_every <= now:
        permasave_file = os.path.join(out_dir, "agent-permasave-%d.pt" % now)
        save(agent, permasave_file)
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
    pattern = f"data/{group_id}/{run_id}/agent-[0-9]*.pt"
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


def maybe_resume(args):
    if not args.resume:
        print("Starting new run %s/%s" % (args.group_id, args.run_id))
        return None, args

    print("Resuming run %s/%s" % (args.group_id, args.run_id))

    # XXX: resume will overwrite all input args except run_id & group_id
    file = find_latest_save(args.group_id, args.run_id)
    agent = torch.load(file)
    print("Loaded agent from %s" % file)

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

    with open(file, 'rb') as fsrc:
        backup = "%s/resumed-%s" % (os.path.dirname(file), os.path.basename(file))
        with open(backup, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
            print("Wrote backup %s" % backup)

    return agent, args


def setup_wandb(args, agent, src_file):
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
            config=asdict(args),
            sync_tensorboard=False,
            save_code=False,  # code saved manually below
            allow_val_change=args.resume,
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
        )

    # https://docs.wandb.ai/ref/python/run#log_code
    # XXX: "path" is relative to THIS dir
    #      but args.cfg_file is relative to vcmi-gym ROOT dir
    def code_include_fn(path):
        res = (
            (os.path.basename(path) == os.path.basename(__file__)) or
            path.endswith(os.path.basename(src_file)) or
            path.endswith(os.path.basename(args.cfg_file or "\u0255")) or
            path.endswith("requirements.txt") or
            path.endswith("requirements.lock")
        )

        # print("Should include %s: %s" % (path, res))
        return res

    wandb.run.log_code(root=os.path.dirname(src_file), include_fn=code_include_fn)
    return wandb.watch(agent.NN, log="all", log_graph=True, log_freq=1000)


def gen_id():
    population = string.ascii_lowercase + string.digits
    return str.join("", random.choices(population, k=8))


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
    all_tags_file = os.path.join(os.path.dirname(__file__), "config", "tags.yml")
    with open(all_tags_file, "r") as f:
        all_tags = yaml.safe_load(f)
    for tag in tags:
        assert tag in all_tags, f"Invalid tag: {tag}"


def save(agent, agent_file, nn_file=None):
    attrs = agent.save_attrs
    data = {k: agent.__dict__[k] for k in attrs}
    state_dict = agent.state_dict()
    # Re-create the entire agent to ensure it's "clean"
    clean_agent = agent.__class__(**data)
    clean_agent.load_state_dict(state_dict, strict=True)
    torch.save(clean_agent, agent_file)

    print("Saved agent to %s" % agent_file)
    # Optionally, save the NN state separately
    # Useful as it is decoupled from the Agent module (which changes often)
    if nn_file:
        torch.save(agent.NN.state_dict(), nn_file)
        print("Saved NN state to %s" % nn_file)
