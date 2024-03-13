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


def create_venv(env_cls, args, writer, map_swaps):
    mappath = "/Users/simo/Library/Application Support/vcmi/Maps"
    all_maps = glob.glob("%s/%s" % (mappath, args.mapmask))
    all_maps = [m.replace("%s/" % mappath, "") for m in all_maps]
    all_maps.sort()

    if args.num_envs == 1:
        n_maps = 1
    else:
        assert args.num_envs % 2 == 0
        assert args.num_envs <= len(all_maps) * 2
        n_maps = args.num_envs // 2

    if args.randomize_maps:
        maps = random.sample(all_maps, n_maps)
    else:
        i = (n_maps * map_swaps) % len(all_maps)
        new_i = (i + n_maps) % len(all_maps)
        # wandb.log({"map_offset": i}, commit=False)
        writer.add_scalar("global/map_offset", i)

        if new_i > i:
            maps = all_maps[i:new_i]
        else:
            maps = all_maps[i:] + all_maps[:new_i]

        assert len(maps) == n_maps

    pairs = [[("attacker", m), ("defender", m)] for m in maps]
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

    vec_env = gym.wrappers.RecordEpisodeStatistics(vec_env)
    return vec_env


def maybe_save(t, args, agent, optimizer, out_dir):
    now = time.time()

    if t is None:
        return now

    if t + args.save_every > now:
        return t

    os.makedirs(out_dir, exist_ok=True)
    agent_file = os.path.join(out_dir, "agent-%d.pt" % now)
    agent.state.optimizer_state_dict = optimizer.state_dict()
    torch.save(agent, agent_file)
    print("Saved agent to %s" % agent_file)

    args_file = os.path.join(out_dir, "args-%d.pt" % now)
    torch.save(args, args_file)
    print("Saved args to %s" % args_file)

    # save file retention (keep latest N saves)
    files = sorted(
        glob.glob(os.path.join(out_dir, "agent-[0-9]*.pt")),
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()),
        reverse=True
    )

    for file in files[args.max_saves:]:
        print("Deleting %s" % file)
        os.remove(file)
        argfile = "%s/args-%s" % (os.path.dirname(file), os.path.basename(file).removeprefix("agent-"))
        if os.path.isfile(argfile):
            print("Deleting %s" % argfile)
            os.remove(argfile)

    return now


def find_latest_save(group_id, run_id):
    pattern = f"data/{group_id}/{run_id}/agent-[0-9]*.pt"
    files = glob.glob(pattern)
    assert len(files) > 0, f"No files found for: {pattern}"

    agent_file = max(files, key=os.path.getmtime)
    args_file = "%s/args-%s" % (os.path.dirname(agent_file), os.path.basename(agent_file).removeprefix("agent-"))
    assert os.path.isfile(args_file)

    return args_file, agent_file


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


def log_params(args, writer):
    for (k, v) in args.logparams.items():
        value = args
        for part in v.split("."):
            assert hasattr(value, part), "No `%s` attribute for: %s" % (part, value)
            value = getattr(value, part)

        if value is None:
            value = 0
        else:
            assert isinstance(value, (int, float, bool)), "Unexpected value type: %s (%s)" % (value, type(value))

        writer.add_scalar(k, float(value))
        print("%s: %s" % (k, float(value)))


def maybe_resume_args(args):
    if not args.resume:
        print("Starting new run %s/%s" % (args.group_id, args.run_id))
        return args

    # XXX: resume will overwrite all input args except run_id & group_id
    args_load_file, agent_load_file = find_latest_save(args.group_id, args.run_id)
    loaded_args = torch.load(args_load_file)
    assert loaded_args.group_id == args.group_id
    assert loaded_args.run_id == args.run_id

    for f in [args_load_file, agent_load_file]:
        backup = "%s/resumed-%s" % (os.path.dirname(f), os.path.basename(f))
        with open(f, 'rb') as fsrc:
            with open(backup, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
                print("Wrote backup %s" % backup)

    # List of arg names to overwrite after loading
    # (some args (incl. overwrite itself) must always be overwritten)
    loaded_args.overwrite = args.overwrite
    loaded_args.wandb = args.wandb

    if args.cfg_file:
        loaded_args.cfg_file = args.cfg_file

    for argname in args.overwrite:
        parts = argname.split(".")
        if len(parts) == 1:
            print("Overwrite %s: %s -> %s" % (argname, getattr(loaded_args, argname), getattr(args, argname)))
            setattr(loaded_args, argname, getattr(args, argname))
        else:
            assert len(parts) == 2
            sub_loaded = getattr(args, parts[0])
            sub_arg = getattr(args, parts[0])
            print("Overwrite %s: %s -> %s" % (argname, getattr(sub_loaded, parts[1]), getattr(sub_arg, args[1])))
            setattr(sub_loaded, parts[1], getattr(sub_arg, parts[1]))

    args = loaded_args
    args.resume = True
    args.agent_load_file = agent_load_file

    print("Resuming run %s/%s" % (args.group_id, args.run_id))
    print("Loaded args from %s" % args_load_file)
    return args


def maybe_setup_wandb(args, src_file):
    if not args.wandb:
        return

    import wandb

    wandb.init(
        project="vcmi-gym",
        group=args.group_id,
        name=args.run_id,
        id=args.run_id,
        notes=args.notes,
        resume="must" if args.resume else "never",
        # resume="allow",  # XXX: reuse id for insta-failed runs
        config=asdict(args),
        sync_tensorboard=True,
        save_code=False,  # code saved manually below
        allow_val_change=args.resume,
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
    )

    # https://docs.wandb.ai/ref/python/run#log_code
    # XXX: "path" is relative to THIS dir
    #      but args.cfg_file is relative to vcmi-gym ROOT dir
    def code_include_fn(path):
        res = (
            path.endswith(os.path.basename(src_file)) or
            path.endswith(os.path.basename(args.cfg_file or "\u0255")) or
            path.endswith("requirements.txt") or
            path.endswith("requirements.lock")
        )

        print("Should include %s: %s" % (path, res))
        return res

    wandb.run.log_code(root=os.path.dirname(src_file), include_fn=code_include_fn)


def init_optimizer(args, agent, optimizer):
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if agent.state.optimizer_state_dict:
        print("Loading optimizer from stored state")
        optimizer.load_state_dict(agent.state.optimizer_state_dict)

    # Need to explicitly set lr after loading state
    # When resuming runs, explicitly check lr (it's easy to mess it up)
    if args.resume and "learning_rate" not in args.overwrite:
        assert optimizer.param_groups[0]["lr"] == args.learning_rate
    else:
        optimizer.param_groups[0]["lr"] = args.learning_rate

    if args.resume and "weight_decay" not in args.overwrite:
        assert optimizer.param_groups[0]["weight_decay"] == args.weight_decay
    else:
        optimizer.param_groups[0]["weight_decay"] = args.weight_decay

    print("Learning rate: %s" % optimizer.param_groups[0]["lr"])
    print("Weight decay: %s" % optimizer.param_groups[0]["weight_decay"])
    return optimizer
