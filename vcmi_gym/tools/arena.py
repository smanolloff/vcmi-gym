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

import os
import argparse
import time
import torch
import json
from datetime import datetime

from rl.algos.mppo_dna_gnn.mppo_dna_gnn import (
    DNAModel,
    DualVecEnv,
    ModelLoader,
    eval_model,
)


class DummyActsample:
    def __init__(self, v_action):
        self.action = torch.tensor(list(v_action))
        self.logprob = torch.zeros(v_action.size(0))
        self.value = torch.zeros(v_action.size(0))
        self.reward = torch.zeros(v_action.size(0))
        self.done = torch.zeros(v_action.size(0))
        self.value = torch.zeros(v_action.size(0))
        self.reward = torch.zeros(v_action.size(0))

    def cpu(self):
        return self


class DummyModel:
    def __init__(self):
        self.venv = None  # set manually
        self.device = DEVICE
        self.model_policy = self
        self.model_value = self

    def get_actsample_eval(self, _):
        return DummyActsample(v_action=torch.tensor(self.venv.call("random_action")))

    def get_value(self, _):
        return torch.zeros(1)


# For eval_model
class DummyLogger:
    def debug(self):
        pass


def main(model, venv, num_vsteps, player, opponent, dual_venv_kwargs):
    assert num_vsteps % 10 == 0
    report_vsteps = num_vsteps // 10
    winrates = []
    resets = []
    times = []

    print("0%...")
    for i in range(10):
        t0 = time.time()
        stats = eval_model(DummyLogger(), model, venv, report_vsteps)
        s = time.time() - t0
        times.append(s)
        resets.append(stats.num_episodes)
        winrates.append(stats.ep_is_success_mean)

        print("%d%%... vstep=%d step=%d episode=%d resets: %d steps/s: %-6.0f resets/s: %-6.2f winrate=(%.0f%%)" % (
            10 + 10*i,
            report_vsteps + report_vsteps*i,
            venv.num_envs * (report_vsteps + report_vsteps*i),
            sum(resets),
            stats.num_episodes,
            venv.num_envs * report_vsteps/s,
            stats.num_episodes/s,
            100 * stats.ep_is_success_mean
        ))

    print("")
    print("*** Player: %s" % player)
    print("*** Opponent: %s" % opponent)
    print("*** %s" % dual_venv_kwargs)
    print("")
    print("* Total time: %.2f seconds" % sum(times))
    print("* Total steps: %d" % (num_vsteps * venv.num_envs))
    print("* Total resets: %d (%.2f resets/s)" % (sum(resets), sum(resets) / sum(times)))
    print("* Average vsteps/s: %.0f (%.0f steps/s)" % (num_vsteps / sum(times), venv.num_envs * num_vsteps / sum(times)))
    print("* Average winrate: %.0f%%" % (100 * sum(winrates) / len(winrates)))

    with open("arena.out", "a") as f:
        f.write("-- %s: %.0f%% | %s <> %s | episodes=%d %s\n" % (
            datetime.strftime(datetime.now(), "%F %T"),
            (100 * sum(winrates) / len(winrates)),
            os.path.basename(player or ""),
            os.path.basename(opponent or ""),
            sum(resets),
            " ".join(f"{k}={v}" for k, v in dual_venv_kwargs["env_kwargs"].items())
        ))


def parse_kv(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("Expected format key=value")
    key, value = text.split("=", 1)
    try:
        value = int(value)
    except ValueError:
        value = float(value)
    except ValueError:
        # string
        pass
    return key, value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", metavar="MODEL", default="rng", help="rng | <MODEL>")
    parser.add_argument("--map", metavar="MAPNAME", default="maps/gym/generated/evaluation/8x512.vmap", help="run id to use (incompatible with -f)")
    parser.add_argument("--opponent", metavar="OPPONENT", default="BattleAI", help="StupidAI | BattleAI | <MODEL>")
    parser.add_argument("--rng-role", metavar="ROLE", default="defender", help="attacker | defender")
    parser.add_argument("--num-envs", metavar="INT", type=int, default=10)
    parser.add_argument("--num-vsteps", metavar="INT", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    parser.add_argument("--envarg", action="append", type=parse_kv, metavar="KEY=VALUE", help="Env kwarg in key=value format")

    args = parser.parse_args()

    DEVICE = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu:
        DEVICE = torch.device("cuda")

    mapname = args.map.removeprefix("maps/")
    print(f"-- mapname: {mapname}")
    assert os.path.isfile(f"maps/{mapname}"), args.map

    if args.player == "rng":
        player_model = DummyModel()
        player_role = args.rng_role
    else:
        assert os.path.isfile(args.player), args.player
        player_cfgfile = args.player.replace("-model-dna.pt", "-config.json")
        print(f"-- cfgfile: {player_cfgfile}")
        assert os.path.isfile(player_cfgfile), player_cfgfile

        with open(player_cfgfile, "r") as f:
            player_cfg = json.load(f)
        player_weights = torch.load(args.player, weights_only=True, map_location=DEVICE.type)
        player_model = DNAModel(player_cfg["model"], DEVICE).eval()
        player_model.load_state_dict(player_weights, strict=True)
        player_role = player_cfg["train"]["env"]["kwargs"]["role"]

    print(f"-- Player role: {player_role}")

    env_kwargs = dict(
        mapname=mapname,
        role=player_role,
        random_heroes=1,
        random_obstacles=1,
        random_stack_chance=0,
        random_terrain_chance=100,
        warmachine_chance=0,
        town_chance=0,
    )

    env_kwargs.update(args.envarg or {})

    dual_venv_kwargs = dict(env_kwargs=env_kwargs)

    bot_loader = None

    if args.opponent in ["StupidAI", "BattleAI", "MMAI_BATTLEAI"]:
        dual_venv_kwargs[f"num_envs_{args.opponent.lower()}"] = args.num_envs
    else:
        bot_weights = args.opponent
        bot_cfgfile = bot_weights.replace("-model-dna.pt", "-config.json")

        assert os.path.isfile(bot_weights), bot_weights
        assert os.path.isfile(bot_cfgfile), bot_cfgfile

        with open(bot_cfgfile, "r") as f:
            bot_cfg = json.load(f)
        bot_role = bot_cfg["train"]["env"]["kwargs"]["role"]
        assert bot_role != player_role

        bot_loader = ModelLoader(DEVICE.type, role=bot_role)
        bot_loader.configure(bot_cfgfile)
        bot_loader.load(bot_weights)

        dual_venv_kwargs["num_envs_model"] = args.num_envs
        dual_venv_kwargs["model_loader"] = bot_loader

    venv = DualVecEnv(**dual_venv_kwargs)

    if args.player == "rng":
        player_model.venv = venv

    with torch.inference_mode():
        main(player_model, venv, args.num_vsteps, args.player, args.opponent, dual_venv_kwargs)
