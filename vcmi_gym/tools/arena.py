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

OLD_TO_NEW_EDGE_MODULE_NAMES = {
    "<Global___Has___Action>": "<Global___To___Action>",
    "<Unit___By___Action>": "<Unit___Has___Action>",
}


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


def migrate_edge_key_typos(state_dict):
    migrated = {}

    for key, value in state_dict.items():
        new_key = key
        for old, new in OLD_TO_NEW_EDGE_MODULE_NAMES.items():
            new_key = new_key.replace(old, new)

        assert new_key not in migrated, f"Checkpoint migration collision: both old and new keys map to {new_key}"
        migrated[new_key] = value

    return migrated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", metavar="MODEL", default="rng", help="rng | <MODEL>")
    parser.add_argument("--map", metavar="MAPNAME", default="maps/gym/ml-eval.vmap", help="run id to use (incompatible with -f)")
    parser.add_argument("--opponent", metavar="OPPONENT", default="BattleAI", help="StupidAI | BattleAI | <MODEL>")  # model may be .onnx or .pt
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

    assert os.path.isfile(args.player), args.player
    player_cfgfile = "-".join(args.player.split("-", 2)[:2] + ["config.json"])
    print(f"-- cfgfile: {player_cfgfile}")
    assert os.path.isfile(player_cfgfile), player_cfgfile

    with open(player_cfgfile, "r") as f:
        player_cfg = json.load(f)

    pw = torch.load(args.player, weights_only=True, map_location=DEVICE.type)

    env_kwargs = dict(
        mapname=mapname,
        random_heroes=1,
        random_obstacles=1,
        # random_stack_chance=0,
        random_terrain_chance=100,
        warmachine_chance=0,
        town_chance=0,
    )

    env_kwargs.update(args.envarg or {})
    dual_venv_kwargs = dict(env_kwargs=env_kwargs)

    is_player_v15 = False
    is_bot_v15 = False

    if player_cfg.get("version", None) == 15 or player_cfg["wandb_group"] == "v15":
        from rl.v15.ppo_gnn import (
            VcmiEnv,
            PPOModel,
            DualVecEnv,
            eval_model,
        )
        # TODO: use func from rl/v15/ppo_gnn.py
        pw = migrate_edge_key_typos(pw)
        player_model = PPOModel(
            node_types=VcmiEnv.node_types(),
            edge_types=VcmiEnv.filtered_edge_types(player_cfg["train"]["env"]["kwargs"]["ignored_edges"]),
            config=player_cfg["model"],
            device=DEVICE
        ).eval()
        is_player_v15 = True
    else:
        from rl.algos.mppo_dna_gnn.mppo_dna_gnn import (
            DNAModel,
            DualVecEnv,
            eval_model,
        )

        import vcmi_gym.envs.v13.pyconnector as player_pyconnector_v13
        import vcmi_gym.envs.v14.pyconnector as player_pyconnector_v14
        from vcmi_gym.envs.v13.vcmi_env import VcmiEnv as player_VcmiEnv_v13
        from vcmi_gym.envs.v14.vcmi_env import VcmiEnv as player_VcmiEnv_v14

        state_size_one_hex = pw["model_policy.encoder_hexes.layers.module_0.convs.<hex___ADJACENT___hex>.lin_src.weight"].shape[1]
        if state_size_one_hex == player_pyconnector_v13.STATE_SIZE_ONE_HEX:
            print("[player] Using v13 connector")
            dual_venv_kwargs["env_version"] = 13
            player_pyconnector = player_pyconnector_v13
            player_obs_space = player_VcmiEnv_v13.OBSERVATION_SPACE
        elif state_size_one_hex == player_pyconnector_v14.STATE_SIZE_ONE_HEX:
            print("[player] Using v14 connector")
            dual_venv_kwargs["env_version"] = 14
            player_pyconnector = player_pyconnector_v14
            player_obs_space = player_VcmiEnv_v14.OBSERVATION_SPACE

        constants = {
            "STATE_SIZE": player_pyconnector.STATE_SIZE,
            "STATE_SIZE_ONE_HEX": player_pyconnector.STATE_SIZE_ONE_HEX,
            "STATE_SIZE_HEXES": player_pyconnector.STATE_SIZE_HEXES,
            "N_ACTIONS": player_pyconnector.N_ACTIONS,
            "N_HEX_ACTIONS": player_pyconnector.N_HEX_ACTIONS,
            "N_NONHEX_ACTIONS": player_pyconnector.N_NONHEX_ACTIONS,
            "GLOBAL_ATTR_MAP": player_pyconnector.GLOBAL_ATTR_MAP,
            "GLOBAL_ACT_MAP": player_pyconnector.GLOBAL_ACT_MAP,
            "HEX_ATTR_MAP": player_pyconnector.HEX_ATTR_MAP,
            "HEX_ACT_MAP": player_pyconnector.HEX_ACT_MAP,
            "LINK_ATTR_SIZES": player_pyconnector.LINK_ATTR_SIZES,
        }

        if any("encoder_other" in k for k in pw.keys()):
            player_cfg["model"]["legacy_global_encoder"] = True
        else:
            player_cfg["model"]["legacy_global_encoder"] = False

        player_model = DNAModel(
            config=player_cfg["model"],
            constants=constants,
            obs_space=player_obs_space,
            device=DEVICE,
        ).eval()

    player_model.load_state_dict(pw, strict=True)
    player_role = player_cfg["train"]["env"]["kwargs"]["role"]

    print(f"-- Player role: {player_role}")

    env_kwargs["role"] = player_role

    bot_loader = None

    if args.opponent in ["StupidAI", "BattleAI", "MMAI_BATTLEAI"]:
        dual_venv_kwargs[f"num_envs_{args.opponent.lower()}"] = args.num_envs
    elif args.opponent.endswith(".onnx"):
        # opponent=MMAI_ONNX is ultimately mapped to opponent=MMAI_MODEL for VcmiEnv
        # but since we have another "model" here (the torch model)
        # => use MMAI_ONNX to avoid confusion
        dual_venv_kwargs["num_envs_mmai_onnx"] = args.num_envs
        dual_venv_kwargs["onnx_model"] = args.opponent
    else:
        assert args.opponent.endswith(".pt")

        bot_weights = args.opponent
        bot_cfgfile = "-".join(bot_weights.split("-", 2)[:2] + ["config.json"])

        assert os.path.isfile(bot_weights), bot_weights
        assert os.path.isfile(bot_cfgfile), bot_cfgfile

        with open(bot_cfgfile, "r") as f:
            bot_cfg = json.load(f)
        bot_role = bot_cfg["train"]["env"]["kwargs"]["role"]
        assert bot_role != player_role

        bw = torch.load(bot_weights, weights_only=True, map_location=DEVICE.type)

        if bot_cfg.get("version", None) == 15 or bot_cfg["wandb_group"] == "v15":
            is_bot_v15 = True
            # TODO: ppo or dna loader depending on bot weights
            if any("dna" in k for k in bw.keys()):
                print("[bot] Using v15 (DNA) model loader")
                from rl.v15.dna_gnn import ModelLoader
            else:
                print("[bot] Using v15 (PPO) model loader")
                from rl.v15.ppo_gnn import ModelLoader
        else:
            if any("encoder_other" in k for k in bw.keys()):
                bot_cfg["model"]["legacy_global_encoder"] = True
            else:
                bot_cfg["model"]["legacy_global_encoder"] = False

            from rl.algos.mppo_dna_gnn.mppo_dna_gnn import ModelLoader

            import vcmi_gym.envs.v13.pyconnector as bot_pyconnector_v13
            import vcmi_gym.envs.v14.pyconnector as bot_pyconnector_v14
            from vcmi_gym.envs.v13.vcmi_env import VcmiEnv as bot_VcmiEnv_v13
            from vcmi_gym.envs.v14.vcmi_env import VcmiEnv as bot_VcmiEnv_v14

            state_size_one_hex = bw["model_policy.encoder_hexes.layers.module_0.convs.<hex___ADJACENT___hex>.lin_src.weight"].shape[1]
            if state_size_one_hex == bot_pyconnector_v13.STATE_SIZE_ONE_HEX:
                print("Using v13 connector")
                bot_pyconnector = bot_pyconnector_v13
                bot_obs_space = bot_VcmiEnv_v13.OBSERVATION_SPACE
            elif state_size_one_hex == bot_pyconnector_v14.STATE_SIZE_ONE_HEX:
                print("Using v14 connector")
                bot_pyconnector = bot_pyconnector_v14
                bot_obs_space = bot_VcmiEnv_v14.OBSERVATION_SPACE

            constants = {
                "STATE_SIZE": bot_pyconnector.STATE_SIZE,
                "STATE_SIZE_ONE_HEX": bot_pyconnector.STATE_SIZE_ONE_HEX,
                "STATE_SIZE_HEXES": bot_pyconnector.STATE_SIZE_HEXES,
                "N_ACTIONS": bot_pyconnector.N_ACTIONS,
                "N_HEX_ACTIONS": bot_pyconnector.N_HEX_ACTIONS,
                "N_NONHEX_ACTIONS": bot_pyconnector.N_NONHEX_ACTIONS,
                "GLOBAL_ATTR_MAP": bot_pyconnector.GLOBAL_ATTR_MAP,
                "GLOBAL_ACT_MAP": bot_pyconnector.GLOBAL_ACT_MAP,
                "HEX_ATTR_MAP": bot_pyconnector.HEX_ATTR_MAP,
                "HEX_ACT_MAP": bot_pyconnector.HEX_ACT_MAP,
                "LINK_ATTR_SIZES": bot_pyconnector.LINK_ATTR_SIZES,
            }

            bot_loader = ModelLoader(
                device_type=DEVICE.type,
                role=bot_role,
                constants=constants,
                obs_space=bot_obs_space,
            )

        # DualVecEnv would need separate SHM buffers for the v14 and v15 observations
        # => can't happen
        if bot_loader and is_bot_v15 != is_player_v15:
            raise Exception(f"can't mix v15 and non-v15 models: is_player_v15={is_player_v15} and is_bot_v15={is_bot_v15}")

        bot_loader.configure(bot_cfg)
        bot_loader.load(bot_weights)
        dual_venv_kwargs["num_envs_model"] = args.num_envs
        dual_venv_kwargs["model_loader"] = bot_loader

    dual_venv_kwargs["env_kwargs"]["vcmi_loglevel_ai"] = "debug"

    print(dual_venv_kwargs)
    venv = DualVecEnv(**dual_venv_kwargs)

    if args.player == "rng":
        player_model.venv = venv

    with torch.inference_mode():
        main(player_model, venv, args.num_vsteps, args.player, args.opponent, dual_venv_kwargs)
