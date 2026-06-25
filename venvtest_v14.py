import time
import torch
import numpy as np
import json
from torch_geometric.data import Batch

from rl.algos.mppo_dna_gnn.dual_vec_env import DualVecEnv
from rl.algos.mppo_dna_gnn.mppo_dna_gnn import to_hdata_list, DNAModel


if __name__ == "__main__":

    steps = 0
    step = 0

    CHECKPOINT_BASE = "ipnkyfqb-best3"

    with open(f"{CHECKPOINT_BASE}-config.json", "r") as f:
        cfg = json.load(f)

    env_kwargs = cfg["eval"]["env_variants"]["BattleAI.open"]["kwargs"]
    model = DNAModel(
        config=cfg["model"],
        device=torch.device("cpu")
    )

    weights = torch.load(f"{CHECKPOINT_BASE}-model-dna.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(weights, strict=True)

    venv = DualVecEnv(
        env_kwargs,
        0,      # StupidAI
        0,      # BattleAI
        1,      # MMAI_BATTLEAI
        0,      # model
        None,
        logprefix="eval-",
    )

    def play(venv, maxgames):
        global step

        games = 0
        wins = 0

        v_done = torch.zeros([1])

        v_obs, v_info = venv.reset()
        last_render = venv.call("render")[0]
        render0 = last_render

        while games < maxgames:
            v_hdata_list = to_hdata_list(torch.as_tensor(v_obs), v_done, venv.call("links"))
            v_hdata_batch = Batch.from_data_list(v_hdata_list).to(model.device)
            v_actsample = model.model_policy.get_actsample_eval(v_hdata_batch)
            v_obs, v_rew, v_term, v_trunc, v_info = venv.step(v_actsample.action.cpu().numpy())
            v_done = torch.as_tensor(np.logical_or(v_term, v_trunc), dtype=torch.bool)

            step += 1

            if v_done[0]:
                render1 = last_render
                games += 1

            last_render = venv.call("render")[0]

            if v_done[0]:
                print(render0)
                print(render1)
                render0 = last_render
                if v_info["final_info"]["is_success"][0]:
                    wins += 1
                    print("\\o/ ", end="")
                else:
                    print("/o\\ ", end="")

                print(CHECKPOINT_BASE)
                print("winrate: %.0f%% (%d/%d) steps=%d" % (100.0 * wins / games, wins, games, step))

                if games % 10 == 0:
                    print("%d/%d (%d%%) steps=%d" % (games, maxgames, games, step))

    ts = time.time()
    play(venv, 100)
    elapsed = time.time() - ts

    print("Elapsed: %.3fs, steps: %d, steps/s: %.2f" % (elapsed, step, step/elapsed))
