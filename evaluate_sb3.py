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

import glob
import gymnasium as gym
import sb3_contrib
import wandb
import threading
import concurrent.futures
import datetime
import os
import sys
import itertools
import numpy as np
import time

import vcmi_gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy


def load_model(model_load_file):
    alg = sb3_contrib.MaskablePPO
    print("Loading %s model from %s" % (alg.__name__, model_load_file))
    model = alg.load(model_load_file)
    return model


def make_vec_env_parallel(j, env_creator, n_envs):
    def initenv():
        env = env_creator()
        env = Monitor(env, filename=None)
        return env

    if n_envs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=j) as executor:
            futures = [executor.submit(initenv) for _ in range(n_envs)]
            results = [future.result() for future in futures]

        funcs = [lambda x=x: x for x in results]
        vec_env = DummyVecEnv(funcs)
    else:
        vec_env = DummyVecEnv([initenv])

    vec_env.seed()
    return vec_env


def create_venv(n_envs, env_cls, run_id, mapmask, opponent):
    mappath = "/Users/simo/Library/Application Support/vcmi/Maps"
    all_maps = glob.glob("%s/%s" % (mappath, mapmask))
    all_maps = [m.replace("%s/" % mappath, "") for m in all_maps]
    all_maps.sort()

    assert n_envs % 2 == 0
    assert n_envs == len(all_maps) * 2

    pairs = [[("attacker", m), ("defender", m)] for m in all_maps]
    pairs = [x for y in pairs for x in y]  # aka. pairs.flatten(1)...
    state = {"n": 0}
    lock = threading.RLock()

    def env_creator(**_env_kwargs):
        with lock:
            assert state["n"] < n_envs
            role, mapname = pairs[state["n"]]
            # logfile = f"/tmp/{run_id}-env{state['n']}-actions.log"
            logfile = None

            env_kwargs2 = dict(
                # env_kwargs,
                mapname=mapname,
                attacker=opponent,
                defender=opponent,
                actions_log_file=logfile
            )

            env_kwargs2[role] = "MMAI_USER"
            print("Env kwargs (env.%d): %s" % (state["n"], env_kwargs2))
            state["n"] += 1

        return env_cls(**env_kwargs2)

    return make_vec_env_parallel(min(n_envs, 8), env_creator, n_envs=n_envs)


def wandb_init(id, group):
    # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/air/integrations/wandb.py#L601-L607
    wandb.init(
        id=id,
        name=id,
        resume="allow",
        reinit=True,
        # To disable System/ stats:
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        group=group,
        project="vcmi",
        sync_tensorboard=False,  # no tensorboard during eval
    )


def find_models():
    # XXX: model.zip is no good as it may get overwritten every ~7 minutes during load
    # model-%d.zip is static, but is being created every 2 hours
    threshold = datetime.datetime.now() - datetime.timedelta(hours=3)
    files = glob.glob("data/*/*/model-[0-9]*.zip")
    assert len(files) > 0, "No files found"
    filtered = [f for f in files if datetime.datetime.fromtimestamp(os.path.getmtime(f)) > threshold]
    grouped = itertools.groupby(filtered, key=lambda x: x.split("/")[-2])

    # {'attrchan-test-2-1708258565': 'data/sparse-rewards/attrchan-test-2-1708258565/model.zip', ...etc}
    return {k: max(v, key=os.path.getmtime) for k, v in grouped}


if __name__ == "__main__":
    # prevent warnings for action_masks method
    gym.logger.set_level(gym.logger.ERROR)
    env_cls = vcmi_gym.VcmiEnv
    evaluated = []
    venv = None
    run_id = None
    model_load_file = None
    once = False

    if len(sys.argv) == 3:
        run_id = sys.argv[1]
        model_load_file = sys.argv[2]
        once = True

    try:
        while True:
            # find candidate models for evaluation
            if once:
                models = {run_id: model_load_file}
            else:
                models = find_models()

            # Discard "evaluated" models which are no longer candidates
            evaluated = [x for x in evaluated if x in models.values()]

            print("Models: %s\nEvaluated: %s" % (models, evaluated))

            for run_id, model_load_file in models.items():
                if model_load_file in evaluated:
                    print("Skip model: %s (already evaluated)" % (model_load_file))
                    continue

                print("*** Evaluating model %s" % model_load_file)
                model = load_model(model_load_file=model_load_file)
                wandb_init(id=f"eval-{run_id}", group="evaluation")
                timestamp = int(model_load_file.split("-")[-1][:-4])
                wandb.log({"model/timestamp": timestamp}, commit=False)
                wandb.log({"evaluator/busy": 1})  # commit here as well

                # List of (rew, len) tuples, where rew and len are lists of mean(ep_reward) and ep_len
                rewards = {"StupidAI": [], "BattleAI": []}
                lengths = {"StupidAI": [], "BattleAI": []}

                for vmap in ["T01.vmap", "T02.vmap", "T03.vmap", "T04.vmap"]:
                    rewards[vmap] = []
                    lengths[vmap] = []
                    for opponent in ["StupidAI", "BattleAI"]:
                        tstart = time.time()
                        venv = create_venv(2, env_cls, run_id, f"ai/generated/{vmap}", opponent)
                        ep_rewards, ep_lengths = evaluate_policy(model=model, env=venv, n_eval_episodes=100, return_episode_rewards=True)  # noqa: E501
                        rewards[opponent].append(np.mean(ep_rewards))
                        lengths[opponent].append(np.mean(ep_lengths))
                        rewards[vmap].append(np.mean(ep_rewards))
                        lengths[vmap].append(np.mean(ep_lengths))
                        venv.close()
                        print("%s/%s: reward=%d length=%d (%.2fs)" % (vmap, opponent, np.mean(ep_rewards), np.mean(ep_lengths), time.time() - tstart))
                    wandb.log({f"map/{vmap}/reward": np.mean(rewards[vmap])}, commit=False)
                    wandb.log({f"map/{vmap}/length": np.mean(lengths[vmap])}, commit=False)
                    print("%s: reward=%d length=%d" % (vmap, np.mean(rewards[vmap]), np.mean(lengths[vmap])))

                # no need to flatten lists, they are all the same lengths so np.mean just works
                wandb.log({"opponent/StupidAI/reward": np.mean(rewards["StupidAI"])}, commit=False)
                wandb.log({"opponent/StupidAI/length": np.mean(lengths["StupidAI"])}, commit=False)
                wandb.log({"opponent/BattleAI/reward": np.mean(rewards["BattleAI"])}, commit=False)
                wandb.log({"opponent/BattleAI/length": np.mean(lengths["BattleAI"])}, commit=False)
                wandb.log({"all/reward": np.mean(rewards["StupidAI"] + rewards["BattleAI"])}, commit=False)
                wandb.log({"all/length": np.mean(lengths["StupidAI"] + lengths["BattleAI"])}, commit=False)

                print("Evaluated %s: reward=%d length=%d" % (
                    run_id,
                    np.mean(rewards["StupidAI"] + rewards["BattleAI"]),
                    np.mean(lengths["StupidAI"] + lengths["BattleAI"]),
                ))

                evaluated.append(model_load_file)

                # XXX: evaluator/busy is only for THIS model
                wandb.log({"evaluator/busy": 0})  # commit here as well

            if once:
                break

            print("Sleeping 300s...")
            time.sleep(30)
    finally:
        if venv:
            venv.close()
        wandb.finish(quiet=True)
