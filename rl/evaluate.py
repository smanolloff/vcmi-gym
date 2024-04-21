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
import re
import gymnasium as gym
import wandb
import datetime
import os
import sys
import itertools
import numpy as np
import time
import torch
import logging
import traceback

from dataclasses import asdict


def evaluate_policy(agent, venv, episodes_per_env):
    n_envs = venv.num_envs
    counts = np.zeros(n_envs, dtype="int")
    ep_results = {"rewards": [], "lengths": [], "net_values": [], "is_successes": []}
    observations, _ = venv.reset()

    # For a vectorized environments the output will be in the form of::
    #     >>> infos = {
    #     ...     "final_observation": "<array<obs> of length num-envs>",
    #     ...     "_final_observation": "<array<bool> of length num-envs>",
    #     ...     "final_info": "<array<hash> of length num-envs>",
    #     ...     "_final_info": "<array<bool> of length num-envs>",
    #     ...     "episode": {
    #     ...         "r": "<array<float> of cumulative reward>",
    #     ...         "l": "<array<int> of episode length>",
    #     ...         "t": "<array<float> of elapsed time since beginning of episode>"
    #     ...     },
    #     ...     "_episode": "<boolean array of length num-envs>"
    #     ... }
    #
    # My notes:
    #   "episode" and "_episode" is added by RecordEpisodeStatistics wrapper
    #   gym's vec env *automatically* collects episode returns and lengths
    #   in envs.return_queue and envs.length_queue
    #   (eg. [-1024.2, 333.6, ...] and [34, 36, 41, ...]) - each element is a full episode
    #
    #  "final_info" and "_final_info" are NOT present at all if no env was done
    #   If at least 1 env was done, both are present, with info about all envs
    #   (this applies for all info keys)
    #
    #   Note that rewards are accessed as infos["episode"]["r"][i]
    #   ... but env's info is accessed as infos["final_info"][i][key]
    #
    # See
    #   https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/vector/sync_vector_env.py#L142-L157
    #   https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/vector/vector_env.py#L275-L300
    #   https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/wrappers/record_episode_statistics.py#L102-L124
    #
    while (counts < episodes_per_env).any():
        actions = agent.predict(
            torch.as_tensor(observations).float(),
            torch.as_tensor(np.array(venv.unwrapped.call("action_mask")))
        )
        observations, rewards, terms, truncs, infos = venv.step(actions)

        if "final_info" not in infos:
            continue

        # info["_episode"][i] is equivalent to done
        # (True whenever that env returned term OR trunc)
        for i, done in enumerate(infos.get("_episode", [])):
            if not done:
                continue

            assert "episode" in infos
            assert "r" in infos["episode"]
            assert "l" in infos["episode"]
            assert len(infos["episode"]["r"]) == n_envs
            assert len(infos["episode"]["l"]) == n_envs

            assert "final_info" in infos
            assert len(infos["final_info"]) == n_envs
            assert "is_success" in infos["final_info"][i]
            assert "net_value" in infos["final_info"][i]

            if counts[i] >= episodes_per_env:
                # Already done with this env
                continue

            counts[i] += 1
            ep_results["rewards"].append(infos["episode"]["r"][i])
            ep_results["lengths"].append(infos["episode"]["l"][i])
            ep_results["net_values"].append(infos["final_info"][i]["net_value"])
            ep_results["is_successes"].append(infos["final_info"][i]["is_success"])

    assert all(counts == episodes_per_env), "Wrong counts: %s" % counts
    assert all(len(v) == n_envs*episodes_per_env for v in ep_results.values()), "Wrong ep_results: %s" % ep_results
    return ep_results


def load_agent(agent_file, run_id):
    # logging.debug("Loading agent from %s" % agent_file)
    agent = torch.load(agent_file)
    assert agent.args.run_id == run_id, "%s != %s" % (agent.args.run_id, run_id)
    return agent


def create_venv(env_cls, env_kwargs, mapname, role, opponent):
    mappath = f"maps/{mapname}"
    assert os.path.isfile(mappath), "Map not found at: %s" % mappath
    assert role in ["attacker", "defender"]

    def env_creator():
        env_kwargs2 = dict(
            env_kwargs,
            random_combat=1,
            mapname=mapname,
            attacker=opponent,
            defender=opponent
        )

        env_kwargs2[role] = "MMAI_USER"
        # logging.debug("Env kwargs: %s" % env_kwargs2)
        return env_cls(**env_kwargs2)

    vec_env = gym.vector.SyncVectorEnv([env_creator])
    return gym.wrappers.RecordEpisodeStatistics(vec_env)


def wandb_init(run):
    if re.match(r"^T\d+$", run.name):
        name = f"(E) {run.name} ({run.id.removesuffix('_00000')})"
    else:
        name = f"(E) {run.name}"

    wandb.init(
        project="vcmi-gym",
        group="evaluator",
        name=name,
        id=f"eval-{run.id}",
        tags=(run.tags + ["eval"]),
        config=dict(
            orig_group=run.group,
            orig_sweep=getattr(run.sweep, "id", None),
        ),
        resume="allow",
        reinit=True,
        sync_tensorboard=False,  # no tensorboard during eval
        save_code=False,  # code saved manually below
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
    )


def find_agents():
    files = glob.glob("data/*/*/agent[0-9]*.pt")

    def should_pick(f):
        # saving large NNs may take a lot of time => wait 1min
        threshold_max = datetime.datetime.now() - datetime.timedelta(minutes=1)
        threshold_min = datetime.datetime.now() - datetime.timedelta(hours=3)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        return mtime > threshold_min and mtime < threshold_max

    filtered = [f for f in files if should_pick(f)]
    grouped = itertools.groupby(filtered, key=lambda x: x.split("/")[-2])
    grouped = {key: list(group) for key, group in grouped}

    # Eval only one PBT worker (the _00000 one)
    files2 = glob.glob("data/PBT-*/*_00000/*/agent.pt")
    filtered2 = [f for f in files2 if should_pick(f)]
    grouped2 = itertools.groupby(filtered2, key=lambda x: x.split("/")[-3])
    grouped2 = {key: list(group) for key, group in grouped2}

    grouped_all = dict(grouped, **grouped2)

    # key=run_id, value=filepath (to latest save)
    # {'attrchan-test-2-1708258565': 'data/sparse-rewards/attrchan-test-2-1708258565/model.zip', ...etc}
    return {k: max(v, key=os.path.getmtime) for k, v in grouped_all.items()}


def main():
    import vcmi_gym
    os.environ["WANDB_SILENT"] = "true"

    # Logger
    formatter = logging.Formatter("-- %(asctime)s %(levelname)s: %(message)s")
    formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
    formatter.default_msec_format = None
    loghandler = logging.StreamHandler()
    loghandler.setLevel(logging.DEBUG)
    loghandler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[loghandler])

    # prevent warnings for action_mask method
    env_cls = vcmi_gym.VcmiEnv
    evaluated = []
    venv = None
    run_id = None
    agent_load_file = None
    once = False
    api = wandb.Api()

    if len(sys.argv) == 3:
        run_id = sys.argv[1]
        agent_load_file = sys.argv[2]
        once = True

    try:
        while True:
            # find candidate agents for evaluation
            if once:
                agents = {run_id: agent_load_file}
            else:
                agents = find_agents()

            # Discard "evaluated" agents which are no longer candidates
            evaluated = [x for x in evaluated if x in agents.values()]

            logging.debug("Agents: %s\nEvaluated: %s" % (agents, evaluated))

            for run_id, agent_load_file in agents.items():
                if agent_load_file in evaluated:
                    logging.debug("Skip agent: %s (already evaluated)" % (agent_load_file))
                    continue

                logging.debug('Evaluating %s' % (agent_load_file))

                try:
                    run = api.run(f"s-manolloff/vcmi-gym/{run_id}")

                    assert run.id == run_id

                    if "no-eval" in run.tags:
                        logging.debug('Skip %s' % (agent_load_file))
                        continue

                    agent = load_agent(agent_file=agent_load_file, run_id=run_id)
                    agent.eval()

                    if os.environ.get("NO_WANDB", None) == "true":
                        def wandb_log(*args, **kwargs):
                            pass
                    else:
                        # For wandb.log, commit=True by default
                        # for wandb_log, commit=False by default
                        def wandb_log(*args, **kwargs):
                            # logging.debug("wandb.log: %s %s" % (args, kwargs))
                            wandb.log(*args, **dict({"commit": False}, **kwargs))
                        wandb_init(run)

                    wandb_log({"evaluator/busy": 0}, commit=True)
                    wandb_log({"agent/num_timesteps": agent.state.global_timestep})
                    wandb_log({"agent/num_seconds": agent.state.global_second})
                    wandb_log({"evaluator/busy": 1}, commit=True)

                    rewards = {"StupidAI": [], "BattleAI": []}
                    lengths = {"StupidAI": [], "BattleAI": []}
                    net_values = {"StupidAI": [], "BattleAI": []}
                    is_successes = {"StupidAI": [], "BattleAI": []}

                    for vmap in ["88-3stack-300K.vmap", "88-3stack-20K.vmap", "88-7stack-300K.vmap"]:
                        rewards[vmap] = []
                        lengths[vmap] = []
                        net_values[vmap] = []
                        is_successes[vmap] = []

                        for opponent in ["StupidAI", "BattleAI"]:
                            tstart = time.time()
                            venv = create_venv(env_cls, asdict(agent.args.env), f"gym/generated/evaluation/{vmap}", "attacker", opponent)
                            ep_results = evaluate_policy(agent, venv, episodes_per_env=50)

                            rewards[opponent].append(np.mean(ep_results["rewards"]))
                            lengths[opponent].append(np.mean(ep_results["lengths"]))
                            net_values[opponent].append(np.mean(ep_results["net_values"]))
                            is_successes[opponent].append(np.mean(ep_results["is_successes"]))

                            rewards[vmap].append(np.mean(ep_results["rewards"]))
                            lengths[vmap].append(np.mean(ep_results["lengths"]))
                            net_values[vmap].append(np.mean(ep_results["net_values"]))
                            is_successes[vmap].append(np.mean(ep_results["is_successes"]))

                            venv.close()

                            print("%-25s %-10s %-8s reward=%-6d net_value=%-6d is_success=%-6.2f length=%-3d (%.2fs)" % (
                                run.name,
                                vmap,
                                opponent,
                                np.mean(ep_results["rewards"]),
                                np.mean(ep_results["net_values"]),
                                np.mean(ep_results["is_successes"]),
                                np.mean(ep_results["lengths"]),
                                time.time() - tstart
                            ))

                        wandb_log({f"map/{vmap}/reward": np.mean(rewards[vmap])})
                        wandb_log({f"map/{vmap}/length": np.mean(lengths[vmap])})
                        wandb_log({f"map/{vmap}/net_value": np.mean(net_values[vmap])})
                        wandb_log({f"map/{vmap}/is_success": np.mean(is_successes[vmap])})

                        # print("%-20s %-20s reward=%-6d net_value=%-6d is_success=%-6.2f length=%-3d" % (
                        #     vmap,
                        #     np.mean(rewards[vmap]),
                        #     np.mean(net_values[vmap]),
                        #     np.mean(is_successes[vmap]),
                        #     np.mean(lengths[vmap]),
                        # ))

                    # no need to flatten lists, they are all the same lengths so np.mean just works
                    wandb_log({"opponent/StupidAI/reward": np.mean(rewards["StupidAI"])})
                    wandb_log({"opponent/StupidAI/length": np.mean(lengths["StupidAI"])})
                    wandb_log({"opponent/StupidAI/net_value": np.mean(net_values["StupidAI"])})
                    wandb_log({"opponent/StupidAI/is_success": np.mean(is_successes["StupidAI"])})

                    wandb_log({"opponent/BattleAI/reward": np.mean(rewards["BattleAI"])})
                    wandb_log({"opponent/BattleAI/length": np.mean(lengths["BattleAI"])})
                    wandb_log({"opponent/BattleAI/net_value": np.mean(net_values["BattleAI"])})
                    wandb_log({"opponent/BattleAI/is_success": np.mean(is_successes["BattleAI"])})

                    wandb_log({"all/reward": np.mean(rewards["StupidAI"] + rewards["BattleAI"])})
                    wandb_log({"all/length": np.mean(lengths["StupidAI"] + lengths["BattleAI"])})
                    wandb_log({"all/net_value": np.mean(net_values["StupidAI"] + net_values["BattleAI"])})
                    wandb_log({"all/is_success": np.mean(is_successes["StupidAI"] + is_successes["BattleAI"])}, commit=True)
                    # ^^^^^^^ commit here

                    # logging.debug("Evaluated %s: reward=%d length=%d net_value=%d is_success=%.2f" % (
                    #     run_id,
                    #     np.mean(rewards["StupidAI"] + rewards["BattleAI"]),
                    #     np.mean(lengths["StupidAI"] + lengths["BattleAI"]),
                    #     np.mean(net_values["StupidAI"] + net_values["BattleAI"]),
                    #     np.mean(is_successes["StupidAI"] + is_successes["BattleAI"]),
                    # ))

                    evaluated.append(agent_load_file)

                    # XXX: evaluator/busy is only for THIS model
                    # XXX: force "square" angles in wandb with Wall Clock axis
                    wandb_log({"evaluator/busy": 1})  # commit here as well
                    wandb_log({"evaluator/busy": 0})  # commit here as well
                except Exception as e:
                    logging.warning("Error while evaluating %s: %s", (
                        agent_load_file,
                        "\n".join(traceback.format_exception_only(e))
                    ))
                    continue

            if once:
                break

            logging.debug("Sleeping 300s...")
            time.sleep(30)
    finally:
        if venv:
            venv.close()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    main()
