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
import wandb
import datetime
import os
import sys
import itertools
import numpy as np
import time
import logging
import traceback
import tempfile
import pathlib
import argparse
import copy
import sqlite3
import torch
import zipfile
import json
import re

from dataclasses import dataclass, asdict
from contextlib import contextmanager
from typing import List

import vcmi_gym


@contextmanager
def dblock(db, lock_id):
    if db is None:
        yield
    else:
        db.execute("BEGIN")
        try:
            db.execute(f"INSERT INTO locks VALUES({lock_id})")
            yield
        finally:
            # ensure transaction does not remain open
            db.execute("ROLLBACK")


def evaluate_policy(agent, venv, episodes_per_env, statsdb):
    n_envs = venv.num_envs
    counts = np.zeros(n_envs, dtype="int")
    ep_results = {"rewards": [], "lengths": [], "net_values": [], "is_successes": []}

    # must reset (the vec env requires it)
    observations, _ = venv.reset()

    # Don't count losses from resets (aka. retreats)
    statsdb.execute("update stats set wins=0, games=0")

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
        if not hasattr(agent.args, "envmaps"):
            # old scheme (without PERCENT_CUR_TO_START_TOTAL_VALUE)
            observations = observations[:, :, :, 1:]

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
    # LOG.debug("Loading agent from %s" % agent_file)
    agent = torch.load(agent_file, map_location="cpu", weights_only=False)
    assert agent.args.run_id == run_id, "%s != %s" % (agent.args.run_id, run_id)
    return agent


def create_venv(env_cls, agent, mapname, role, opponent, wrappers, env_kwargs={}):
    mappath = f"maps/{mapname}"
    assert os.path.isfile(mappath), "Map not found at: %s (cwd: %s)" % (mappath, os.getcwd())
    assert role in ["attacker", "defender"]

    def env_creator():
        kwargs = dict(
            asdict(agent.args.env),
            seed=42,
            random_heroes=1,
            random_obstacles=1,
            warmachine_chance=50,
            town_chance=0,
            mana_min=0,
            mana_max=0,
            swap_sides=agent.args.env.swap_sides,
            mapname=mapname,
        )

        kwargs = dict(kwargs, **env_kwargs)

        match env_cls:
            case vcmi_gym.VcmiEnv_v3:
                kwargs.pop("conntype", None)
                kwargs["attacker"] = opponent
                kwargs["defender"] = opponent
                kwargs[role] = "MMAI_USER"
            case vcmi_gym.VcmiEnv_v4:
                kwargs["role"] = agent.args.mapside
                kwargs["opponent"] = opponent
                kwargs["opponent_model"] = agent.args.opponent_load_file
                kwargs["conntype"] = "proc"
            case _:
                raise Exception("env cls not supported: %s" % env_cls)

        for a in kwargs.pop("deprecated_args", ["encoding_type"]):
            kwargs.pop(a, None)

        # LOG.debug("Env kwargs: %s" % kwargs)
        env = env_cls(**kwargs)
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)
        return env

    vec_env = gym.vector.SyncVectorEnv([env_creator])
    return gym.wrappers.RecordEpisodeStatistics(vec_env)


def wandb_init(run):
    return wandb.init(
        id=run.id,
        resume="must",
        reinit=True,
        sync_tensorboard=False,  # no tensorboard during eval
        save_code=False,  # code saved manually below
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
    )


def find_local_agents(LOG, _WORKER_ID, _N_WORKERS, _statedict):
    evaluated = []

    while True:
        files = glob.glob("data/*/*/agent-[0-9]*.pt")

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

        for run_id, agent_load_file in grouped_all.items():
            if agent_load_file in evaluated:
                LOG.debug("Skip agent: %s (already evaluated)" % (agent_load_file))
                continue

            LOG.debug('Evaluating %s' % (agent_load_file))

            try:
                run = wandb.Api().run(f"s-manolloff/vcmi-gym/{run_id}")
            except Exception as e:
                LOG.debug('Skip run %s due to: %s' % (agent_load_file, traceback.format_exception_only(e)))

            if "no-eval" in run.tags:
                LOG.debug('Skip %s' % (agent_load_file))
                continue

            yield run, agent_load_file, {}

        LOG.debug("Sleeping 30s...")
        time.sleep(30)


def find_remote_agents(LOG, WORKER_ID, N_WORKERS, statedict):
    with tempfile.TemporaryDirectory(prefix="vcmi-gym-evaluator") as tmpdir:
        while True:
            try:
                gt = datetime.datetime.now() - datetime.timedelta(days=3)
                runs = wandb.Api().runs(
                    path="s-manolloff/vcmi-gym",
                    filters={
                        "updatedAt": {"$gt": gt.isoformat()},
                        "tags": {"$nin": ["no-eval"], "$in": ["v4"]},
                        "display_name": "T0"
                    }
                )

                for run in runs:
                    LOG.info("Scanning artifacts of run %s (%s/%s)" % (run.name, run.group, run.id))

                    # XXX: assume artifact timestamps are in UTC ("Z" timezone)
                    artifacts = [(a, datetime.datetime.strptime(a.created_at, "%Y-%m-%dT%H:%M:%SZ")) for a in run.logged_artifacts()]

                    LOG.debug("Found %d artifacts" % len(artifacts))
                    for artifact, dt in sorted(artifacts, key=lambda x: x[1]):
                        # LOG.debug("Inspecting artifact %s" % artifact.name)
                        md = artifact.metadata

                        if not artifact.name.startswith("agent.pt:v"):
                            continue

                        version = int(artifact.version[1:])

                        if version % N_WORKERS != WORKER_ID:
                            continue
                        if md.get("evaluated", False):
                            continue
                        if dt <= gt:
                            continue

                        files = list(artifact.files())
                        assert len(files) == 1, "expected one file, got: "
                        assert files[0].name == "agent.pt"

                        # add timezone information to dt for printing correct time
                        dt = dt.replace(tzinfo=datetime.timezone.utc).astimezone()
                        LOG.info(f"Downloading artifact {artifact.name} from {time.ctime(dt.timestamp())}, step={md.get('step', '?')})")

                        f = files[0].download(tmpdir, replace=True)

                        retries = 3
                        for retry in range(retries):
                            statedict["result"] = None
                            yield run, f.name, dict(artifact.metadata, artifact_version=version)
                            if statedict["result"] == "error":
                                LOG.warning(f"Evaluation failed ({retry + 1})")
                                continue
                            break

                        if statedict["result"] == "error":
                            LOG.error(f"Giving up after {retries} retries, marking as evaluated anyway")

                        md["evaluated"] = True
                        md["evaluated_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                        md["evaluated_by"] = f"{os.uname().nodename} (PID {os.getpid()}, worker {WORKER_ID}/{N_WORKERS})"
                        md["evaluated_result"] = statedict["result"]

                        # artifact.delete(delete_aliases=True)
                        artifact.ttl = datetime.timedelta(days=1)
                        artifact.save()

            except wandb.errors.CommError:
                LOG.error("Communication error", exc_info=True)

            LOG.debug("Sleeping 30s...")
            time.sleep(30)


def find_agents(LOG, WORKER_ID, N_WORKERS, statedict):
    if os.getenv("NO_WANDB") == "true":
        return find_local_agents(LOG, WORKER_ID, N_WORKERS, statedict)
    else:
        return find_remote_agents(LOG, WORKER_ID, N_WORKERS, statedict)


# Flatten dict:
# {"a": 1, "b": {"c": 2, "d": 3}}
# => {"a": 1, "b.c": 2, "b.d": 3}
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@dataclass
class Result:
    opponent: str   # "StupidAI", "BattleAI"
    siege: bool
    reward: float
    length: float
    net_value: float
    is_success: float
    pooldata: List[float]


def main(worker_id=0, n_workers=1, database=None, watchdog_file=None, model=None):
    os.environ["WANDB_SILENT"] = "true"

    LOG = logging.getLogger("evaluator")
    LOG.setLevel(logging.DEBUG)

    WORKER_ID = worker_id
    N_WORKERS = n_workers

    assert WORKER_ID >= 0 and WORKER_ID < N_WORKERS, "worker ID must be between 0 and N_WORKERS - 1 (%d), got: %d" % (N_WORKERS - 1, WORKER_ID)

    if N_WORKERS > 1 and database is None:
        print("database is required when N_WORKERS > 1")
        sys.exit(1)

    db = None

    # To init DB: CREATE TABLE locks (id PRIMARY KEY)
    if args.database:
        db = sqlite3.connect(args.database)
        db.execute("PRAGMA busy_timeout = 60000")

        # test table
        with dblock(db, WORKER_ID):
            pass

    formatter = logging.Formatter(f"-- %(asctime)s <%(process)d> [{WORKER_ID}] %(levelname)s: %(message)s")
    formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
    formatter.default_msec_format = None
    loghandler = logging.StreamHandler()
    loghandler.setFormatter(formatter)
    LOG.addHandler(loghandler)

    evaluated = []
    venv = None
    statedict = {}

    if model:
        rid = torch.load(model, map_location="cpu", weights_only=False).args.run_id
        it = [(wandb.Api().run(f"s-manolloff/vcmi-gym/{rid}"), model, {})]
    else:
        it = find_agents(LOG, WORKER_ID, N_WORKERS, statedict)

    # store this as it will get changed when VCMI is loaded as a thread
    cwd = os.getcwd()
    print(f"CWD: {cwd}")

    n_eval_episodes = 1000  # combinations for 8x64 maps are 448

    # Build list of pool names based on the hero names in the map JSON
    vmap = "gym/generated/evaluation/8x64.vmap"
    mappath = f"maps/{vmap}"
    pools = []
    n_heroes = 0
    print("Parsing map %s" % mappath)
    with zipfile.ZipFile(mappath, 'r') as zipf:
        with zipf.open("objects.json") as f:
            for k, v in json.load(f).items():
                if not k.startswith("hero_"):
                    continue
                m = re.match(r"hero_\d+_pool_(.*)", k)
                assert m, f"invalid hero name: {k}"
                if m.group(1) not in pools:
                    pools.append(m.group(1))
                n_heroes += 1

    print("Found %d heroes in %d pools: %s" % (n_heroes, len(pools), pools))

    assert n_heroes % len(pools) == 0

    tmpdir = os.path.join(cwd, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    statsdbpath = os.path.join(tmpdir, f"evaluate-{WORKER_ID}.sqlite3")

    script_parts = [
        "DROP TABLE IF EXISTS stats;",
        "DROP TABLE IF EXISTS stats_md;"
    ]

    with open(f"{cwd}/vcmi/server/ML/sql/structure.sql", "r") as f:
        script_parts.append(re.sub(r"--.*", "", f.read()).strip())

    with open(f"{cwd}/vcmi/server/ML/sql/seed.sql", "r") as f:
        script_parts.append(re.sub(r"--.*", "", f.read()).strip())

    stats_sql_script_template = "\n".join(script_parts)

    print(f"Connecting to {statsdbpath}")
    statsdb = sqlite3.connect(statsdbpath, isolation_level=None)

    try:
        for run, agent_load_file, metadata in it:
            try:
                LOG.info('Evaluating %s (%s/%s)' % (run.name, run.group, run.id))
                agent = load_agent(agent_file=agent_load_file, run_id=run.id)
                agent.eval()

                nn = getattr(agent, "NN", getattr(agent, "NN_policy", None))
                assert nn, "agent has neighter .NN nor .NN_policy properties"

                # XXX: backport for models with old action space
                if nn.actor.out_features == 2311:
                    print("Legacy model detected -- using LegacyActionSpaceWrapper")
                    env_version = 3
                    wrappers = [vcmi_gym.LegacyActionSpaceWrapper]
                else:
                    env_version = agent.env_version
                    wrappers = []

                match env_version:
                    case 1:
                        env_cls = vcmi_gym.VcmiEnv_v1
                    case 2:
                        env_cls = vcmi_gym.VcmiEnv_v2
                    case 3:
                        env_cls = vcmi_gym.VcmiEnv_v3
                    case 4:
                        env_cls = vcmi_gym.VcmiEnv_v4
                    case _:
                        raise Exception("unsupported env version: %d" % env_version)

                side = run.config.get("mapside")
                side_int = 0 if side == "attacker" else 1
                side_col = "red" if side == "attacker" else "blue"

                stats_sql_script = stats_sql_script_template.replace("(?, ?, ?)", f"({len(pools)}, {n_heroes//len(pools)}, {side_int})")
                results = []
                wandb_results = {}

                for k, v in flatten_dict(metadata, sep=".").items():
                    wandb_results[f"eval/metadata/{k}"] = v

                for opponent in ["StupidAI", "BattleAI"]:
                    # reset db
                    statsdb.executescript(stats_sql_script)

                    for siege in [False, True]:
                        tstart = time.time()
                        venv = create_venv(
                            env_cls,
                            agent,
                            vmap,
                            side,
                            opponent,
                            wrappers,
                            env_kwargs=dict(
                                vcmi_stats_storage=statsdbpath,
                                # XXX: mid-game resets are OK (retreats are not collected as stats)
                                vcmi_stats_persist_freq=n_eval_episodes,
                                vcmi_stats_mode=side_col,
                                # vcmi_loglevel_stats="trace",
                                town_chance=100 if siege else 0
                            )
                        )

                        ep_results = evaluate_policy(agent, venv, episodes_per_env=n_eval_episodes, statsdb=statsdb)
                        venv.close()

                        rows = statsdb.execute(
                            "select pool, 1.0*sum(wins)/sum(games) as winrate "
                            "from stats "
                            "where games > 0 "
                            "group by pool "
                            "order by pool asc"
                        ).fetchall()

                        assert len(pools) == len(rows), f"{len(pools)} == {len(rows)}"

                        result = Result(
                            opponent=opponent,
                            siege=siege,
                            reward=np.mean(ep_results["rewards"]),
                            length=np.mean(ep_results["lengths"]),
                            net_value=np.mean(ep_results["net_values"]),
                            is_success=np.mean(ep_results["is_successes"]),
                            pooldata=[winrate for (_, winrate) in rows]
                        )

                        results.append(result)

                        print("%-4s %-10s %-8s siege=%d reward=%-5d net_value=%-8d is_success=%-5.2f length=%-3d (%.2fs)" % (
                            run.name,
                            os.path.basename(vmap),
                            result.opponent,
                            result.siege,
                            result.reward,
                            result.net_value,
                            result.is_success,
                            result.length,
                            time.time() - tstart
                        ))

                        print("%15s %s" % ("", "  ".join([f"{pools[i]}={f:.2f}" for (i, f) in enumerate(result.pooldata)])))

                        if watchdog_file:
                            pathlib.Path(watchdog_file).touch()

                m = os.path.basename(vmap)
                wandb_results[f"eval/map/{m}/reward"] = np.mean([r.reward for r in results])
                wandb_results[f"eval/map/{m}/length"] = np.mean([r.length for r in results])
                wandb_results[f"eval/map/{m}/net_value"] = np.mean([r.net_value for r in results])
                wandb_results[f"eval/map/{m}/is_success"] = np.mean([r.is_success for r in results])

                for o in ["StupidAI", "BattleAI"]:
                    wandb_results[f"eval/opponent/{o}/reward"] = np.mean([r.reward for r in results if r.opponent == o])
                    wandb_results[f"eval/opponent/{o}/length"] = np.mean([r.length for r in results if r.opponent == o])
                    wandb_results[f"eval/opponent/{o}/net_value"] = np.mean([r.net_value for r in results if r.opponent == o])
                    wandb_results[f"eval/opponent/{o}/is_success"] = np.mean([r.is_success for r in results if r.opponent == o])

                for i, p in enumerate(pools):
                    wandb_results[f"eval/pool/{p}/is_success"] = np.mean([r.pooldata[i] for r in results])

                if os.getenv("NO_WANDB") != "true":
                    with dblock(db, WORKER_ID):
                        with wandb_init(run) as irun:
                            irun.log(copy.deepcopy(wandb_results))
                else:
                    print("WANDB RESULTS (skipped):\n%s" % wandb_results)

                # LOG.debug("Evaluated %s: reward=%d length=%d net_value=%d is_success=%.2f" % (
                #     run.id,
                #     np.mean(rewards["StupidAI"] + rewards["BattleAI"]),
                #     np.mean(lengths["StupidAI"] + lengths["BattleAI"]),
                #     np.mean(net_values["StupidAI"] + net_values["BattleAI"]),
                #     np.mean(is_successes["StupidAI"] + is_successes["BattleAI"]),
                # ))

                evaluated.append(agent_load_file)

                # XXX: force "square" angles in wandb with Wall Clock axis
                statedict["result"] = copy.deepcopy(wandb_results)
            except KeyboardInterrupt:
                print("SIGINT received. Exiting gracefully.")
                sys.exit(0)
            except Exception as e:
                LOG.warning("Error while evaluating %s: %s" % (
                    agent_load_file,
                    "\n".join(traceback.format_exception(e))
                ))
                statedict["result"] = "error"
                continue

    finally:
        if venv:
            venv.close()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--watchdog-file', type=str, help="file to touch periodically")
    parser.add_argument('-m', '--model', type=str, help="path to model file to evaluate")
    parser.add_argument('-i', '--worker-id', type=int, default=0, help="this worker's ID (0-based)")
    parser.add_argument('-I', '--n-workers', type=int, default=1, help="total number of workers")
    parser.add_argument('-d', '--database', type=str, help="path to sqlite3 database for locking")
    args = parser.parse_args()

    main(**args.__dict__)
