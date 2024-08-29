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

# Rebalance hero armies on the given map based on statistical data
# from a database populated with VCMI's --stats-* options.
#
# Usage: python -m maps.mapgen.rebalance -h
#
# See also: `maps/mapgen/watchdog.zsh` script for automated usage
#

import os
import zipfile
import io
import json
import sys
import re
import random
import shutil
import datetime
import argparse
import copy
import numpy as np
import sqlite3
import warnings
from dataclasses import dataclass, field, replace

from .mapgen import (
    PoolConfig,
    Army,
    ArmyConfig,
    CreditsExceeded,
    NoSlotsLeft,
    UnbuildableArmy,
    UncorrectableArmy,
    Stack,
    build_pool_configs,
)

# Limit corrections to (1-clip, 1+clip) to avoid destructive updates
ARMY_VALUE_CORRECTION_CLIP = 0.2


def load(path):
    print("Loading %s" % path)

    with zipfile.ZipFile(path, 'r') as zipf:
        with zipf.open("header.json") as f:
            header = json.load(f)

        with zipf.open("objects.json") as f:
            objects = json.load(f)

        with zipf.open("surface_terrain.json") as f:
            surface_terrain = json.load(f)

    return path, (header, objects, surface_terrain)


def save(path, header, objects, surface_terrain):
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zipf:
        zipf.writestr('header.json', json.dumps(header))
        zipf.writestr('objects.json', json.dumps(objects))
        zipf.writestr('surface_terrain.json', json.dumps(surface_terrain))

    print("Creating %s" % path)
    with open(path, 'wb') as f:
        f.write(memory_zip.getvalue())


def backup(path):
    for i in reversed(range(1, 9)):
        if os.path.exists(f"{path}.{i}"):
            shutil.move(f"{path}.{i}", f"{path}.{i+1}")
    shutil.move(path, f"{path}.1")


@dataclass
class PoolRebalance:
    pool_config: PoolConfig
    min_id: int
    max_id: int = None
    winrates: dict = field(default_factory=lambda: {})
    n_games: int = 0
    n_total: int = 0     # total heroes on the map (sum of counters below)
    n_unknown: int = 0   # heroes we have no stats for
    n_skipped: int = 0   # heroes that don't need any change
    n_updated: int = 0   # heroes changed
    n_rebuilt: int = 0   # rebuilt from scratch (special cases only)
    n_failed: int = 0    # heroes failed
    armies: dict = field(default_factory=lambda: {})
    mean_army_value: float = 0.0
    stddev_army_value: float = 0.0
    stddev_army_value_frac: float = 0.0
    winlist: dict = field(default_factory=lambda: {})
    mean_winrate: float = 0.0
    stddev_winrate: float = 0.0
    stddev_winrate_frac: float = 0.0
    mean_correction: float = 0.0


class RebalanceFailed(Exception):
    pass


def rebalance_army(army_config, correction_factor):
    # The min guaranteed correction is used only during initial map generation
    new_pool_config = replace(army_config.pool_config, min_guaranteed_correction=0)
    army_config = replace(army_config, pool_config=new_pool_config)
    rebuilt = False

    for i in range(100):
        try:
            return army_config.generate_army(), i, rebuilt
        except UnbuildableArmy as e:
            if army_config.verbosity > 0:
                print(f"({i}) {e.__class__.__name__}: {str(e)}")

            weights_to_reduce = []

            if isinstance(e, NoSlotsLeft):
                # try reducing the weight of the stacks with qty == max
                # Iterate over the `army_config.creatures` instead of `stacks`
                # (`stacks` may contain more elements -- the weakest creature
                #   can occupy extra stacks at the end)
                for i, c in enumerate(army_config.creatures):
                    if e.stacks[i].qty == army_config.pool_config.stack_qty_max:
                        weights_to_reduce.append(i)

                # if no stacks are at max_qty, why is NoSlotsLeft raised?
                assert len(weights_to_reduce) > 0, f"{len(weights_to_reduce) > 0}"
            elif isinstance(e, CreditsExceeded):
                # try reducing the weight of the strongest stack with qty > 1
                #
                # Example: target=11k
                # Creatures: Angel(5k), Devil(3k), Champion(1k), Pikeman(80)
                # Initial army:
                # 2xAngel + 1xDevil + 1xChampion + 1xPikeman = 14080 (-3080)
                # => reduce the weight of the Angel, repeat for the Devil:
                # 1xAngel + 2xDevil + 1xChampion + 1xPikeman = 12080 (-1080)
                # 1xAngel + 1xDevil + 3xChampion + 1xPikeman = 11080 (-80) => OK
                for i, c in enumerate(army_config.creatures):
                    if e.stacks[i].qty > 1:
                        weights_to_reduce.append(i)
                        break

                # if no stacks have qty > 1, nothing more can be done
                # XXX: the difference with an UncorrectableArmy is that here the
                #      army cannot reach the target value at all (cannot be built).
                #      An uncorrectable army has reached it (i.e. can be built).
                if not weights_to_reduce:
                    assert all(s.qty == 1 for s in e.stacks), [s.qty for s in e.stacks]
                    raise RebalanceFailed()

            elif isinstance(e, UncorrectableArmy):
                raise RebalanceFailed()
            else:
                # ???
                raise

            # Reduce selected weights by 20%
            new_weights = copy.deepcopy(army_config.weights)
            for i in weights_to_reduce:
                new_weights[i] *= 0.8
                if army_config.verbosity > 2:
                    print(f"Reduced weight of {army_config.creatures[i].name} by 20%")

            # Re-normalize
            new_sum = sum(new_weights)
            new_weights = [w / new_sum for w in new_weights]
            army_config = replace(army_config, weights=new_weights)
        #
        # XXX: old logic where weights were randomly modified
        # except UnbuildableArmy as e:
        #     # Nudge the weights with a small amount
        #     nudge_factor = 0.1
        #     new_weights = [
        #         # if factor=0.1 => 0.9 + rand(0.2) should give 0.9..1.1
        #         w * ((1-nudge_factor) + 2*nudge_factor*random.random())
        #         for w in army_config.weights
        #     ]
        #     # Re-normalize
        #     new_sum = sum(new_weights)
        #     new_weights = [w / new_sum for w in new_weights]
        #     army_config = replace(army_config, weights=new_weights)

    raise RebalanceFailed()


def verify_data(db, objects, pool_rebalances):
    n_json_pool_heroes = {pr.pool_config.name: 0 for pr in pool_rebalances}

    for k, v in objects.items():
        if not k.startswith("hero_"):
            continue

        m = re.match(r"^hero_(\d+)_pool_([0-9A-Za-z]+)$", k)
        assert m, "invalid hero name: %s" % k
        hero_id = int(m.group(1))
        pool_name = m.group(2)
        assert pool_name not in pool_rebalances, "%s in %s" % (pool_name, list(pool_rebalances.keys()))

        # [configs <=> JSON]
        # Hero IDs in each pool
        pr = next(pr for pr in pool_rebalances if pr.pool_config.name == pool_name)
        assert hero_id >= pr.min_id, f"{hero_id} >= {pr.min_id}"
        assert hero_id <= pr.max_id, f"{hero_id} <= {pr.max_id}"
        n_json_pool_heroes[pool_name] += 1

    # [configs <=> JSON]
    # Number of pools
    assert len(pool_rebalances) == len(n_json_pool_heroes.keys())

    # [configs <=> JSON]
    # Number of heroes in each pool
    for (pool_name, pool_size) in n_json_pool_heroes.items():
        pr = next(pr for pr in pool_rebalances if pr.pool_config.name == pool_name)
        assert pr.pool_config.size == pool_size, f"[{pool_name}] {pr.pool_config.size} == {pool_size}"

    sql = (
        "select pool,"
        "       count(*),"
        "       count(distinct lhero),"
        "       count(distinct rhero),"
        "       min(lhero),"
        "       min(rhero),"
        "       max(lhero),"
        "       max(rhero) "
        "from stats group by pool"
    )

    rows = db.execute(sql).fetchall()

    # [configs <=> DB]
    # Number of pools
    assert len(pool_rebalances) == len(rows), f"{len(pool_rebalances)} == {len(rows)}"

    for (pool_id, n_pairings, n_lheroes, n_rheroes, min_lhero, min_rhero, max_lhero, max_rhero) in rows:
        pr = next(pr for pr in pool_rebalances if pr.pool_config.id == pool_id)

        # [configs <=> DB]
        # Number of heroes in each pool
        assert pr.pool_config.size == n_lheroes, f"[{pool_id}] {pr.pool_config.size} == {n_lheroes}"
        assert pr.pool_config.size == n_rheroes, f"[{pool_id}] {pr.pool_config.size} == {n_rheroes}"

        # [configs <=> DB]
        # Number of pairings in each pool
        config_n_pairings = pr.pool_config.size * (pr.pool_config.size - 1)
        assert config_n_pairings == n_pairings, f"[{pool_id}] {config_n_pairings} == {n_pairings}"

        # [configs <=> DB]
        # Hero IDs in each pool
        assert pr.min_id == min_lhero, f"[{pool_id}] {pr.min_id} == {min_lhero}"
        assert pr.min_id == min_rhero, f"[{pool_id}] {pr.min_id} == {min_rhero}"
        assert pr.max_id == max_lhero, f"[{pool_id}] {pr.max_id} == {max_lhero}"
        assert pr.max_id == max_rhero, f"[{pool_id}] {pr.max_id} == {max_rhero}"


def gather_pool_rebalance_stats(db, objects, pr):
    poolclause = f"(lhero >= {pr.min_id} AND lhero <= {pr.max_id})"

    # XXX: heroes with no stats will not be modified
    query = f"select count(*) from (select lhero from stats where {poolclause} group by lhero having sum(games) == 0)"
    [(count,)] = db.execute(query).fetchall()
    if count > 0:
        warnings.warn(f"[{pr.pool_config.name}] there are {count} heroes with no stats")

    query = f"select 'hero_' || lhero || '_pool_{pr.pool_config.name}', 1.0*sum(wins)/sum(games) from stats where {poolclause} AND games > 0 GROUP BY lhero"
    pr.winrates = dict(db.execute(query).fetchall())

    query = f"select sum(games) from stats where {poolclause}"
    pr.n_games = db.execute(query).fetchall()[0][0]

    for hero_id in range(pr.min_id, pr.max_id+1):
        # XXX: using `k` and `v` var names for historical reasons
        k = f"hero_{hero_id}_pool_{pr.pool_config.name}"
        v = objects[k]
        pr.n_total += 1

        # assert k in winrates, "hero %s not in stats db" % k
        if k not in pr.winrates:
            pr.n_unknown += 1
            continue

        stacks = []
        for jstack in v["options"]["army"]:
            if not jstack:
                continue
            vcminame = jstack["type"].removeprefix("core:")
            creature = next(c for c in pr.pool_config.creatures if c.vcminame == vcminame)
            stacks.append(Stack(
                creature=creature,
                qty=jstack["amount"],
                target_value=-1
            ))

        assert len(stacks) > 0 and len(stacks) <= 7
        pr.armies[k] = Army(None, sorted(stacks, key=lambda s: -s.creature.value))

    army_value_list = [a.value() for a in pr.armies.values()]

    pr.mean_army_value = np.mean(army_value_list)
    pr.stddev_army_value = np.std(army_value_list)
    pr.stddev_army_value_frac = pr.stddev_army_value / pr.mean_army_value

    pr.winlist = list(pr.winrates.values())
    pr.mean_winrate = np.mean(pr.winlist)
    pr.stddev_winrate = np.std(pr.winlist)
    pr.stddev_winrate_frac = pr.stddev_winrate / pr.mean_winrate

    print("[%s] Stats:" % pr.pool_config.name)
    print("[%s]   n_games: %d" % (pr.pool_config.name, pr.n_games))
    print("[%s]   mean_winrate: %.2f" % (pr.pool_config.name, pr.mean_winrate))
    print("[%s]   stddev_winrate: %.2f (%.2f%%)" % (pr.pool_config.name, pr.stddev_winrate, pr.stddev_winrate_frac * 100))
    print("[%s]   mean_army_value: %.2f" % (pr.pool_config.name, pr.mean_army_value))
    print("[%s]   stddev_army_value: %.2f (%.2f%%)" % (pr.pool_config.name, pr.stddev_army_value, pr.stddev_army_value_frac * 100))


def rebalance_pool(pr, objects, verbosity=0, debugger=False):
    changed = False
    sum_corrections = 0
    for hero_name, hero_army in pr.armies.items():
        hero_winrate = pr.winrates[hero_name]
        correction_factor = (pr.mean_winrate - hero_winrate) * pr.stddev_winrate * 2
        correction_factor = np.clip(correction_factor, -ARMY_VALUE_CORRECTION_CLIP, ARMY_VALUE_CORRECTION_CLIP)
        sum_corrections += abs(correction_factor)

        if abs(correction_factor) <= pr.pool_config.max_allowed_error:
            # nothing to correct
            pr.n_skipped += 1
            if verbosity > 1:
                print(f"=== {hero_name}: skip (correction={correction_factor:.3f}, errmax={pr.pool_config.max_allowed_error:.2f}, winrate={hero_winrate:.2f}) ===")
            continue

        new_value = int(hero_army.value() * (1 + correction_factor))

        army_config = ArmyConfig(
            pool_config=pr.pool_config,
            target_value=new_value,
            creatures=[s.creature for s in hero_army.stacks],
            weights=[s.value() / hero_army.value() for s in hero_army.stacks],
            verbosity=verbosity
        )

        if verbosity > 1:
            print("=== %s ===\n  winrate:\t\t%.2f (mean=%.2f)\n  value (current):\t%d\n  value (target):\t%d (%s%.2f%%)" % (
                hero_name,
                hero_winrate,
                pr.mean_winrate,
                hero_army.value(),
                new_value,
                "+" if correction_factor > 0 else "",
                correction_factor * 100,
            ))

        try:
            new_army, attempt, rebuilt = rebalance_army(copy.deepcopy(army_config), new_value)
            changed = True
            pr.n_updated += 1
            pr.n_rebuilt += int(rebuilt)

            # TODO: implement army.render()
            if verbosity > 1:
                print("  attempts:\t\t%d" % (attempt + 1))
                print("  army (old):\t%s" % ", ".join([s.render() for s in hero_army.stacks]))
                print("  army (new):\t%s" % ", ".join([s.render() for s in new_army.stacks]))
        except RebalanceFailed:
            new_army = hero_army  # just use the old army
            pr.n_failed += 1
            print("Failed to rebuild %s" % hero_name)
            if debugger:
                breakpoint()

        objects[hero_name]["options"]["army"] = new_army.to_json()

    pr.mean_correction = sum_corrections / pr.n_total
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', help="analyze data and exit", action="store_true")
    parser.add_argument('-v', help="be more verbose (-vvv for even more)", action="count", default=0)
    parser.add_argument('--dry-run', help="don't save anything", action="store_true")
    parser.add_argument('--seed', help="random seed", type=int)
    parser.add_argument('--debugger', help="enable breakpoint() RebalanceFailed errors", action="store_true")
    parser.add_argument('map', metavar="<map>", type=str, help="map to be rebalanced")
    parser.add_argument('statsdb', metavar="<statsdb>", type=str, help="map's stats database file (sqlite3)")
    args = parser.parse_args()

    seed = args.seed or random.randint(0, 2**31)
    print("Using seed %s" % seed)
    random.seed(seed)

    path, (header, objects, surface_terrain) = load(args.map)
    db = sqlite3.connect(args.statsdb)

    print("Loading database %s ..." % args.statsdb)

    pool_rebalances = []

    for pc in build_pool_configs():
        min_id = pc.id * pc.size
        max_id = min_id + pc.size - 1
        pool_rebalances.append(PoolRebalance(pool_config=pc, min_id=min_id, max_id=max_id))

    # Verify data integrity (configs <=> json <=> DB)
    # The pool_configs (from mapgen.py) must be the same ones
    # used during the map's initial generation.
    verify_data(db, objects, pool_rebalances)

    for pr in pool_rebalances:
        gather_pool_rebalance_stats(db, objects, pr)

    if args.a:
        # analyze only
        sys.exit(0)

    changes = [rebalance_pool(pr, objects, args.v, args.debugger) for pr in pool_rebalances]
    changed = any(changes)

    if not changed:
        print("Nothing to do.")
        sys.exit(1)
    else:
        with open(os.path.join(os.path.dirname(path), "rebalance.log"), "a") as f:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            report = f"[{t}] Rebalanced {path} with stats:"

            for pr in pool_rebalances:
                report += f"\n  - pool {pr.pool_config.name}:"
                report += f"\n    tot={pr.n_total}, "
                report += f"\n    ukn={pr.n_unknown*100.0/pr.n_total:.0f}%, "
                report += f"\n    skip={pr.n_skipped*100.0/pr.n_total:.0f}%, "
                report += f"\n    upd={pr.n_updated*100.0/pr.n_total:.0f}%, "
                report += f"\n    fail={pr.n_failed*100.0/pr.n_total:.0f}%, "
                report += f"\n    rebuilt={pr.n_rebuilt*100.0/pr.n_total:.0f}%, "
                report += f"\n    mean_winrate={pr.mean_winrate:.2f}, stddev_winrate={pr.stddev_winrate:.2f} ({pr.stddev_winrate_frac*100:.2f}%), "
                report += f"\n    mean_army_value={pr.mean_army_value:.2f}, stddev_army_value={pr.stddev_army_value:.2f} ({pr.stddev_army_value_frac*100:.2f}%), "
                report += f"\n    mean_correction={pr.mean_correction:.2f} (max_correction={ARMY_VALUE_CORRECTION_CLIP:.2f})\n"

            if args.dry_run:
                print(report)
                print("Not saving (--dry-run)")
                print("Seed: %s" % seed)
                sys.exit(0)

            f.write(report)
            print(report)
            print("Seed: %s" % seed)

        backup(path)
        save(path, header, objects, surface_terrain)


if __name__ == "__main__":
    main()
