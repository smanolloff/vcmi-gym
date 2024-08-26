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

#
# Usage: python maps/mapgen/rebalance -h
#
# See also: `maps/mapgen/watchdog.zsh` script for automated usage
#

import os
import zipfile
import io
import json
import sys
import re
import shutil
import datetime
import argparse
import numpy as np
import sqlite3
import warnings
from dataclasses import dataclass, field

from mapgen_4096_pooled import (
    ALL_CREATURES,
    POOLS,
    ArmyConfig,
    build_army_with_retry,
    MaxAttemptsExceeded
)

# Max value for (unused_credits / target_value)
# (None = auto)
ARMY_VALUE_ERROR_MAX = None

# Limit corrections to (1-clip, 1+clip) to avoid destructive updates
# (None = auto)
ARMY_VALUE_CORRECTION_CLIP = None

# Change army composition if the current one does not allow for adjustment
# (overriden to True if "-c" flag is given)
ALLOW_ARMY_COMP_CHANGE = False


def load(path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(path):
        path = os.path.abspath(f"{current_dir}/../{path}")

    path = os.path.relpath(path, os.getcwd())
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
class Pool:
    name: str
    start_id: int
    cfg: ArmyConfig
    end_id: int = None
    winrates: dict = field(default_factory=lambda: {})
    n_games: int = 0
    n_total: int = 0     # total heroes on the map (sum of counters below)
    n_unknown: int = 0   # heroes we have no stats for
    n_skipped: int = 0   # heroes that don't need any change
    n_updated: int = 0   # heroes changed
    n_failed: int = 0    # heroes failed
    heroes_data: dict = field(default_factory=lambda: {})
    army_value_list: dict = field(default_factory=lambda: [])
    mean_army_value: float = 0.0
    stddev_army_value: float = 0.0
    stddev_army_value_frac: float = 0.0
    winlist: dict = field(default_factory=lambda: {})
    mean_winrate: float = 0.0
    stddev_winrate: float = 0.0
    stddev_winrate_frac: float = 0.0
    mean_correction: float = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', help="analyze data and exit", action="store_true")
    parser.add_argument('-v', help="be more verbose", action="store_true")
    parser.add_argument('--dry-run', help="don't save anything", action="store_true")
    parser.add_argument('map', metavar="<map>", type=str, help="map to be rebalanced (as passed to VCMI)")
    parser.add_argument('statsdb', metavar="<statsdb>", type=str, help="map's stats database file (sqlite3)")
    args = parser.parse_args()

    path, (header, objects, surface_terrain) = load(args.map)
    creatures_dict = {vcminame: (name, value) for (vcminame, name, value) in ALL_CREATURES}

    clip = ARMY_VALUE_CORRECTION_CLIP
    if clip is None:
        # 0.2 = 20% correction limit
        clip = 0.2

    db = sqlite3.connect(args.statsdb)

    print("Loading database %s ..." % args.statsdb)

    # {"10k": {"start_id":0,"end_id":1023}, "50k": ...}
    last_hero_id = -1
    last_pool_name = ""
    pools = {"": Pool(name="", cfg=ArmyConfig("", 0, 0), start_id=1, end_id=-1)}

    for k, v in objects.items():
        if not k.startswith("hero_"):
            continue

        m = re.match(r"^hero_(\d+)_pool_([0-9A-Za-z]+)$", k)
        assert m, "invalid hero name: %s" % k
        hero_id = int(m.group(1))
        pool_name = m.group(2)

        assert hero_id == last_hero_id + 1, f"{hero_id} != {last_hero_id} + 1"

        if pool_name != last_pool_name:
            assert pool_name not in pools, "%s in %s" % (pool_name, list(pools.keys()))
            # XXX: `POOLS` in mapgen script must contain the configs for this map
            cfg = next((ac for ac in POOLS if ac.id == pool_name))
            pools[last_pool_name].end_id = last_hero_id
            pools[pool_name] = Pool(name=pool_name, cfg=cfg, start_id=hero_id)
        last_hero_id = hero_id
        last_pool_name = pool_name

    pools[last_pool_name].end_id = last_hero_id
    del pools[""]
    print("Pools: %s" % list(pools.keys()))

    for pool in pools.values():
        poolclause = f"(lhero >= {pool.start_id} AND lhero <= {pool.end_id})"

        # XXX: heroes with no stats will not be modified
        query = f"select count(*) from (select lhero from stats where {poolclause} group by lhero having sum(games) == 0)"
        [(count,)] = db.execute(query).fetchall()
        if count > 0:
            warnings.warn(f"[{pool.name}] there are {count} heroes with no stats")

        query = f"select 'hero_' || lhero || '_pool_{pool.name}', 1.0*sum(wins)/sum(games) from stats where {poolclause} AND games > 0 GROUP BY lhero"
        pool.winrates = dict(db.execute(query).fetchall())

        query = f"select sum(games) from stats where {poolclause}"
        pool.n_games = db.execute(query).fetchall()[0][0]

        for hero_id in range(pool.start_id, pool.end_id+1):
            # XXX: using `k` and `v` var names for historical reasons
            k = f"hero_{hero_id}_pool_{pool.name}"
            v = objects[k]
            pool.n_total += 1

            # assert k in winrates, "hero %s not in stats db" % k
            if k not in pool.winrates:
                pool.n_unknown += 1
                continue

            pool.heroes_data[k] = {"old_army": [], "army_value": 0, "army_creatures": []}
            for stack in v["options"]["army"]:
                if not stack:
                    continue
                cr_vcminame = stack["type"].removeprefix("core:")
                cr_name, cr_value = creatures_dict[cr_vcminame]
                cr_amount = stack["amount"]
                pool.heroes_data[k]["army_value"] += cr_amount * cr_value
                pool.heroes_data[k]["army_creatures"].append((cr_vcminame, cr_name, cr_value))
                pool.heroes_data[k]["old_army"].append((cr_vcminame, cr_name, cr_amount))

        pool.army_value_list = [k["army_value"] for k in pool.heroes_data.values()]
        pool.mean_army_value = np.mean(pool.army_value_list)
        pool.stddev_army_value = np.std(pool.army_value_list)
        pool.stddev_army_value_frac = pool.stddev_army_value / pool.mean_army_value

        pool.winlist = list(pool.winrates.values())
        pool.mean_winrate = np.mean(pool.winlist)
        pool.stddev_winrate = np.std(pool.winlist)
        pool.stddev_winrate_frac = pool.stddev_winrate / pool.mean_winrate

        print("[%s] Stats:" % pool.name)
        print("[%s]   n_games: %d" % (pool.name, pool.n_games))
        print("[%s]   mean_winrate: %.2f" % (pool.name, pool.mean_winrate))
        print("[%s]   stddev_winrate: %.2f (%.2f%%)" % (pool.name, pool.stddev_winrate, pool.stddev_winrate_frac * 100))
        print("[%s]   mean_army_value: %.2f" % (pool.name, pool.mean_army_value))
        print("[%s]   stddev_army_value: %.2f (%.2f%%)" % (pool.name, pool.stddev_army_value, pool.stddev_army_value_frac * 100))

    if args.a:
        sys.exit(0)

    changed = False

    for pool in pools.values():
        sum_corrections = 0
        for hero_name, hero_data in pool.heroes_data.items():
            hero_winrate = pool.winrates[hero_name]
            correction_factor = (pool.mean_winrate - hero_winrate) * pool.stddev_winrate * 2
            correction_factor = np.clip(correction_factor, -clip, clip)
            sum_corrections += abs(correction_factor)

            if abs(correction_factor) <= pool.cfg.max_allowed_error:
                # nothing to correct
                pool.n_skipped += 1
                print(f"[{pool.name}] Skip {hero_name} (corr={correction_factor:.3f}, errmax={pool.cfg.max_allowed_error:.3f}, winrate={hero_winrate})")
                continue

            army_value = hero_data["army_value"]
            army_creatures = hero_data["army_creatures"]
            old_army = hero_data["old_army"]

            new_value = int(army_value * (1 + correction_factor))

            if args.v:
                print("=== %s ===\n  winrate:\t\t%.2f (mean=%.2f)\n  value (current):\t%d\n  value (target):\t%d (%s%.2f%%)" % (
                    hero_name,
                    hero_winrate,
                    pool.mean_winrate,
                    army_value,
                    new_value,
                    "+" if correction_factor > 0 else "",
                    correction_factor * 100,
                ))

            new_army = None
            for r in range(1, 10):
                try:
                    new_army = build_army_with_retry(pool.cfg, creatures=army_creatures, verbose=args.v, print_creatures=False)
                    if args.v:
                        print("  creatures (old):\t%s" % ", ".join([f"{number} \"{name}\"" for (_, name, number) in sorted(old_army, key=lambda x: x[1])]))
                        print("  creatures (new):\t%s" % ", ".join([f"{number} \"{name}\"" for (_, name, number) in sorted(new_army, key=lambda x: x[1])]))
                    changed = True
                    pool.n_updated += 1
                    break
                except MaxAttemptsExceeded as e:
                    if args.v:
                        print("Give up rebuilding %s due to: %s (ALLOW_ARMY_COMP_CHANGE=false)" % (hero_name, str(e)))
                    new_army = old_army
                    pool.n_failed += 1
                    break

            if not new_army:
                raise Exception("Failed to generate army for %s" % hero_name)

            objects[hero_name]["options"]["army"] = [{} for i in range(7)]
            for (slot, (vcminame, _, number)) in enumerate(new_army):
                objects[hero_name]["options"]["army"][slot] = dict(amount=number, type=f"core:{vcminame}")
        pool.mean_correction = sum_corrections / pool.n_total

    if not changed:
        print("Nothing to do.")
        sys.exit(1)
    else:
        with open(os.path.join(os.path.dirname(path), "rebalance.log"), "a") as f:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            report = f"[{t}] Rebalanced {path} with stats:"

            for pool in pools.values():
                report += f"\n  - pool {pool.name}:"
                report += f"\n    tot={pool.n_total}, "
                report += f"\n    ukn={pool.n_unknown*100.0/pool.n_total:.0f}%, "
                report += f"\n    skip={pool.n_skipped*100.0/pool.n_total:.0f}%, "
                report += f"\n    upd={pool.n_updated*100.0/pool.n_total:.0f}%, "
                report += f"\n    fail={pool.n_failed*100.0/pool.n_total:.0f}%, "
                report += f"\n    mean_winrate={pool.mean_winrate:.2f}, stddev_winrate={pool.stddev_winrate:.2f} ({pool.stddev_winrate_frac*100:.2f}%), "
                report += f"\n    mean_army_value={pool.mean_army_value:.2f}, stddev_army_value={pool.stddev_army_value:.2f} ({pool.stddev_army_value_frac*100:.2f}%), "
                report += f"\n    mean_correction={pool.mean_correction:.2f} (max_correction={clip:.2f})\n"

            if args.dry_run:
                print(report)
                print("Not saving (--dry-run)")
                sys.exit(0)

            f.write(report)
            print(report)

        backup(path)
        save(path, header, objects, surface_terrain)
