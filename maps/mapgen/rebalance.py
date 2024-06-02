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
import shutil
import random
import datetime
import argparse
import numpy as np
import sqlite3

from mapgen_4096 import (
    get_all_creatures,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', help="analyze data and exit", action="store_true")
    parser.add_argument('--change', help="allow army comp change", action="store_true")
    parser.add_argument('--dry-run', help="don't save anything", action="store_true")
    parser.add_argument('map', metavar="<map>", type=str, help="map to be rebalanced (as passed to VCMI)")
    parser.add_argument('statsdb', metavar="<statsdb>", type=str, help="map's stats database file (sqlite3)")
    args = parser.parse_args()

    path, (header, objects, surface_terrain) = load(args.map)
    creatures_dict = {vcminame: (name, value) for (vcminame, name, value) in get_all_creatures()}

    db = sqlite3.connect(args.statsdb)

    print("Loading database %s ..." % args.statsdb)

    # XXX: permit stats where some heroes have 0 games?
    query = "select count(*) from (select lhero from stats where side == 0 group by lhero having sum(games) == 0)"
    [(count,)] = db.execute(query).fetchall()
    assert count == 0, f"there are {count} heroes with no stats"

    query = "select 'hero_' || lhero, 1.0*sum(wins)/sum(games) from stats where games > 0 GROUP BY lhero"
    winrates = dict(db.execute(query).fetchall())

    heroes_data = {}

    for k, v in objects.items():
        if not k.startswith("hero_"):
            continue
        assert k in winrates, "hero %s not in stats db" % k

        heroes_data[k] = {"old_army": [], "army_value": 0, "army_creatures": []}
        for stack in v["options"]["army"]:
            if not stack:
                continue
            cr_vcminame = stack["type"].removeprefix("core:")
            cr_name, cr_value = creatures_dict[cr_vcminame]
            cr_amount = stack["amount"]
            heroes_data[k]["army_value"] += cr_amount * cr_value
            heroes_data[k]["army_creatures"].append((cr_vcminame, cr_name, cr_value))
            heroes_data[k]["old_army"].append((cr_vcminame, cr_name, cr_amount))

    army_value_list = [k["army_value"] for k in heroes_data.values()]
    mean_army_value = np.mean(army_value_list)
    stddev_army_value = np.std(army_value_list)
    stddev_army_value_frac = stddev_army_value / mean_army_value

    winlist = list(winrates.values())
    mean_winrate = np.mean(winlist)
    stddev_winrate = np.std(winlist)
    stddev_winrate_frac = stddev_winrate / mean_winrate

    clip = ARMY_VALUE_CORRECTION_CLIP

    if clip is None:
        # 0.2 = 20% correction limit
        clip = 0.2

    errmax = ARMY_VALUE_ERROR_MAX
    if errmax is None:
        """
        Maximum allowed error when generating the army value is ~500
        (given peasant=15, imp=50, ogre=416, minotaur=835, etc.)
        For 5K armies, this is 0.1
        """
        errmax = 500 / mean_army_value

    print("Stats:")
    print("  mean_winrate: %d" % mean_winrate)
    print("  stddev_winrate: %d (%.2f%%)" % (stddev_winrate, stddev_winrate_frac * 100))
    print("  mean_army_value: %d" % mean_army_value)
    print("  stddev_army_value: %d (%.2f%%)" % (stddev_army_value, stddev_army_value_frac * 100))

    if args.a:
        sys.exit(0)

    if args.change:
        ALLOW_ARMY_COMP_CHANGE = True

    changed = False

    for hero_name, hero_data in heroes_data.items():
        hero_winrate = winrates[hero_name]
        correction_factor = (mean_winrate - hero_winrate) * stddev_winrate * 2
        correction_factor = np.clip(correction_factor, -clip, clip)
        if abs(correction_factor) <= errmax:
            # nothing to correct
            continue

        army_value = hero_data["army_value"]
        army_creatures = hero_data["army_creatures"]
        old_army = hero_data["old_army"]

        new_value = int(army_value * (1 + correction_factor))
        print("=== %s ===\n  winrate:\t\t%.2f (mean=%.2f)\n  value (current):\t%d\n  value (target):\t%d (%s%.2f%%)" % (
            hero_name,
            hero_winrate,
            mean_winrate,
            army_value,
            new_value,
            "+" if correction_factor > 0 else "",
            correction_factor * 100,
        ))

        new_army = None
        for r in range(1, 10):
            try:
                new_army = build_army_with_retry(new_value, errmax, creatures=army_creatures, print_creatures=False)
                print("  creatures (old):\t%s" % ", ".join([f"{number} \"{name}\"" for (_, name, number) in sorted(old_army, key=lambda x: x[1])]))
                print("  creatures (new):\t%s" % ", ".join([f"{number} \"{name}\"" for (_, name, number) in sorted(new_army, key=lambda x: x[1])]))
                break
            except MaxAttemptsExceeded:
                if not ALLOW_ARMY_COMP_CHANGE:
                    print("Give up rebuilding (ALLOW_ARMY_COMP_CHANGE=false); skipping hero")
                    new_army = old_army
                    break

                print("[%d] Trying different army composition for hero %s" % (r, hero_name))
                army_creatures = random.sample(list(creatures_dict.items()), len(army_creatures))
                army_creatures = [(a, b, c) for a, (b, c) in army_creatures]

        if not new_army:
            raise Exception("Failed to generate army for %s" % hero_name)

        objects[hero_name]["options"]["army"] = [{} for i in range(7)]

        for (slot, (vcminame, _, number)) in enumerate(new_army):
            objects[hero_name]["options"]["army"][slot] = dict(amount=number, type=f"core:{vcminame}")

        changed = True

    if not changed:
        print("Nothing to do.")
        sys.exit(1)
    elif args.dry_run:
        print("Nothing to do (--dry-run)")
    else:
        with open(os.path.join(os.path.dirname(path), "rebalance.log"), "a") as f:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report = (
                f"[{t}] Rebalanced {path} with stats: "
                f"mean_winrate={mean_winrate:.2f}, stddev_winrate={stddev_winrate:.2f} ({stddev_winrate_frac*100:.2f}%), "
                f"mean_army_value={mean_army_value:.2f}, stddev_army_value={stddev_army_value:.2f} ({stddev_army_value_frac*100:.2f}%), "
                f"max_correction={clip:.2f}\n"
            )

            f.write(report)
            print(report)

        backup(path)
        save(path, header, objects, surface_terrain)
