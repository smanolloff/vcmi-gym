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

# Given the script is invoked with the following JSON:
#
#   {
#     "map": "gym/generated/88/88-1stack-01.vmap",
#     "battles": 43000,
#     "wins": {"hero_3": 1039, "hero_4":1242, "hero_5":975, ...}
#   }
#
# ("map" is absolute or relative to "<script_dir>/..")
# rebalance the value of each hero's army with respect to the winrate.
#
# Script also supports STDIN, can be used to continuously rebalance a map
# like so (fish shell):
#
# for i in (seq 5)
#   $VCMI/rel/bin/myclient-headless --gymdir $VCMIGYM \
#       --map gym/generated/88/88-7stack-300K-00.vmap --loglevel-ai error \
#       --loglevel-global error --attacker-ai StupidAI --defender-ai StupidAI \
#       --random-combat 1 --map-eval 10000 | python maps/mapgen/rebalance.py -
#
#   if test $status -ne 0
#       # means no more optimization to do or a script error
#       break
#   end
# end
#
# Note VCMI terminates rather abruptly (exit(0) from a non-main thread), so
# this method is not very good.


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
from math import log

from mapgen import (
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
    parser.add_argument('-f', help="read input json from file", metavar="FILE")
    parser.add_argument('-a', help="analyze input and exit", action='store_true')
    parser.add_argument('--change', help="allow army comp change", action='store_true')
    parser.add_argument('--dry-run', help="don't save anything", action='store_true')
    args = parser.parse_args()

    j = None
    if (args.f):
        with open(args.f, "r") as file:
            j = file.read()
    else:
        print("Waiting for stdin...")
        try:
            j = input()
            print("Got json: %s" % j)
        except EOFError:
            pass

    j = json.loads(j)
    path, (header, objects, surface_terrain) = load(j["map"])
    creatures_dict = {vcminame: (name, value) for (vcminame, name, value) in get_all_creatures()}

    # analyze map
    heroes_data = {}

    for (k, v) in objects.items():
        if not k.startswith("hero_"):
            continue

        heroes_data[k] = {"old_army": [], "army_value": 0, "army_creatures": []}
        for stack in v["options"]["army"]:
            if not stack:
                continue
            cr_vcminame = stack["type"].removeprefix("core:")
            cr_name, cr_value = creatures_dict[cr_vcminame]
            cr_amount = stack["amount"]
            heroes_data[k]["army_value"] += cr_amount * cr_value
            heroes_data[k]["army_creatures"].append((cr_vcminame, cr_name, cr_value))
            heroes_data[k]["old_army"].append((cr_vcminame, None, cr_amount))

    army_value_list = [k["army_value"] for k in heroes_data.values()]
    mean_army_value = np.mean(army_value_list)
    stddev_army_value = np.std(army_value_list)
    stddev_army_value_frac = stddev_army_value / mean_army_value

    winlist = list(j["wins"].values())
    mean_wins = np.mean(winlist)
    stddev_wins = np.std(winlist)
    stddev_wins_frac = stddev_wins / mean_wins

    clip = ARMY_VALUE_CORRECTION_CLIP
    if clip is None:
        """
        A 0.1 (10%) correction limit is good. Example:

            mean_wins = 30
            examples = [1, 5, 10, 20, 25, 30, 35, 50, 100]
            corrections = ["%d => %.2f" % (w, log(mean_wins/w)/30) for w in examples]
            print("corrections (mean_wins=%d):\n%s" % (mean_wins, "\n".join(corrections)))

        corrections (mean_wins=30):
        1 => 0.11
        5 => 0.06
        10 => 0.04
        20 => 0.01
        25 => 0.01
        30 => 0.00
        35 => -0.01
        50 => -0.02
        100 => -0.04
        """
        clip = 0.1

    errmax = ARMY_VALUE_ERROR_MAX
    if errmax is None:
        """
        Maximum allowed error when generating the army value is ~1000
        (given peasant=15, imp=50, ogre=416, minotaur=835, etc.)
        For 100K armies, this is 0.01
        """
        errmax = 1000 / mean_army_value

    print("Stats:")
    print("  mean_wins: %d" % mean_wins)
    print("  stddev_wins: %d (%.2f%%)" % (stddev_wins, stddev_wins_frac * 100))
    print("  mean_army_value: %d" % mean_army_value)
    print("  stddev_army_value: %d (%.2f%%)" % (stddev_army_value, stddev_army_value_frac * 100))

    if args.a:
        sys.exit(0)

    if args.change:
        ALLOW_ARMY_COMP_CHANGE = True

    changed = False

    for hero_name, hero_data in heroes_data.items():
        hero_wins = j["wins"].get(hero_name, 0)
        correction_factor = log(mean_wins/(hero_wins or 1)) / 30
        correction_factor = np.clip(correction_factor, -clip, clip)
        if abs(correction_factor) <= errmax:
            # nothing to correct
            continue

        army_value = hero_data["army_value"]
        army_creatures = hero_data["army_creatures"]
        old_army = hero_data["old_army"]

        new_value = int(army_value * (1 + correction_factor))
        print("Hero %s wins=%s (mean=%d): army value adjustment %d -> %d (%.2f%%)" % (
            hero_name,
            hero_wins,
            mean_wins,
            army_value,
            new_value,
            correction_factor * 100,
        ))

        new_army = None
        for r in range(1, 10):
            try:
                new_army = build_army_with_retry(new_value, errmax, creatures=army_creatures)
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
                f"mean_wins={mean_wins:.2f}, stddev_wins={stddev_wins:.2f} ({stddev_wins_frac*100:.2f}%), "
                f"mean_army_value={mean_army_value:.2f}, stddev_army_value={stddev_army_value:.2f} ({stddev_army_value_frac*100:.2f}%), "
                f"max_correction={clip:.2f}\n"
            )

            f.write(report)
            print(report)

        backup(path)
        save(path, header, objects, surface_terrain)
