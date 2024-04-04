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
# there will be occasional error messages, but a loop here takes care for that.


import os
import zipfile
import io
import json
import sys
import shutil
import random
import numpy as np
from math import log

from mapgen import (
    get_all_creatures,
    build_army_with_retry,
    MaxAttemptsExceeded
)

# Max value for (unused_credits / target_value)
ARMY_VALUE_ERROR_MAX = 0.03

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
    for i in reversed(range(1, 3)):
        if os.path.exists(f"{path}.{i}"):
            shutil.move(f"{path}.{i}", f"{path}.{i+1}")
    shutil.move(path, f"{path}.1")


if __name__ == "__main__":
    j = None
    if (sys.argv[1] == "-"):
        print("Waiting for stdin...")
        try:
            j = input()
            print("Got json: %s" % j)
            j = json.loads(j)
        except EOFError:
            pass
    else:
        j = json.loads(sys.argv[1])

    if j is None:
        print("Failed to read stdin")
        sys.exit(0)

    path, (header, objects, surface_terrain) = load(j["map"])
    creatures_dict = {vcminame: (name, value) for (vcminame, name, value) in get_all_creatures()}

    winlist = list(j["wins"].values())
    mean_wins = np.mean(winlist)
    stddev = np.std(winlist)
    stddev_frac = stddev / mean_wins
    clip = ARMY_VALUE_CORRECTION_CLIP or (stddev_frac / 5)

    print("Stats:\nmean_wins: %d\nstddev: %d (%.2f%%)\nCorrection clip: %.2f" % (
        mean_wins,
        stddev,
        stddev_frac * 100,
        clip
    ))

    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg in ["-a", "--analyze"]:
                sys.exit(0)
            elif arg in ["-c", "--change"]:
                ALLOW_ARMY_COMP_CHANGE = True
            else:
                print("Unrecognized argument: %s" % arg)
                sys.exit(1)

    changed = False

    for (hero_name, hero_wins) in j["wins"].items():
        correction_factor = (log(mean_wins) / log(hero_wins or 2))**1
        correction_factor = np.clip(correction_factor, 1-clip, 1+clip)

        if abs(1 - correction_factor) <= ARMY_VALUE_ERROR_MAX:
            # nothing to correct
            continue

        army = [a for a in objects[hero_name]["options"]["army"] if a]
        army_value = 0
        army_creatures = []
        old_army = []

        for stack in army:
            cr_vcminame = stack["type"].removeprefix("core:")
            cr_name, cr_value = creatures_dict[cr_vcminame]
            army_value += stack["amount"] * cr_value
            army_creatures.append((cr_vcminame, cr_name, cr_value))
            old_army.append((cr_vcminame, None, stack["amount"]))

        new_value = int(army_value * correction_factor)
        print("Adjusting army value of %s: %d -> %d (%.2f%%)" % (
            hero_name,
            army_value,
            new_value,
            correction_factor * 100,
        ))

        new_army = None
        for r in range(1, 10):
            try:
                new_army = build_army_with_retry(new_value, ARMY_VALUE_ERROR_MAX, creatures=army_creatures)
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

    if not changed:
        print("Nothing to do.")
        sys.exit(1)
    else:
        backup(path)
        save(path, header, objects, surface_terrain)
