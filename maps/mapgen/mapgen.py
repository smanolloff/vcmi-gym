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

import json
import os
import random
import io
import zipfile

# relative to script dir
MAP_DIR = "../gym/generated/88"
# name template containing a single integer to be replaced with MAP_ID
MAP_NAME_TEMPLATE = "88-1stack-%02d"
# id of maps to generate (inclusive)
MAP_ID_START = 1
MAP_ID_END = 8

ARMY_N_STACKS_SAME = True  # same for both sides
ARMY_N_STACKS_MAX = 1
ARMY_N_STACKS_MIN = 1
ARMY_VALUE_MAX = 100_000
ARMY_VALUE_MIN = 30_000

# Round values for better descruptions
ARMY_VALUE_ROUND = 1000

# Max value for (unused_credits / target_value)
ARMY_UNUSED_CREDIT_FRACTION_MAX = 0.1


class StackTooBigError(Exception):
    pass


class UnusedCreditError(Exception):
    pass


class NotEnoughStacksError(Exception):
    pass


def get_all_creatures():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    all_creatures = None
    with open(os.path.join(current_dir, "all_creatures.json"), "r") as f:
        all_creatures = json.load(f)

    return [(vcminame, name, value) for (vcminame, (name, value)) in all_creatures.items()]


def get_templates():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    header = None
    with open(os.path.join(current_dir, "templates", "88", "header.json"), "r") as f:
        header = json.load(f)

    objects = None
    with open(os.path.join(current_dir, "templates", "88", "objects.json"), "r") as f:
        objects = json.load(f)

    surface_terrain = None
    with open(os.path.join(current_dir, "templates", "88", "surface_terrain.json"), "r") as f:
        surface_terrain = json.load(f)

    return header, objects, surface_terrain


def build_army_with_retry(*args, **kwargs):
    retry_limit = 100
    retries = 0

    while True:
        try:
            return build_army(*args, **kwargs)
        except (StackTooBigError, UnusedCreditError) as e:
            if retries < retry_limit:
                retries += 1
                print("[%d] Rebuilding army due to: %s" % (retries, str(e)))
            else:
                raise Exception("Retry limit (%d) hit" % retry_limit)


def build_army(all_creatures, value, n_stacks):
    per_stack = value / n_stacks
    army_creatures = random.sample(all_creatures, n_stacks)

    army = [None] * n_stacks
    credit = value
    for (i, (vcminame, name, aivalue)) in enumerate(army_creatures):
        number = int(per_stack / aivalue)
        if number > 5000:
            raise StackTooBigError("Stack too big: %s: %d" % (name, number))
        credit -= number * aivalue
        army[i] = (vcminame, name, number)

    # repeat with remaining credit
    for (i, (vcminame, name, aivalue)) in enumerate(army_creatures):
        number = int(min(credit, per_stack) / aivalue)
        credit -= number * aivalue
        assert army[i] is not None
        (vcminame0, name0, number0) = army[i]
        assert vcminame0 == vcminame
        assert name0 == name
        army[i] = (vcminame, name, number0 + number)

    frac = credit / value
    print("Leftover credit: %d (%.2f%%)" % (credit, frac * 100))
    if frac > ARMY_UNUSED_CREDIT_FRACTION_MAX:
        raise UnusedCreditError("Too much unused credit: %d (target value: %d)" % (credit, value))

    if any(s is None for s in army):
        raise NotEnoughStacksError("Not all stacks were filled")

    # list of (vcminame, name, number) tuples
    return army


def describe(army):
    lines = []
    for (_, name, number) in army:
        lines.append("%-4d %s" % (number, name))

    return "\n".join(lines)


def save(header, objects, surface_terrain):
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zipf:
        zipf.writestr('header.json', json.dumps(header))
        zipf.writestr('objects.json', json.dumps(objects))
        zipf.writestr('surface_terrain.json', json.dumps(surface_terrain))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.relpath(f"{current_dir}/{MAP_DIR}/{header['name']}.vmap", os.getcwd())

    print("Creating %s" % dest)
    with open(dest, 'wb') as f:
        f.write(memory_zip.getvalue())


if __name__ == "__main__":
    all_creatures = get_all_creatures()
    all_values = list(range(ARMY_VALUE_MIN, ARMY_VALUE_MAX, ARMY_VALUE_ROUND))

    for mapid in range(MAP_ID_START, MAP_ID_END + 1):
        print(f"*** Generating map #{mapid}")
        header, objects, surface_terrain0 = get_templates()
        value = random.choice(all_values)
        stacks = random.randint(ARMY_N_STACKS_MIN, ARMY_N_STACKS_MAX)

        header["name"] = MAP_NAME_TEMPLATE % mapid
        header["description"] = "AI test map %s\nTarget army values: %d" % (header["name"], value)

        for j in range(0, 64):
            print(f"* Generating army for hero #{j}")
            stacks = stacks if ARMY_N_STACKS_SAME else random.randint(ARMY_N_STACKS_MIN, ARMY_N_STACKS_MAX)
            army = build_army_with_retry(all_creatures, value, stacks)
            objects[f"hero_{j}"]["options"]["army"] = [{} for i in range(7)]

            for (slot, (vcminame, _, number)) in enumerate(army):
                if army[slot] is None:
                    continue
                objects[f"hero_{j}"]["options"]["army"][slot] = dict(amount=number, type=f"core:{vcminame}")

        save(header, objects, surface_terrain0)
