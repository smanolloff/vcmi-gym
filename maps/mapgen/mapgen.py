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
import copy
import io
import zipfile

# relative to script dir
MAP_DIR = "../gym/generated/mirror_3stack"

# id of maps to generate (inclusive)
MAP_ID_START = 1
MAP_ID_END = 8

# name template containing a single integer to be replaced with MAP_ID
MAP_NAME_TEMPLATE = "M%02d"

ARMY_N_STACKS_SAME = True  # same for both sides
ARMY_N_STACKS_MAX = 3
ARMY_N_STACKS_MIN = 3
ARMY_VALUE_SAME = True  # same for both sides
ARMY_VALUE_MAX = 30_000
ARMY_VALUE_MIN = 5_000

# Rounding for values for better descirptions
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
    with open(os.path.join(current_dir, "templates", "header.json"), "r") as f:
        header = json.load(f)

    objects = None
    with open(os.path.join(current_dir, "templates", "objects.json"), "r") as f:
        objects = json.load(f)

    surface_terrain = None
    with open(os.path.join(current_dir, "templates", "surface_terrain.json"), "r") as f:
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
    header0, objects0, surface_terrain0 = get_templates()
    all_creatures = get_all_creatures()
    all_values = list(range(ARMY_VALUE_MIN, ARMY_VALUE_MAX, ARMY_VALUE_ROUND))

    for i in range(MAP_ID_START, MAP_ID_END + 1):
        value = random.choice(all_values)
        stacks_a = random.randint(ARMY_N_STACKS_MIN, ARMY_N_STACKS_MAX)
        stacks_b = stacks_a if ARMY_N_STACKS_SAME else random.randint(ARMY_N_STACKS_MIN, ARMY_N_STACKS_MAX)
        army_a = build_army_with_retry(all_creatures, value, stacks_a)
        army_b = army_a if ARMY_VALUE_SAME else build_army_with_retry(all_creatures, value, stacks_b)

        header = copy.deepcopy(header0)
        header["name"] = MAP_NAME_TEMPLATE % i
        header["description"] = "AI test map %s\n\nTarget army values: %d\nAttacker:\n%s\n\nDefender:\n%s" % (
            header["name"],
            value,
            describe(army_a),
            describe(army_b)
        )

        objects = copy.deepcopy(objects0)
        for (i, (vcminame, _, number)) in enumerate(army_a):
            if army_a[i] is None:
                continue
            objects["hero_0"]["options"]["army"][i] = dict(amount=number, type=f"core:{vcminame}")

        for (i, (vcminame, _, number)) in enumerate(army_b):
            if army_b[i] is None:
                continue
            objects["hero_1"]["options"]["army"][i] = dict(amount=number, type=f"core:{vcminame}")

        save(header, objects, surface_terrain0)
