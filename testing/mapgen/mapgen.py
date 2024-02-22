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


ARMY_VALUE_MAX = 500_000
ARMY_VALUE_MIN = 10_000


class StackTooBigError(Exception):
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
    retry_limit = 10
    retries = 0

    while True:
        try:
            return build_army(*args, **kwargs)
        except StackTooBigError as e:
            if retries < retry_limit:
                print("Rebuilding army due to: %s" % e.str())
                retries += 1
            else:
                raise Exception("Retry limit (%d) hit" % retry_limit)


def build_army(all_creatures, value):
    per_stack = value / 7
    army_creatures = random.sample(all_creatures, 7)

    army = [None] * 7
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

    print("Leftover credit: %d" % credit)
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
    dest = os.path.relpath(f"{current_dir}/../maps/generated/{header['name']}.vmap", os.getcwd())

    print("Creating %s" % dest)
    with open(dest, 'wb') as f:
        f.write(memory_zip.getvalue())


if __name__ == "__main__":
    header0, objects0, surface_terrain0 = get_templates()
    all_creatures = get_all_creatures()

    for i in range(1, 100):
        mult = 10_000

        value = mult * random.randint(ARMY_VALUE_MIN / mult, ARMY_VALUE_MAX / mult)
        army_a = build_army_with_retry(all_creatures, value)
        army_b = build_army_with_retry(all_creatures, value)

        header = copy.deepcopy(header0)
        header["name"] = "B%02d" % i
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
