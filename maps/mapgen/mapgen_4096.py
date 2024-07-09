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

# Generate 4096 armies on a 72x72 map template

import json
import os
import random
import io
import zipfile
import collections

from creatures_core import ALL_CREATURES

# relative to script dir
MAP_DIR = "../gym/generated/4096"
# name template containing a single {id} token to be replaced with MAP_ID
MAP_NAME_TEMPLATE = "4096-all-mixstack-100K-{id:02d}"
# id of maps to generate (inclusive)
MAP_ID_START = 1
MAP_ID_END = 1

# ARMY_N_STACKS_SAME = False  # same for both sides
ARMY_N_STACKS_MIN = 1
ARMY_N_STACKS_MAX = 7
ARMY_N_STACKS_ENFORCE = False  # whether to fail if not all stacks are filled

# XXX: these should be equal, otherwise battle will be one-sided
ARMY_VALUE_MIN = 100_000
ARMY_VALUE_MAX = 100_000

STACK_QTY_MAX = 1023

# Round values for better descriptions
ARMY_VALUE_ROUND = 1000

# Max value for abs(1 - army_value/target_value)
ARMY_VALUE_ERROR_MAX = 0.01

# Max value for a single unit of the weakest creature type in the army
# (this to allow rebalancing, i.e. avoid armies with 6+ tier units only)
#
# Some ref values:
#   peasant         = 15    (tier 1 - weakest unit)
#   pikeman         = 80    (tier 1)
#   archer          = 126   (tier 2)
#   griffin         = 351   (tier 3)
#   swordsman       = 445   (tier 4)
#   monk            = 485   (tier 5)
#   cavalier        = 1946  (tier 6)
#   angel           = 5000  (tier 7)
#   crystal dragon  = 39338 (tier 8 - strongest unit, azure dragon is removed)
#
# i.e. a value of "15" will ensure all armies contain a stack of peasants.
ARMY_WEAKEST_CREATURE_VALUE_MAX = 100

# Hero IDs are re-mapped when game starts
# => use hero experience as a reference
# HERO_EXP_REF = 10000000


class StackTooBigError(Exception):
    pass


class UnusedCreditError(Exception):
    pass


class NotAllStacksFilled(Exception):
    pass


class WeakestCreatureTooStrongError(Exception):
    pass


class MaxAttemptsExceeded(Exception):
    pass


def get_all_creatures():
    return [(c["vcminame"], c["name"], c["value"]) for c in ALL_CREATURES]


def get_templates():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_dir, "templates", "4096", "header.json"), "r") as f:
        header = json.load(f)

    with open(os.path.join(current_dir, "templates", "4096", "objects.json"), "r") as f:
        objects = json.load(f)

    with open(os.path.join(current_dir, "templates", "4096", "surface_terrain.json"), "r") as f:
        surface_terrain = json.load(f)

    with open(os.path.join(current_dir, "templates", "4096", "hero_mapping.json"), "r") as f:
        hero_mapping = json.load(f)

    return header, objects, surface_terrain, hero_mapping


def build_army_with_retry(*args, **kwargs):
    max_attempts = 1000
    counters = collections.defaultdict(lambda: 0)
    for r in range(1, max_attempts + 1):
        try:
            return build_army(*args, **kwargs)
        except (StackTooBigError, UnusedCreditError, NotAllStacksFilled, WeakestCreatureTooStrongError) as e:
            counters[e.__class__.__name__] += 1
            # print("[%d] Rebuilding army due to: %s" % (r, str(e)))
        else:
            if r > 1:
                print("Made %d army rebuilds due to: %s" % (r, dict(counters)))
    raise MaxAttemptsExceeded("Max attempts (%d) exceeded. Errors: %s" % (max_attempts, dict(counters)))


def build_army(target_value, err_frac_max, creatures=None, n_stacks=None, all_creatures=None, print_creatures=True):
    if creatures is None:
        assert all_creatures, "when creatures is None, all_creatures is required"
        assert n_stacks, "when creatures is None, n_stacks is required"
        creatures = random.sample(all_creatures, n_stacks)
    else:
        assert all_creatures is None, "when creatures is given, all_creatures must be None"
        assert n_stacks is None, "when creatures is given, n_stacks must be None"

    army = [None] * len(creatures)
    per_stack = target_value / len(creatures)
    credit = target_value
    weakest = 100_000  # azure dragon is 80k
    filled_creatures = {name: 0 for (_, name, _) in creatures}

    for (i, (vcminame, name, aivalue)) in enumerate(creatures):
        number = int(per_stack / aivalue)
        if number == 0:
            continue
        elif number > STACK_QTY_MAX:
            # raise StackTooBigError("Stack too big: %s: %d" % (name, number))
            number = STACK_QTY_MAX
        credit -= number * aivalue
        army[i] = (vcminame, name, number)
        weakest = min(weakest, aivalue)
        filled_creatures[name] = number

    # repeat with remaining credit
    for _ in range(10):
        for (i, (vcminame, name, aivalue)) in random.sample(list(enumerate(creatures)), len(creatures)):
            number = int(min(credit, per_stack) / aivalue)
            if number == 0:
                continue
            assert army[i] is not None
            (vcminame0, name0, number0) = army[i]
            assert vcminame0 == vcminame
            assert name0 == name

            to_add = min(number, STACK_QTY_MAX - number0)
            if to_add == 0:
                # raise StackTooBigError("Stack too big: %s: %d" % (name, number0 + number))
                continue

            assert to_add > 0

            credit -= to_add * aivalue
            army[i] = (vcminame, name, number0 + to_add)
            weakest = min(weakest, aivalue)
            filled_creatures[name] = number0 + to_add

    if weakest > ARMY_WEAKEST_CREATURE_VALUE_MAX:
        raise WeakestCreatureTooStrongError("Weakest creature has value %d > %d" % (weakest, ARMY_WEAKEST_CREATURE_VALUE_MAX))

    real_value = target_value - credit
    real_value = real_value or 1  # fix zero div error theres no army
    error = 1 - target_value/real_value

    print("  value (new):\t\t%d (%s%.2f%%)\n  max allowed error:\t%.2f%%\n  stacks:\t\t%d" % (
        real_value,
        "+" if error > 0 else "",
        error*100,
        err_frac_max*100,
        len(creatures)
    ))

    if print_creatures:
        print("  creatures:\t\t%s" % ", ".join([f"{number} \"{name}\"" for (name, number) in sorted(filled_creatures.items(), key=lambda x: x[0])]))

    if abs(error) > err_frac_max:
        # raise UnusedCreditError("Too much unused credit: %d (target value: %d)" % (credit, value))
        # print("Too much unused credit: %d, will add 1 more unit" % credit)

        # Try adding 1 of the weakest creature to see if it gets us closer
        # (this will cause negative remaining credit => use abs)
        i, (vcminame, name, aivalue) = min(enumerate(creatures), key=lambda x: x[1][2])
        newnumber = 1 + (army[i][2] if army[i] else 0)
        real_value += aivalue
        error = 1 - target_value/real_value
        print("  * added 1 '%s': army value: %d of %d (%.2f%%)" % (name, real_value, target_value, error*100))
        if abs(error) > err_frac_max:
            raise UnusedCreditError(f"Could not reach target value of {target_value}")
        elif newnumber > STACK_QTY_MAX:
            raise StackTooBigError("Stack too big: %s: %d" % (name, newnumber))

        army[i] = (vcminame, name, newnumber)

    if any(s is None for s in army):
        raise NotAllStacksFilled("Not all stacks were filled")

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
    all_values = list(range(ARMY_VALUE_MIN, ARMY_VALUE_MAX+1, ARMY_VALUE_ROUND))

    for mapid in range(MAP_ID_START, MAP_ID_END + 1):
        print(f"*** Generating map #{mapid}")
        header, objects, surface_terrain0, hero_mapping = get_templates()
        value = random.choice(all_values)

        header["name"] = MAP_NAME_TEMPLATE.format(id=mapid)
        header["description"] = (
            f"AI test map {header['name']}\n"
            f"Target army values: {value}\n"
            f"Stack count min/max: {ARMY_N_STACKS_MIN}/{ARMY_N_STACKS_MAX}\n"
        )
        oid = 0

        assert 8 == len([k for k in objects if k.startswith("town_")]), "expected 8 towns"

        # add forts for the 8 towns
        fortnames = ["core:fort", "core:citadel", "core:castle"]
        fortlvls = [3] * 5  # 5 castles
        fortlvls += [2] * 2  # 2 citadels
        fortlvls += [1] * 1  # 1 fort
        for i, fortlvl in enumerate(fortlvls):
            # mapeditor generates them in reversed order
            buildings = list(reversed(fortnames[:fortlvl]))
            objects[f"town_{i}"]["options"]["buildings"]["allOf"] = buildings

        for y in range(2, 66):  # 64 rows
            for x in range(5, 69):  # 64 columns (heroes are 2 tiles for some reason)
                print(f"* Generating army for hero #{oid}")
                n_stacks = random.randint(ARMY_N_STACKS_MIN, ARMY_N_STACKS_MAX)
                army = build_army_with_retry(value, ARMY_VALUE_ERROR_MAX, n_stacks=n_stacks, all_creatures=all_creatures)
                hero_army = [{} for i in range(7)]
                for (slot, (vcminame, _, number)) in enumerate(army):
                    hero_army[slot] = dict(amount=number, type=f"core:{vcminame}")

                random.shuffle(hero_army)
                values = dict(hero_mapping[(x-2) % 8], id=oid, x=x, y=y)
                color = values["color"]
                header["players"][color]["heroes"][f"hero_{oid}"] = dict(type=f"core:{values['name']}")

                primary_skills = {"knowledge": 20}
                primary_skills["spellpower"] = random.randint(5, 15)

                secondary_skills = []
                skilllevels = ["basic", "advanced", "expert"]

                ballistics_chance = 20
                ballistics_roll = random.randint(0, 100)
                if ballistics_roll < ballistics_chance:
                    secondary_skills.append({"skill": "core:ballistics", "level": skilllevels[ballistics_roll % 3]})

                artillery_chance = 20
                artillery_roll = random.randint(0, 100)
                if artillery_roll < artillery_chance:
                    secondary_skills.append({"skill": "core:artillery", "level": skilllevels[ballistics_roll % 3]})

                spell_book = [
                    "preset",
                    "core:fireElemental",
                    "core:earthElemental",
                    "core:waterElemental",
                    "core:airElemental"
                ]

                objects[f"hero_{oid}"] = dict(
                    type="hero", subtype=values["type"], x=x, y=y, l=0,
                    options=dict(
                        experience=oid,
                        name=f"hero_{oid}",
                        formation="wide",
                        gender=1,
                        owner=values["color"],
                        portrait=f"core:{values['name']}",
                        type=f"core:{values['name']}",
                        army=hero_army,
                        primarySkills=primary_skills,
                        secondarySkills=secondary_skills,
                        spellBook=spell_book
                    ),
                    template=dict(
                        animation=f"{values['animation']}_.def",
                        editorAnimation=f"{values['animation']}_E.def",
                        mask=["VVV", "VAV"],
                        visitableFrom=["+++", "+-+", "+++"],
                    )
                )

                oid += 1

        for y in range(5, 66, 3):
            for x in range(8, 72, 5):
                values = dict(id=oid, x=x, y=y)
                objects[f"cursedGround_{oid}"] = dict(
                    type="cursedGround", x=x, y=y, l=0,
                    subtype="object",
                    template=dict(
                        animation="AVXcrsd0.def",
                        editorAnimation="",
                        mask=["VVVVVV", "VVVVVV", "VVVVVV", "VVVVVV"],
                        zIndex=100
                    )
                )
                oid += 1

        save(header, objects, surface_terrain0)
