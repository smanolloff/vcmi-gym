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

# Generate 4 pools of 1024 armies each on a 72x72 map template
# (total 4096 armies). Each pool has army different target army strength.

import json
import os
import random
import io
import re
import zipfile
import collections
from typing import NamedTuple

import creatures_core


# relative to script dir
MAP_DIR = "../gym/generated/4096"
MAP_NAME = "4x1024"

# A note regarding `weakest_creature_value_max`:
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
# XXX: pay attention to target army value and STACK_QTY_MAX
#      Example:
#       target army value: 100K
#       STACK_QTY_MAX: 1023
#       ARMY_WEAKEST_CREATURE_VALUE_MAX: 100
#       => the only allowed 1-stack armies will be:
#           a) 1000 centaurs (value=100)
#           b) 1020 walking dead (value=98)
#


ALL_SHOOTER_CREATURES = [(c["vcminame"], c["name"], c["value"]) for c in creatures_core.ALL_CREATURES if c["shooter"]]
ALL_MELEE_CREATURES = [(c["vcminame"], c["name"], c["value"]) for c in creatures_core.ALL_CREATURES if not c["shooter"]]
ALL_CREATURES = ALL_SHOOTER_CREATURES + ALL_MELEE_CREATURES

# debug counters for logging purposes
N_ONLY_SHOOTER_ARMIES = 0
N_ATLEAST1_SHOOTER_ARMIES = 0


class ArmyConfig(NamedTuple):
    id: str  # unique identifier, must match /^[0-9A-Za-z]$/
    target_value: int
    weakest_creature_value_max: int  # see note above
    max_allowed_error: float = 0.05  # max value for (unused_credits / target_value). None = auto
    chance_only_shooters: float = 0.05  # include only shooters in an army
    chance_atleast_one_shooter: float = 0.5  # include at least 1 shooter in an army
    stack_qty_max: int = 1024
    stacks_min: int = 1
    stacks_max: int = 7
    stacks_enforce: bool = False  # fail if not all stacks are filled

    def __post_init__(self):
        """
        Maximum allowed error when generating the army value is ~500
        (given peasant=15, imp=50, ogre=416, minotaur=835, etc.)
        For 5K armies, this is 0.1
        """
        if self.max_allowed_error is None:
            self.max_allowed_error = 500 / self.target_value


POOLS = [
    ArmyConfig(id="10k", target_value=10_000, weakest_creature_value_max=100),
    ArmyConfig(id="50k", target_value=50_000, weakest_creature_value_max=500),
    ArmyConfig(id="150k", target_value=150_000, weakest_creature_value_max=1000),
    ArmyConfig(id="500k", target_value=500_000, weakest_creature_value_max=2000)
]


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


def get_templates():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_dir, "templates", "4096", "header.json"), "r") as f:
        header = json.load(f)

    with open(os.path.join(current_dir, "templates", "4096", "objects.json"), "r") as f:
        objects = json.load(f)

    with open(os.path.join(current_dir, "templates", "4096", "surface_terrain.json"), "r") as f:
        surface_terrain = json.load(f)

    return header, objects, surface_terrain


def build_army_with_retry(*args, verbose, **kwargs):
    max_attempts = 1000
    counters = collections.defaultdict(lambda: 0)
    for r in range(1, max_attempts + 1):
        try:
            return build_army(*args, **dict(kwargs, verbose=verbose))
        except (StackTooBigError, UnusedCreditError, NotAllStacksFilled, WeakestCreatureTooStrongError) as e:
            counters[e.__class__.__name__] += 1
            # print("[%d] Rebuilding army due to: %s" % (r, str(e)))
        else:
            if r > 1 and verbose:
                print("Made %d army rebuilds due to: %s" % (r, dict(counters)))
    raise MaxAttemptsExceeded("Max attempts (%d) exceeded. Errors: %s" % (max_attempts, dict(counters)))


def build_army(cfg, creatures=None, n_stacks=None, verbose=True, print_creatures=True):
    if creatures is None:
        assert n_stacks, "when creatures is None, n_stacks is required"
        creatures = random.choices(ALL_CREATURES, k=n_stacks)

        if random.random() < cfg.chance_only_shooters:
            creatures = random.choices(ALL_SHOOTER_CREATURES, k=n_stacks)
        elif random.random() < cfg.chance_atleast_one_shooter:
            if not any(c in ALL_SHOOTER_CREATURES for c in creatures):
                creatures[0] = random.choice(ALL_SHOOTER_CREATURES)
    else:
        assert n_stacks is None, "when creatures is given, n_stacks must be None"

    army = [None] * len(creatures)
    per_stack = cfg.target_value / len(creatures)
    credit = cfg.target_value
    weakest = 100_000  # azure dragon is 80k
    filled_creatures = {name: 0 for (_, name, _) in creatures}

    for (i, (vcminame, name, aivalue)) in enumerate(creatures):
        number = int(per_stack / aivalue)
        if number == 0:
            continue
        elif number > cfg.stack_qty_max:
            # raise StackTooBigError("Stack too big: %s: %d" % (name, number))
            number = cfg.stack_qty_max
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

            to_add = min(number, cfg.stack_qty_max - number0)
            if to_add == 0:
                # raise StackTooBigError("Stack too big: %s: %d" % (name, number0 + number))
                continue

            assert to_add > 0

            credit -= to_add * aivalue
            army[i] = (vcminame, name, number0 + to_add)
            weakest = min(weakest, aivalue)
            filled_creatures[name] = number0 + to_add

    if weakest > cfg.weakest_creature_value_max:
        raise WeakestCreatureTooStrongError("Weakest creature has value %d > %d" % (weakest, cfg.weakest_creature_value_max))

    real_value = cfg.target_value - credit
    real_value = real_value or 1  # fix zero div error theres no army
    error = 1 - cfg.target_value/real_value

    if verbose:
        print("  value (new):\t\t%d (%s%.2f%%)\n  max allowed error:\t%.2f%%\n  stacks:\t\t%d" % (
            real_value,
            "+" if error > 0 else "",
            error*100,
            cfg.max_allowed_error*100,
            len(creatures)
        ))
        if print_creatures:
            print("  creatures:\t\t%s" % ", ".join([f"{number} \"{name}\"" for (name, number) in sorted(filled_creatures.items(), key=lambda x: x[0])]))

    if abs(error) > cfg.max_allowed_error:
        # raise UnusedCreditError("Too much unused credit: %d (target value: %d)" % (credit, value))
        # print("Too much unused credit: %d, will add 1 more unit" % credit)

        # Try adding 1 of the weakest creature to see if it gets us closer
        # (this will cause negative remaining credit => use abs)
        i, (vcminame, name, aivalue) = min(enumerate(creatures), key=lambda x: x[1][2])
        newnumber = 1 + (army[i][2] if army[i] else 0)
        real_value += aivalue
        error = 1 - cfg.target_value/real_value
        if verbose:
            print("  * added 1 '%s': army value: %d of %d (%.2f%%)" % (name, real_value, cfg.target_value, error*100))
        if abs(error) > cfg.max_allowed_error:
            raise UnusedCreditError(f"Could not reach target value of {cfg.target_value}")
        elif newnumber > cfg.stack_qty_max:
            raise StackTooBigError("Stack too big: %s: %d" % (name, newnumber))

        army[i] = (vcminame, name, newnumber)

    if any(s is None for s in army):
        raise NotAllStacksFilled("Not all stacks were filled")

    if any(c in ALL_SHOOTER_CREATURES for c in creatures):
        global N_ATLEAST1_SHOOTER_ARMIES
        N_ATLEAST1_SHOOTER_ARMIES += 1
        if all(c in ALL_SHOOTER_CREATURES for c in creatures):
            global N_ONLY_SHOOTER_ARMIES
            N_ONLY_SHOOTER_ARMIES += 1

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
    header, objects, surface_terrain0 = get_templates()

    # basic checks
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mappath = os.path.relpath(f"{current_dir}/{MAP_DIR}/{MAP_NAME}.vmap", os.getcwd())
    assert not os.path.exists(mappath), f"destination '{mappath}' already exists"
    assert 8 == len([k for k in objects if k.startswith("town_")]), "expected 8 towns"

    ids = [g.id for g in POOLS]
    assert len(ids) == len(set(ids)), f"pool IDs are not unique: {ids}"
    for id in ids:
        assert re.match(r"^[0-9A-Za-z]+$", id), f"invalid pool ID: {id}"

    # this is not a problem. There will simply be a few less heroes generated
    # (e.g. 4095 instead of 4096)
    n_heroes = (4096 // len(POOLS)) * len(POOLS)

    # add forts for the 8 towns
    fortnames = ["core:fort", "core:citadel", "core:castle"]
    fortlvls = [3] * 4  # 4 castles
    fortlvls += [2] * 2  # 2 citadels
    fortlvls += [1] * 1  # 1 fort
    fortlvls += [0] * 1  # 1 village (still useful: prevents regular obstacles)
    for i, fortlvl in enumerate(fortlvls):
        # mapeditor generates them in reversed order
        buildings = list(reversed(fortnames[:fortlvl]))
        objects[f"town_{i}"]["options"]["buildings"]["allOf"] = buildings

    print(f"*** Generating map #{MAP_NAME}")
    header["name"] = MAP_NAME
    header["description"] = f"AI test map {header['name']}"
    header["description"] += f"\nNumber of heroes: {n_heroes}"
    header["description"] += f"\nNumber of pools: {len(POOLS)}"

    for cfg in POOLS:
        header["description"] += f'\nHero pool "{cfg.id}":'
        header["description"] += f"\n  Target army values: {[cfg.target_value]}"
        header["description"] += f"\n  Number of stacks: min={[cfg.stacks_min]}, max={[cfg.stacks_max]}"
        header["description"] += f"\n  Max stack size: {cfg.stack_qty_max}"
        header["description"] += f"\n  Weakest creature type max value: {cfg.weakest_creature_value_max}"
        header["description"] += f"\n  Chance for only shooters in army: {cfg.chance_only_shooters}"

    oid = 0
    colornames = ["red", "blue", "tan", "green", "orange", "purple", "teal", "pink"]

    # XXX: All hero IDs within a pool must be SEQUENTIAL
    #      e.g. pool "10k": 0,1,2,3
    #           pool "50k": 5,6,7,8
    #           ...etc
    def config_iterator(total, pools):
        per_pool = total // len(pools)
        for i, pool in enumerate(pools):
            for j in range(per_pool):
                yield (pool, i*per_pool + j)

    it = config_iterator(4096, POOLS)

    for y in range(2, 66):  # 64 rows
        for x in range(5, 69):  # 64 columns (heroes are 2 tiles for some reason)
            cfg, oid = it.__next__()

            # XXX: hero ID must be GLOBALLY unique and incremental (0...N_HEROES)
            #      hero name MUST start with "hero_<ID>" (may be followed by non-numbers)
            #      Example: "hero_1234", "hero_513_kur"
            #      VCMI stats collection uses the numeric ID as a unique DB index
            hero_name = f"hero_{oid}_pool_{cfg.id}"

            assert re.match(r"^hero_\d+_pool_[0-9A-Za-z]+$", hero_name), f"invalid hero name: {hero_name}"

            print(f"* Generating army for {hero_name}")
            n_stacks = random.randint(cfg.stacks_min, cfg.stacks_max)

            army = build_army_with_retry(cfg, n_stacks=n_stacks, verbose=True)

            hero_army = [{} for i in range(7)]
            for (slot, (vcminame, _, number)) in enumerate(army):
                hero_army[slot] = dict(amount=number, type=f"core:{vcminame}")

            random.shuffle(hero_army)
            values = dict(
                color=colornames[(x-2) % 8],
                name=hero_name,
                type=f"ml:hero_{oid}",
                animation="AH01",
                id=oid,
                x=x,
                y=y
            )
            color = values["color"]
            header["players"][color]["heroes"][values["name"]] = dict(type=values["type"])

            primary_skills = {"knowledge": 20}  # no effect due VCMI mana randomization
            primary_skills["spellpower"] = random.randint(5, 15)

            secondary_skills = []
            skilllevels = ["basic", "advanced", "expert"]

            if random.random() < 0.2:
                secondary_skills.append({"skill": "core:ballistics", "level": skilllevels[random.randint(0, 2)]})

            if random.random() < 20:
                secondary_skills.append({"skill": "core:artillery", "level": skilllevels[random.randint(0, 2)]})

            spell_book = [
                "preset",
                "core:fireElemental",
                "core:earthElemental",
                "core:waterElemental",
                "core:airElemental"
            ]

            objects[values["name"]] = dict(
                type="hero", subtype="core:cleric", x=x, y=y, l=0,
                options=dict(
                    experience=10000000,
                    name=values["name"],
                    # formation="wide",
                    # gender=1,
                    owner=values["color"],
                    # portrait=f"core:{values['name']}",
                    type=values["type"],
                    army=hero_army,
                    primarySkills=primary_skills,
                    secondarySkills=secondary_skills,
                    spellBook=spell_book
                ),
                template=dict(
                    animation=f"{values['animation']}_",
                    editorAnimation=f"{values['animation']}_E",
                    mask=["VVV", "VAV"],
                    visitableFrom=["+++", "+-+", "+++"],
                )
            )

            oid += 1

    # for y in range(5, 66, 3):
    #     for x in range(8, 72, 5):
    #         values = dict(id=oid, x=x, y=y)
    #         objects[f"cursedGround_{oid}"] = dict(
    #             type="cursedGround", x=x, y=y, l=0,
    #             subtype="object",
    #             template=dict(
    #                 animation="AVXcrsd0.def",
    #                 editorAnimation="",
    #                 mask=["VVVVVV", "VVVVVV", "VVVVVV", "VVVVVV"],
    #                 zIndex=100
    #             )
    #         )
    #         oid += 1

    print("N_ONLY_SHOOTER_ARMIES: %d\nN_ATLEAST1_SHOOTER_ARMIES: %d" % (N_ONLY_SHOOTER_ARMIES, N_ATLEAST1_SHOOTER_ARMIES))

    save(header, objects, surface_terrain0)
