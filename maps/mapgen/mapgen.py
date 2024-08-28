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

# Generate up to 4096 heroes with on a 72x72 map.
# Heroes may optionally be grouped into pools of different configurations.
#
# Usage: python -m maps.mapgen.mapgen -h
#


import json
import os
import sys
import random
import io
import re
import zipfile
import argparse
from dataclasses import dataclass, field

from . import creatures_core

# XXX: the ONLY_SHOOTERS rate will be a bit higher (e.g. 15% instead of 5%)
#      due to armies with 1 stack happening to select a shooter anyway
#      ATLEAST_ONE_SHOOTER check is not performed on 1-stack armies to prevent
#      altering the only-shooter chance.
#
# Chance to select n_stacks=1 is 14.3%
# Chance to select a shooter stack is 19% (27 of 141)
# => chance to naturally select a shooter_only 1-stack army is 2.7%
#
# => real chance is CHANCE_ONLY_SHOOTERS + 3%
#
TOTAL_HEROES = 4096                 # Must be divisible by 8; max is 4096
MAX_ALLOWED_ERROR = 0.05            # Max value for (unused_credits / target_value).
STACK_WEIGHT_MAXDEV = 0.4           # Deviation (0..1); 0 means weight=avgweight, 0.5 means weight = (0.5*avgweight ... 1.5*avgweight)
CHANCE_ATLEAST_ONE_SHOOTER = 0.5    # Chance to replace 1 stack with a shooter in melee armies
CHANCE_ONLY_SHOOTERS = 0.07         # Additional chance to have only shooter stacks (natural is 0.03)
STACK_QTY_MAX = 1000                # Can be exceeded during rebalance
STACKS_MIN = 1
STACKS_MAX = 7

# k=name, v=target_value
POOLS = {
    "10k": 10_000,
    "50k": 50_000,
    "150k": 150_000,
    "500k": 500_000
}


class UnbuildableArmy(Exception):
    def __init__(self, army_config, stacks, msg):
        self.army_config = army_config
        self.stacks = stacks
        super().__init__(msg)


class NoSlotsLeft(UnbuildableArmy):
    pass


class CreditsExceeded(UnbuildableArmy):
    pass


@dataclass(frozen=True)
class Creature:
    id: int
    vcminame: str
    name: str
    value: int
    power: int
    shooter: bool

    def render(self):
        return f"{self.name}"


@dataclass
class Stack:
    creature: Creature
    qty: int
    target_value: int  # for this stack

    def value(self):
        return self.qty * self.creature.value

    def render(self):
        return f'{self.qty} "{self.creature.render()}"'


@dataclass(frozen=True)
class PoolConfig:
    name: str  # unique identifier, must match /^[0-9A-Za-z]$/
    target_value: int
    size: int
    creatures: list[Creature]
    max_allowed_error: float
    stack_weight_maxdev: float
    chance_only_shooters: float
    chance_atleast_one_shooter: float
    stack_qty_max: int
    stacks_min: int
    stacks_max: int
    verbose: bool = False

    _melee_creatures: list[Creature] = field(init=False)
    _ranged_creatures: list[Creature] = field(init=False)

    def __post_init__(self):
        assert self.stack_weight_maxdev > 0.2
        assert self.stack_weight_maxdev < 0.8
        super().__setattr__('_melee_creatures', [c for c in self.creatures if not c.shooter])
        super().__setattr__('_ranged_creatures', [c for c in self.creatures if c.shooter])

    def max_allowed_error_value(self):
        return int(self.max_allowed_error * self.target_value)

    def weakest_value_max(self):
        return 2 * self.max_allowed_error_value()

    @staticmethod
    def generate_weights(n, maxdev):
        avg_stack_weight = 1 / n
        min_stack_weight = avg_stack_weight - maxdev * avg_stack_weight

        if min_stack_weight * n > 1:
            raise ValueError(f"cfg.min_stack_weight ({min_stack_weight:.2f}) is too large for {n} stacks")
        # Allocate the minimum value to each element
        floats = [min_stack_weight] * n
        remaining_sum = 1 - sum(floats)
        # Generate random values and normalize
        random_values = [random.random() for _ in range(n)]
        total_random = sum(random_values)
        res = [min_stack_weight + (x / total_random) * remaining_sum for x in random_values]
        return res

    def generate_creatures(self, n):
        creatures = []
        while not any(c.value < self.weakest_value_max() for c in creatures):
            creatures = random.choices(self.creatures, k=n)

            if random.random() < self.chance_only_shooters:
                creatures = random.choices(self._ranged_creatures, k=n)
            elif len(creatures) > 1 and random.random() < self.chance_atleast_one_shooter:
                if not any(c in self._ranged_creatures for c in creatures):
                    creatures[0] = random.choice(self._ranged_creatures)

        # Sort by value (desc)
        return sorted(creatures, key=lambda c: -c.value)

    def generate_army_config(self):
        n_stacks = random.randint(self.stacks_min, self.stacks_max)

        return ArmyConfig(
            pool_config=self,
            target_value=self.target_value,
            creatures=self.generate_creatures(n_stacks),
            weights=self.__class__.generate_weights(n_stacks, self.stack_weight_maxdev),
            verbose=self.verbose
        )


@dataclass(frozen=True)
class ArmyConfig:
    pool_config: PoolConfig
    target_value: int
    creatures: list[Creature]
    weights: list[float]
    verbose: bool = False

    _credit: int = field(init=False)

    def __post_init__(self):
        assert len(self.weights) == len(self.creatures), f"{self.weights} / {self.creatures}"
        assert round(sum(self.weights), 3) == 1

        # https://stackoverflow.com/a/54119384
        super().__setattr__('_credit', self.target_value)

    def generate_army(self):
        stacks = [
            Stack(creature=c, qty=0, target_value=int(w*self.target_value))
            for (c, w) in zip(self.creatures, self.weights)
        ]

        # up to 7 value may be lost due to rounding => add to last stack
        rounding_error = self.target_value - sum([s.target_value for s in stacks])
        stacks[-1].target_value += rounding_error

        credit = self.target_value
        weakest_value = self.creatures[-1].value

        if self.verbose:
            print("Generating army with:")
            print(f"  target_value={self.target_value}")
            print(f"  max_err_value={self.pool_config.max_allowed_error_value()}")
            print(f"  weakest_value={weakest_value}")
            print(f"  weakest_value_max={self.pool_config.weakest_value_max()}")
            print(f"  weights={[round(w, 2) for w in self.weights]}")
            print("  stacks:")

        carry = 0
        for stack in stacks:
            target_value = stack.target_value + carry
            stack.qty = int(target_value / stack.creature.value)
            stack.qty = max(stack.qty, 1)
            stack.qty = min(stack.qty, self.pool_config.stack_qty_max)
            credit -= stack.value()
            carry = target_value - stack.value()

            if self.verbose:
                print("    %4s %-25s %-13s %-23s %-14s %s" % (
                    stack.qty,
                    f"'{stack.creature.name}' ({stack.creature.value})",
                    f"value={stack.value()}",
                    f"target={stack.target_value} → {target_value}",
                    f"credit={credit}",
                    f"carry={carry}",
                ))

        if credit > 0:
            # If remaining credit POSITIVE => 2 cases
            if credit < weakest_value:
                # Happy path (normal army generation).
                # +1 of the weakest creature may improve the acccuracy here.
                #
                # Example 1.1:
                #   target_value: 1000
                #   creatures: Gorgon(890), Goblin(60)
                #   army: 950 = 890(1xGorgon) + 60(1xGoblin)
                #   credit: 50
                #   => +1 Goblin => credit = -10 => better
                #
                if abs(credit - weakest_value) < credit:
                    stacks[-1].qty += 1
                    credit -= stacks[-1].creature.value
            else:
                # credit > weakest_value
                #
                # Caused by the qty_max limitation.
                # Splitting the weakest army into 2 (or more) will help here.
                #
                # Example 2.1:
                #   target_value: 100K
                #   creatures: Pikemen(80)
                #   army: 80K (1000xPikemen) - max qty
                #   credit: 20K
                #   => creatures: Pikemen(80), Pikemen(80)
                #   army: 80K (1000xPikemen) + 20K (250xPikemen)
                #
                while credit > weakest_value:
                    assert stacks[-1].qty == self.pool_config.stack_qty_max, f"{stacks[-1].qty} == {self.pool_config.stack_qty_max}"
                    c = self.creatures[-1]
                    s = stacks[-1]

                    # Total qty and value needed (exceeding the max_qty limit)
                    qty = s.qty + round(credit / c.value)
                    assert qty > self.pool_config.stack_qty_max

                    # XXX: if needed qty
                    qty1 = self.pool_config.stack_qty_max
                    qty2 = min(qty - qty1, self.pool_config.stack_qty_max)

                    if (len(stacks) >= 7):
                        raise NoSlotsLeft(self, stacks, f"Cannot add more stacks (needed for {qty-qty1} {s.creature.name})")

                    w1 = qty1 / qty
                    w2 = qty2 / qty
                    value1 = qty1 * c.value
                    value2 = qty2 * c.value
                    s1 = Stack(creature=c, qty=qty1, target_value=value1)
                    s2 = Stack(creature=c, qty=qty2, target_value=value2)

                    # These mutate the objects => will not violate "frozen=True"
                    self.creatures.append(c)
                    self.weights.pop()
                    self.weights.extend([w1, w2])
                    stacks.pop()
                    stacks.extend([s1, s2])
                    credit -= ((s1.value() + s2.value()) - s.value())

                    if self.verbose:
                        print("  + %4s %-25s %-13s %-23s %-14s %s" % (
                            s2.qty,
                            f"'{s2.creature.name}' ({s2.creature.value})",
                            f"value={s2.value()}",
                            f"target={s2.target_value} → {target_value}",
                            f"credit={credit}",
                            f"more={qty - (qty1 + qty2)}",
                        ))

            maxerrvalue = self.pool_config.max_allowed_error_value()
            assert abs(credit) < maxerrvalue, f"abs({credit}) < {maxerrvalue}"
            super().__setattr__('_credit', credit)
        elif credit < 0:
            # Remaining credit NEGATIVE => 2 cases
            #
            # Means a stack exceeded its target_value due to min_qty=1 requirement
            # All subsequent stacks already had negative credit => qty=1 also
            #
            # Example: target=11k
            # Creatures: Angel(5k), Devil(3k), Champion(1k), Pikeman(80)
            # Army: 2xAngel + 1xDevil + 1xChampion + 1xPikeman = 14080
            # => credit = -3080
            # It's too complicated to try and "fix" this
            # Just fail and generate a new army config instead
            raise CreditsExceeded(self, stacks, "Failed to build army with:\n\ttarget=%d,\n\tweights=%s\n\tcreatures=%s" % (
                self.target_value,
                [round(w, 2) for w in self.weights],
                [f"{c.name} ({c.value})" for c in self.creatures]
            ))
        else:
            # 0 credit remaining - perfect generation
            pass

        return Army(config=self, stacks=stacks)


@dataclass(frozen=True)
class Pool:
    config: PoolConfig
    armies: list["Army"] = field(init=False, default_factory=list)

    def __post_init__(self):
        for _ in range(self.config.size):
            self.armies.append(self.generate_army())

    def generate_army(self):
        for i in range(100):
            try:
                return self.config.generate_army_config().generate_army()
            except UnbuildableArmy as e:
                if self.config.verbose:
                    print(f"({i}) {str(e)}")


@dataclass
class Army:
    config: ArmyConfig
    stacks: list[Stack]

    def value(self):
        return sum(s.value() for s in self.stacks)

    def to_json(self):
        res = [{} for i in range(7)]
        for (slot, stack) in enumerate(self.stacks):
            res[slot] = dict(amount=stack.qty, type=f"core:{stack.creature.vcminame}")
        random.shuffle(res)
        return res


def get_templates():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_dir, "templates", "4096", "header.json"), "r") as f:
        header = json.load(f)

    with open(os.path.join(current_dir, "templates", "4096", "objects.json"), "r") as f:
        objects = json.load(f)

    with open(os.path.join(current_dir, "templates", "4096", "surface_terrain.json"), "r") as f:
        surface_terrain = json.load(f)

    return header, objects, surface_terrain


def build_pool_configs(verbose=False):
    res = []
    all_creatures = [Creature(**creature) for creature in creatures_core.CREATURES]
    pool_size = TOTAL_HEROES // len(POOLS)
    for name, value in POOLS.items():
        res.append(PoolConfig(
            name=name,
            target_value=value,
            size=pool_size,
            creatures=all_creatures,
            max_allowed_error=MAX_ALLOWED_ERROR,
            stack_weight_maxdev=STACK_WEIGHT_MAXDEV,
            chance_only_shooters=CHANCE_ONLY_SHOOTERS,
            chance_atleast_one_shooter=CHANCE_ATLEAST_ONE_SHOOTER,
            stack_qty_max=STACK_QTY_MAX,
            stacks_min=STACKS_MIN,
            stacks_max=STACKS_MAX,
            verbose=verbose
        ))

    return res


def save(dest, header, objects, surface_terrain):
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zipf:
        zipf.writestr('header.json', json.dumps(header))
        zipf.writestr('objects.json', json.dumps(objects))
        zipf.writestr('surface_terrain.json', json.dumps(surface_terrain))

    print("Creating %s" % dest)
    with open(dest, 'wb') as f:
        f.write(memory_zip.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help="be more verbose", action="store_true")
    parser.add_argument('--force', help="overwrite map if already exists", action="store_true")
    parser.add_argument('--dry-run', help="don't save anything", action="store_true")
    parser.add_argument('--seed', help="random seed", type=int)
    parser.add_argument('map', metavar="<map>", type=str, help="path to map")
    args = parser.parse_args()

    if os.path.exists(args.map) and not args.force:
        print("map already exists: %s" % args.map)
        sys.exit(1)

    directory = os.path.dirname(args.map)
    if directory and not os.path.isdir(directory):
        print("not a directory: %s" % directory)

    seed = args.seed or random.randint(0, 2**31)
    random.seed(seed)

    header, objects, surface_terrain0 = get_templates()

    # basic checks
    assert 8 == len([k for k in objects if k.startswith("town_")]), "expected 8 towns"
    assert args.map.endswith(".vmap"), "map name must end with .vmap"

    for name in POOLS.keys():
        assert re.match(r"^[0-9A-Za-z]+$", name), f"invalid pool name: {name}"

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

    print(f"*** Generating {os.path.basename(args.map.removesuffix('.vmap'))}")
    header["name"] = args.map
    header["description"] = f"AI test map {header['name']}"
    header["description"] += f"\nNumber of heroes: {TOTAL_HEROES}"
    header["description"] += f"\nNumber of pools: {len(POOLS)}"

    colornames = ["red", "blue", "tan", "green", "orange", "purple", "teal", "pink"]
    pools = [Pool(config=pc) for pc in build_pool_configs()]

    def army_iterator():
        oid = 0
        for pool in pools:
            for army in pool.armies:
                yield oid, army
                oid += 1

    it = army_iterator()

    for y in range(2, 66):  # 64 rows
        for x in range(5, 69):  # 64 columns (heroes are 2 tiles for some reason)
            oid, army = it.__next__()

            # XXX: hero ID must be GLOBALLY unique and incremental (0...N_HEROES)
            #      hero name MUST start with "hero_<ID>" (may be followed by non-numbers)
            #      Example: "hero_1234", "hero_513_kur"
            #      VCMI stats collection uses the numeric ID as a unique DB index
            hero_name = f"hero_{oid}_pool_{army.config.pool_config.name}"

            assert re.match(r"^hero_\d+_pool_[0-9A-Za-z]+$", hero_name), f"invalid hero name: {hero_name}"

            hero_army = [{} for i in range(7)]
            for (slot, stack) in enumerate(army.stacks):
                hero_army[slot] = dict(amount=stack.qty, type=f"core:{stack.creature.vcminame}")

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

    total_n_armies = 0
    total_n_only_shooter_armies = 0
    total_n_atleast1_shooter_armies = 0
    for pool in pools:
        print("Pool %s" % pool.config.name)
        n_armies = 0
        n_only_shooter_armies = 0
        n_atleast1_shooter_armies = 0
        for army in pool.armies:
            n_armies += 1
            if all(s.creature.shooter for s in army.stacks):
                n_only_shooter_armies += 1
            if any(s.creature.shooter for s in army.stacks):
                n_atleast1_shooter_armies += 1
        total_n_armies += n_armies
        total_n_only_shooter_armies += n_only_shooter_armies
        total_n_atleast1_shooter_armies += n_atleast1_shooter_armies

        print("  Armies with shooters only:       %d (%.2f%%)" % (n_only_shooter_armies, 100 * n_only_shooter_armies / n_armies))
        print("  Armies with at least 1 shooter:  %d (%.2f%%)" % (n_atleast1_shooter_armies, 100 * n_atleast1_shooter_armies / n_armies))
        print("  Armies total:                    %d" % n_armies)

    print("Total:")
    print("  Armies with shooters only:       %d (%.2f%%)" % (total_n_only_shooter_armies, 100 * total_n_only_shooter_armies / total_n_armies))
    print("  Armies with at least 1 shooter:  %d (%.2f%%)" % (total_n_atleast1_shooter_armies, 100 * total_n_atleast1_shooter_armies / total_n_armies))
    print("  Armies total:                    %d" % total_n_armies)
    print("Seed: %s" % seed)

    if args.dry_run:
        print("Nothing to do (--dry-run)")
    else:
        save(args.map, header, objects, surface_terrain0)


if __name__ == "__main__":
    main()
