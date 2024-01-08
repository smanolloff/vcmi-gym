import json
import os
import random
import copy
import io
import zipfile


ARMY_VALUE = 300_000


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


def build_army(all_creatures):
    per_stack = ARMY_VALUE / 7
    army_creatures = random.sample(all_creatures, 7)

    army = [None] * 7
    credit = ARMY_VALUE
    for (i, (vcminame, name, aivalue)) in enumerate(army_creatures):
        number = int(per_stack / aivalue)
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

    for i in range(100, 1000):
        army_a = build_army(all_creatures)
        army_b = build_army(all_creatures)

        header = copy.deepcopy(header0)
        header["name"] = "A%02d" % i
        header["description"] = "AI test map %s\n\nAttacker:\n%s\n\nDefender:\n%s" % (
            header["name"],
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
