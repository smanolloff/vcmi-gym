import json
import sys
import zipfile
import io
import random

# Usage:
# python maps/mapgen/add_hero_names.py path/to/map.vmap


if __name__ == "__main__":
    mapname = sys.argv[1]
    header = None
    objects = None
    surface_terrain = None

    with zipfile.ZipFile(mapname, 'r') as zip_ref:
        with zip_ref.open("header.json") as file:
            header = json.load(file)
        with zip_ref.open("objects.json") as file:
            objects = json.load(file)
        with zip_ref.open("surface_terrain.json") as file:
            surface_terrain = json.load(file)

    for k, v in objects.items():
        if k.startswith("hero_"):
            sp = random.randint(5, 15)
            # mana is randomized before each battle => knowledge is irrelevant
            v["options"]["primarySkills"] = {"knowledge": 1, "spellpower": sp}
            v["options"]["spellBook"] = [
                "preset",
                "core:fireElemental",
                "core:earthElemental",
                "core:waterElemental",
                "core:airElemental"
            ]

    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zipf:
        zipf.writestr('header.json', json.dumps(header))
        zipf.writestr('objects.json', json.dumps(objects))
        zipf.writestr('surface_terrain.json', json.dumps(surface_terrain))

    print("Updating %s" % mapname)
    with open(mapname, 'wb') as f:
        f.write(memory_zip.getvalue())
