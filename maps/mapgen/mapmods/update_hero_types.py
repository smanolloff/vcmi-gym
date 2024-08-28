import json
import sys
import zipfile
import io

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
        if not k.startswith("hero_"):
            continue

        v["options"]["type"] = "ml:%s" % k
        v["subtype"] = "core:cleric"
        v["template"]["animation"] = "AH01_.def"
        v["template"]["editorAnimation"] = "AH01_E.def"
        del v["options"]["formation"]
        del v["options"]["gender"]
        del v["options"]["portrait"]

    for v0 in header["players"].values():
        for k1, v1 in v0["heroes"].items():
            v1["type"] = "ml:%s" % k1

    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zipf:
        zipf.writestr('header.json', json.dumps(header))
        zipf.writestr('objects.json', json.dumps(objects))
        zipf.writestr('surface_terrain.json', json.dumps(surface_terrain))

    print("Updating %s" % mapname)
    with open(mapname, 'wb') as f:
        f.write(memory_zip.getvalue())
