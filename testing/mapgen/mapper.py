import json as jsonorig
import json5 as json
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    base = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/config/creatures"

    files = [
        "castle.json",
        "conflux.json",
        "dungeon.json",
        "fortress.json",
        "inferno.json",
        "necropolis.json",
        "neutral.json",
        "rampart.json",
        "stronghold.json",
        "tower.json"
    ]

    indexes = {}
    with open(f"{current_dir}/creatures_indexes.json", "r") as f:
        indexes = json.load(f)

    values = {}
    with open(f"{current_dir}/creatures_aivalues.json", "r") as f:
        values = json.load(f)

    vcminames = {}
    for filename in files:
        with open(f"{base}/{filename}", "r") as f:
            print(f)
            j = json.load(f)
            for (name, data) in j.items():
                assert name not in vcminames
                vcminames[data["index"]] = name

    name_to_vcminame = {}
    for (name, index) in indexes.items():
        name_to_vcminame[name] = vcminames[index]

    print(jsonorig.dumps(name_to_vcminame))

    vcminame_to_name_and_value = {}
    for (name, index) in values.items():
        vcminame = name_to_vcminame[name]
        vcminame_to_name_and_value[vcminame] = [name, values[name]]

    print(jsonorig.dumps(vcminame_to_name_and_value))
