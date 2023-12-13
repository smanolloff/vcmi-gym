import json as jsonorig
import json5 as json
import os
import glob

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

    creatures = {}

    for filename in files:
        with open(f"{base}/{filename}", "r") as f:
            print(f)
            j = json.load(f)
            for (name, data) in j.items():
                assert name not in creatures
                creatures[name] = data["index"]

    print(jsonorig.dumps(creatures))
