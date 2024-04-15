if __name__ == "__main__":
    all_creatures = get_all_creatures()
    all_values = list(range(ARMY_VALUE_MIN, ARMY_VALUE_MAX+1, ARMY_VALUE_ROUND))

    for mapid in range(MAP_ID_START, MAP_ID_END + 1):
        print(f"*** Generating map #{mapid}")
        header, objects, surface_terrain0, hero_mapping = get_templates()
        value = random.choice(all_values)
        n_stacks = random.randint(ARMY_N_STACKS_MIN, ARMY_N_STACKS_MAX)

        header["name"] = MAP_NAME_TEMPLATE.format(id=mapid)
        header["description"] = "AI test map %s\nTarget army values: %d" % (header["name"], value)
        oid = 0

        # for y in range(2, 11):  # 28 rows
        #     for x in range(2, 11):  # 32 columns
        for y in range(2, 30):  # 28 rows
            for x in range(3, 35):  # 32 columns
                print(f"* Generating army for hero #{oid}")
                n_stacks = n_stacks if ARMY_N_STACKS_SAME else random.randint(ARMY_N_STACKS_MIN, ARMY_N_STACKS_MAX)
                army = build_army_with_retry(value, ARMY_VALUE_ERROR_MAX, n_stacks=n_stacks, all_creatures=all_creatures)
                hero_army = [{} for i in range(7)]
                for (slot, (vcminame, _, number)) in enumerate(army):
                    hero_army[slot] = dict(amount=number, type=f"core:{vcminame}")

                values = dict(hero_mapping[(x-2) % 8], id=oid, x=x, y=y)
                color = values["color"]
                header["players"][color]["heroes"][f"hero_{oid}"] = dict(type=f"core:{values['name']}")

                objects[f"hero_{oid}"] = dict(
                    type="hero", subtype=values["type"], x=x, y=y, l=0,
                    options=dict(
                        experience=oid,
                        formation="wide",
                        gender=1,
                        owner=values["color"],
                        portrait=f"core:{values['name']}",
                        type=f"core:{values['name']}",
                        army=hero_army
                    ),
                    template=dict(
                        animation=f"{values['animation']}_.def",
                        editorAnimation=f"{values['animation']}_E.def",
                        mask=["VVV", "VAV"],
                        visitableFrom=["+++", "+-+", "+++"],
                    )
                )

                oid += 1

        for y in range(5, 30, 3):
            for x in range(5, 36, 5):
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
