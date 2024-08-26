# Generated from within VCMI (minor post-processing needed):
#
# VLC->creatures()->forEach([](const Creature *c, bool _) {
#     printf(",{\"id\":%d,\"vcminame\":\"%s\",\"name\":\"%s\",\"value\":%d,\"power\":%d}", c->getIndex(), c->getNamePluralTextID().c_str(), c->getNameSingularTranslated().c_str(), c->getAIValue(), c->getFightValue());
# });
#
# NOTE: the above will generate names like "creatures.core.XXX.name.plural"
#       => all text except XXX must be manually removed
# NOTE: The 4 "NOT USED" creatures, warmachines and arrow tower are commented out
#

ALL_CREATURES = [
    {
        "id": 0,
        "vcminame": "pikeman",
        "name": "Pikeman",
        "value": 80,
        "power": 100,
        "shooter": False
    },
    {
        "id": 1,
        "vcminame": "halberdier",
        "name": "Halberdier",
        "value": 115,
        "power": 115,
        "shooter": False
    },
    {
        "id": 2,
        "vcminame": "archer",
        "name": "Archer",
        "value": 126,
        "power": 115,
        "shooter": True
    },
    {
        "id": 3,
        "vcminame": "marksman",
        "name": "Marksman",
        "value": 184,
        "power": 115,
        "shooter": True
    },
    {
        "id": 4,
        "vcminame": "griffin",
        "name": "Griffin",
        "value": 351,
        "power": 324,
        "shooter": False
    },
    {
        "id": 5,
        "vcminame": "royalGriffin",
        "name": "Royal Griffin",
        "value": 448,
        "power": 364,
        "shooter": False
    },
    {
        "id": 6,
        "vcminame": "swordsman",
        "name": "Swordsman",
        "value": 445,
        "power": 445,
        "shooter": False
    },
    {
        "id": 7,
        "vcminame": "crusader",
        "name": "Crusader",
        "value": 588,
        "power": 588,
        "shooter": False
    },
    {
        "id": 8,
        "vcminame": "monk",
        "name": "Monk",
        "value": 485,
        "power": 485,
        "shooter": True
    },
    {
        "id": 9,
        "vcminame": "zealot",
        "name": "Zealot",
        "value": 750,
        "power": 500,
        "shooter": True
    },
    {
        "id": 10,
        "vcminame": "cavalier",
        "name": "Cavalier",
        "value": 1946,
        "power": 1668,
        "shooter": False
    },
    {
        "id": 11,
        "vcminame": "champion",
        "name": "Champion",
        "value": 2100,
        "power": 1800,
        "shooter": False
    },
    {
        "id": 12,
        "vcminame": "angel",
        "name": "Angel",
        "value": 5019,
        "power": 3585,
        "shooter": False
    },
    {
        "id": 13,
        "vcminame": "archangel",
        "name": "Archangel",
        "value": 8776,
        "power": 6033,
        "shooter": False
    },
    {
        "id": 14,
        "vcminame": "centaur",
        "name": "Centaur",
        "value": 100,
        "power": 100,
        "shooter": False
    },
    {
        "id": 15,
        "vcminame": "centaurCaptain",
        "name": "Centaur Captain",
        "value": 138,
        "power": 115,
        "shooter": False
    },
    {
        "id": 16,
        "vcminame": "dwarf",
        "name": "Dwarf",
        "value": 138,
        "power": 194,
        "shooter": False
    },
    {
        "id": 17,
        "vcminame": "battleDwarf",
        "name": "Battle Dwarf",
        "value": 209,
        "power": 209,
        "shooter": False
    },
    {
        "id": 18,
        "vcminame": "woodElf",
        "name": "Wood Elf",
        "value": 234,
        "power": 195,
        "shooter": True
    },
    {
        "id": 19,
        "vcminame": "grandElf",
        "name": "Grand Elf",
        "value": 331,
        "power": 195,
        "shooter": True
    },
    {
        "id": 20,
        "vcminame": "pegasus",
        "name": "Pegasus",
        "value": 518,
        "power": 407,
        "shooter": False
    },
    {
        "id": 21,
        "vcminame": "silverPegasus",
        "name": "Silver Pegasus",
        "value": 532,
        "power": 418,
        "shooter": False
    },
    {
        "id": 22,
        "vcminame": "dendroidGuard",
        "name": "Dendroid Guard",
        "value": 517,
        "power": 690,
        "shooter": False
    },
    {
        "id": 23,
        "vcminame": "dendroidSoldier",
        "name": "Dendroid Soldier",
        "value": 803,
        "power": 765,
        "shooter": False
    },
    {
        "id": 24,
        "vcminame": "unicorn",
        "name": "Unicorn",
        "value": 1806,
        "power": 1548,
        "shooter": False
    },
    {
        "id": 25,
        "vcminame": "warUnicorn",
        "name": "War Unicorn",
        "value": 2030,
        "power": 1740,
        "shooter": False
    },
    {
        "id": 26,
        "vcminame": "greenDragon",
        "name": "Green Dragon",
        "value": 4872,
        "power": 3654,
        "shooter": False
    },
    {
        "id": 27,
        "vcminame": "goldDragon",
        "name": "Gold Dragon",
        "value": 8613,
        "power": 6220,
        "shooter": False
    },
    {
        "id": 28,
        "vcminame": "gremlin",
        "name": "Gremlin",
        "value": 44,
        "power": 55,
        "shooter": False
    },
    {
        "id": 29,
        "vcminame": "masterGremlin",
        "name": "Master Gremlin",
        "value": 66,
        "power": 55,
        "shooter": True
    },
    {
        "id": 30,
        "vcminame": "stoneGargoyle",
        "name": "Stone Gargoyle",
        "value": 165,
        "power": 150,
        "shooter": False
    },
    {
        "id": 31,
        "vcminame": "obsidianGargoyle",
        "name": "Obsidian Gargoyle",
        "value": 201,
        "power": 155,
        "shooter": False
    },
    {
        "id": 32,
        "vcminame": "ironGolem",
        "name": "Stone Golem",
        "value": 250,
        "power": 339,
        "shooter": False
    },
    {
        "id": 33,
        "vcminame": "stoneGolem",
        "name": "Iron Golem",
        "value": 412,
        "power": 412,
        "shooter": False
    },
    {
        "id": 34,
        "vcminame": "mage",
        "name": "Mage",
        "value": 570,
        "power": 418,
        "shooter": True
    },
    {
        "id": 35,
        "vcminame": "archMage",
        "name": "Arch Mage",
        "value": 680,
        "power": 467,
        "shooter": True
    },
    {
        "id": 36,
        "vcminame": "genie",
        "name": "Genie",
        "value": 884,
        "power": 680,
        "shooter": False
    },
    {
        "id": 37,
        "vcminame": "masterGenie",
        "name": "Master Genie",
        "value": 942,
        "power": 748,
        "shooter": False
    },
    {
        "id": 38,
        "vcminame": "naga",
        "name": "Naga",
        "value": 2016,
        "power": 2016,
        "shooter": False
    },
    {
        "id": 39,
        "vcminame": "nagaQueen",
        "name": "Naga Queen",
        "value": 2840,
        "power": 2485,
        "shooter": False
    },
    {
        "id": 40,
        "vcminame": "giant",
        "name": "Giant",
        "value": 3718,
        "power": 3146,
        "shooter": False
    },
    {
        "id": 41,
        "vcminame": "titan",
        "name": "Titan",
        "value": 7500,
        "power": 5000,
        "shooter": True
    },
    {
        "id": 42,
        "vcminame": "imp",
        "name": "Imp",
        "value": 50,
        "power": 50,
        "shooter": False
    },
    {
        "id": 43,
        "vcminame": "familiar",
        "name": "Familiar",
        "value": 60,
        "power": 60,
        "shooter": False
    },
    {
        "id": 44,
        "vcminame": "gog",
        "name": "Gog",
        "value": 159,
        "power": 145,
        "shooter": True
    },
    {
        "id": 45,
        "vcminame": "magog",
        "name": "Magog",
        "value": 240,
        "power": 210,
        "shooter": True
    },
    {
        "id": 46,
        "vcminame": "hellHound",
        "name": "Hell Hound",
        "value": 357,
        "power": 275,
        "shooter": False
    },
    {
        "id": 47,
        "vcminame": "cerberus",
        "name": "Cerberus",
        "value": 392,
        "power": 308,
        "shooter": False
    },
    {
        "id": 48,
        "vcminame": "demon",
        "name": "Demon",
        "value": 445,
        "power": 445,
        "shooter": False
    },
    {
        "id": 49,
        "vcminame": "hornedDemon",
        "name": "Horned Demon",
        "value": 480,
        "power": 480,
        "shooter": False
    },
    {
        "id": 50,
        "vcminame": "pitFiend",
        "name": "Pit Fiend",
        "value": 765,
        "power": 765,
        "shooter": False
    },
    {
        "id": 51,
        "vcminame": "pitLord",
        "name": "Pit Lord",
        "value": 1224,
        "power": 1071,
        "shooter": False
    },
    {
        "id": 52,
        "vcminame": "efreet",
        "name": "Efreeti",
        "value": 1670,
        "power": 1413,
        "shooter": False
    },
    {
        "id": 53,
        "vcminame": "efreetSultan",
        "name": "Efreet Sultan",
        "value": 1848,
        "power": 1584,
        "shooter": False
    },
    {
        "id": 54,
        "vcminame": "devil",
        "name": "Devil",
        "value": 5101,
        "power": 3759,
        "shooter": False
    },
    {
        "id": 55,
        "vcminame": "archDevil",
        "name": "Arch Devil",
        "value": 7115,
        "power": 5243,
        "shooter": False
    },
    {
        "id": 56,
        "vcminame": "skeleton",
        "name": "Skeleton",
        "value": 60,
        "power": 75,
        "shooter": False
    },
    {
        "id": 57,
        "vcminame": "skeletonWarrior",
        "name": "Skeleton Warrior",
        "value": 85,
        "power": 85,
        "shooter": False
    },
    {
        "id": 58,
        "vcminame": "walkingDead",
        "name": "Walking Dead",
        "value": 98,
        "power": 140,
        "shooter": False
    },
    {
        "id": 59,
        "vcminame": "zombieLord",
        "name": "Zombie",
        "value": 128,
        "power": 160,
        "shooter": False
    },
    {
        "id": 60,
        "vcminame": "wight",
        "name": "Wight",
        "value": 252,
        "power": 231,
        "shooter": False
    },
    {
        "id": 61,
        "vcminame": "wraith",
        "name": "Wraith",
        "value": 315,
        "power": 252,
        "shooter": False
    },
    {
        "id": 62,
        "vcminame": "vampire",
        "name": "Vampire",
        "value": 555,
        "power": 518,
        "shooter": False
    },
    {
        "id": 63,
        "vcminame": "vampireLord",
        "name": "Vampire Lord",
        "value": 783,
        "power": 652,
        "shooter": False
    },
    {
        "id": 64,
        "vcminame": "lich",
        "name": "Lich",
        "value": 848,
        "power": 742,
        "shooter": True
    },
    {
        "id": 65,
        "vcminame": "powerLich",
        "name": "Power Lich",
        "value": 1079,
        "power": 889,
        "shooter": True
    },
    {
        "id": 66,
        "vcminame": "blackKnight",
        "name": "Black Knight",
        "value": 2087,
        "power": 1753,
        "shooter": False
    },
    {
        "id": 67,
        "vcminame": "dreadKnight",
        "name": "Dread Knight",
        "value": 2382,
        "power": 2029,
        "shooter": False
    },
    {
        "id": 68,
        "vcminame": "boneDragon",
        "name": "Bone Dragon",
        "value": 3388,
        "power": 2420,
        "shooter": False
    },
    {
        "id": 69,
        "vcminame": "ghostDragon",
        "name": "Ghost Dragon",
        "value": 4696,
        "power": 3228,
        "shooter": False
    },
    {
        "id": 70,
        "vcminame": "troglodyte",
        "name": "Troglodyte",
        "value": 59,
        "power": 73,
        "shooter": False
    },
    {
        "id": 71,
        "vcminame": "infernalTroglodyte",
        "name": "Infernal Troglodyte",
        "value": 84,
        "power": 84,
        "shooter": False
    },
    {
        "id": 72,
        "vcminame": "harpy",
        "name": "Harpy",
        "value": 154,
        "power": 140,
        "shooter": False
    },
    {
        "id": 73,
        "vcminame": "harpyHag",
        "name": "Harpy Hag",
        "value": 238,
        "power": 196,
        "shooter": False
    },
    {
        "id": 74,
        "vcminame": "beholder",
        "name": "Beholder",
        "value": 336,
        "power": 240,
        "shooter": True
    },
    {
        "id": 75,
        "vcminame": "evilEye",
        "name": "Evil Eye",
        "value": 367,
        "power": 245,
        "shooter": True
    },
    {
        "id": 76,
        "vcminame": "medusa",
        "name": "Medusa",
        "value": 517,
        "power": 379,
        "shooter": True
    },
    {
        "id": 77,
        "vcminame": "medusaQueen",
        "name": "Medusa Queen",
        "value": 577,
        "power": 423,
        "shooter": True
    },
    {
        "id": 78,
        "vcminame": "minotaur",
        "name": "Minotaur",
        "value": 835,
        "power": 835,
        "shooter": False
    },
    {
        "id": 79,
        "vcminame": "minotaurKing",
        "name": "Minotaur King",
        "value": 1068,
        "power": 890,
        "shooter": False
    },
    {
        "id": 80,
        "vcminame": "manticore",
        "name": "Manticore",
        "value": 1547,
        "power": 1215,
        "shooter": False
    },
    {
        "id": 81,
        "vcminame": "scorpicore",
        "name": "Scorpicore",
        "value": 1589,
        "power": 1248,
        "shooter": False
    },
    {
        "id": 82,
        "vcminame": "redDragon",
        "name": "Red Dragon",
        "value": 4702,
        "power": 3762,
        "shooter": False
    },
    {
        "id": 83,
        "vcminame": "blackDragon",
        "name": "Black Dragon",
        "value": 8721,
        "power": 6783,
        "shooter": False
    },
    {
        "id": 84,
        "vcminame": "goblin",
        "name": "Goblin",
        "value": 60,
        "power": 60,
        "shooter": False
    },
    {
        "id": 85,
        "vcminame": "hobgoblin",
        "name": "Hobgoblin",
        "value": 78,
        "power": 65,
        "shooter": False
    },
    {
        "id": 86,
        "vcminame": "goblinWolfRider",
        "name": "Wolf Rider",
        "value": 130,
        "power": 130,
        "shooter": False
    },
    {
        "id": 87,
        "vcminame": "hobgoblinWolfRider",
        "name": "Wolf Raider",
        "value": 203,
        "power": 174,
        "shooter": False
    },
    {
        "id": 88,
        "vcminame": "orc",
        "name": "Orc",
        "value": 192,
        "power": 175,
        "shooter": True
    },
    {
        "id": 89,
        "vcminame": "orcChieftain",
        "name": "Orc Chieftain",
        "value": 240,
        "power": 200,
        "shooter": True
    },
    {
        "id": 90,
        "vcminame": "ogre",
        "name": "Ogre",
        "value": 416,
        "power": 520,
        "shooter": False
    },
    {
        "id": 91,
        "vcminame": "ogreMage",
        "name": "Ogre Mage",
        "value": 672,
        "power": 672,
        "shooter": False
    },
    {
        "id": 92,
        "vcminame": "roc",
        "name": "Roc",
        "value": 1027,
        "power": 790,
        "shooter": False
    },
    {
        "id": 93,
        "vcminame": "thunderbird",
        "name": "Thunderbird",
        "value": 1106,
        "power": 869,
        "shooter": False
    },
    {
        "id": 94,
        "vcminame": "cyclop",
        "name": "Cyclops",
        "value": 1266,
        "power": 1055,
        "shooter": False
    },
    {
        "id": 95,
        "vcminame": "cyclopKing",
        "name": "Cyclops King",
        "value": 1443,
        "power": 1110,
        "shooter": False
    },
    {
        "id": 96,
        "vcminame": "behemoth",
        "name": "Behemoth",
        "value": 3162,
        "power": 3162,
        "shooter": False
    },
    {
        "id": 97,
        "vcminame": "ancientBehemoth",
        "name": "Ancient Behemoth",
        "value": 6168,
        "power": 5397,
        "shooter": False
    },
    {
        "id": 98,
        "vcminame": "gnoll",
        "name": "Gnoll",
        "value": 56,
        "power": 70,
        "shooter": False
    },
    {
        "id": 99,
        "vcminame": "gnollMarauder",
        "name": "Gnoll Marauder",
        "value": 90,
        "power": 90,
        "shooter": False
    },
    {
        "id": 100,
        "vcminame": "lizardman",
        "name": "Lizardman",
        "value": 126,
        "power": 115,
        "shooter": True
    },
    {
        "id": 101,
        "vcminame": "lizardWarrior",
        "name": "Lizard Warrior",
        "value": 156,
        "power": 130,
        "shooter": True
    },
    {
        "id": 102,
        "vcminame": "gorgon",
        "name": "Gorgon",
        "value": 890,
        "power": 890,
        "shooter": False
    },
    {
        "id": 103,
        "vcminame": "mightyGorgon",
        "name": "Mighty Gorgon",
        "value": 1028,
        "power": 1028,
        "shooter": False
    },
    {
        "id": 104,
        "vcminame": "serpentFly",
        "name": "Serpent Fly",
        "value": 268,
        "power": 215,
        "shooter": False
    },
    {
        "id": 105,
        "vcminame": "fireDragonFly",
        "name": "Dragon Fly",
        "value": 312,
        "power": 250,
        "shooter": False
    },
    {
        "id": 106,
        "vcminame": "basilisk",
        "name": "Basilisk",
        "value": 552,
        "power": 506,
        "shooter": False
    },
    {
        "id": 107,
        "vcminame": "greaterBasilisk",
        "name": "Greater Basilisk",
        "value": 714,
        "power": 561,
        "shooter": False
    },
    {
        "id": 108,
        "vcminame": "wyvern",
        "name": "Wyvern",
        "value": 1350,
        "power": 1050,
        "shooter": False
    },
    {
        "id": 109,
        "vcminame": "wyvernMonarch",
        "name": "Wyvern Monarch",
        "value": 1518,
        "power": 1181,
        "shooter": False
    },
    {
        "id": 110,
        "vcminame": "hydra",
        "name": "Hydra",
        "value": 4120,
        "power": 4120,
        "shooter": False
    },
    {
        "id": 111,
        "vcminame": "chaosHydra",
        "name": "Chaos Hydra",
        "value": 5931,
        "power": 5272,
        "shooter": False
    },
    {
        "id": 112,
        "vcminame": "airElemental",
        "name": "Air Elemental",
        "value": 356,
        "power": 324,
        "shooter": False
    },
    {
        "id": 113,
        "vcminame": "earthElemental",
        "name": "Earth Elemental",
        "value": 330,
        "power": 415,
        "shooter": False
    },
    {
        "id": 114,
        "vcminame": "fireElemental",
        "name": "Fire Elemental",
        "value": 345,
        "power": 345,
        "shooter": False
    },
    {
        "id": 115,
        "vcminame": "waterElemental",
        "name": "Water Elemental",
        "value": 315,
        "power": 315,
        "shooter": False
    },
    {
        "id": 116,
        "vcminame": "goldGolem",
        "name": "Gold Golem",
        "value": 600,
        "power": 600,
        "shooter": False
    },
    {
        "id": 117,
        "vcminame": "diamondGolem",
        "name": "Diamond Golem",
        "value": 775,
        "power": 775,
        "shooter": False
    },
    {
        "id": 118,
        "vcminame": "pixie",
        "name": "Pixie",
        "value": 55,
        "power": 40,
        "shooter": False
    },
    {
        "id": 119,
        "vcminame": "sprite",
        "name": "Sprite",
        "value": 95,
        "power": 70,
        "shooter": False
    },
    {
        "id": 120,
        "vcminame": "psychicElemental",
        "name": "Psychic Elemental",
        "value": 1669,
        "power": 1431,
        "shooter": False
    },
    {
        "id": 121,
        "vcminame": "magicElemental",
        "name": "Magic Elemental",
        "value": 2012,
        "power": 1724,
        "shooter": False
    },
    # {
    #     "id": 122,
    #     "vcminame": "unused122",
    #     "name": "NOT USED (1)",
    #     "value": 0,
    #     "power": 0,
    # "shooter": False
    # },
    {
        "id": 123,
        "vcminame": "iceElemental",
        "name": "Ice Elemental",
        "value": 380,
        "power": 315,
        "shooter": True
    },
    # {
    #     "id": 124,
    #     "vcminame": "unused124",
    #     "name": "NOT USED (2) ",
    #     "value": 0,
    #     "power": 0,
    # "shooter": False
    # },
    {
        "id": 125,
        "vcminame": "magmaElemental",
        "name": "Magma Elemental",
        "value": 490,
        "power": 490,
        "shooter": False
    },
    # {
    #     "id": 126,
    #     "vcminame": "unused126",
    #     "name": "NOT USED (3)",
    #     "value": 0,
    #     "power": 0,
    # "shooter": False
    # },
    {
        "id": 127,
        "vcminame": "stormElemental",
        "name": "Storm Elemental",
        "value": 486,
        "power": 324,
        "shooter": True
    },
    # {
    #     "id": 128,
    #     "vcminame": "unused128",
    #     "name": "NOT USED (4)",
    #     "value": 0,
    #     "power": 0,
    # "shooter": False
    # },
    {
        "id": 129,
        "vcminame": "energyElemental",
        "name": "Energy Elemental",
        "value": 470,
        "power": 360,
        "shooter": False
    },
    {
        "id": 130,
        "vcminame": "firebird",
        "name": "Firebird",
        "value": 4547,
        "power": 3248,
        "shooter": False
    },
    {
        "id": 131,
        "vcminame": "phoenix",
        "name": "Phoenix",
        "value": 6721,
        "power": 4929,
        "shooter": False
    },
    {
        "id": 132,
        "vcminame": "azureDragon",
        "name": "Azure Dragon",
        "value": 78845,
        "power": 56315,
        "shooter": False
    },
    {
        "id": 133,
        "vcminame": "crystalDragon",
        "name": "Crystal Dragon",
        "value": 39338,
        "power": 30260,
        "shooter": False
    },
    {
        "id": 134,
        "vcminame": "fairieDragon",
        "name": "Faerie Dragon",
        "value": 19580,
        "power": 16317,
        "shooter": False
    },
    {
        "id": 135,
        "vcminame": "rustDragon",
        "name": "Rust Dragon",
        "value": 26433,
        "power": 24030,
        "shooter": False
    },
    {
        "id": 136,
        "vcminame": "enchanter",
        "name": "Enchanter",
        "value": 1210,
        "power": 805,
        "shooter": True
    },
    {
        "id": 137,
        "vcminame": "sharpshooter",
        "name": "Sharpshooter",
        "value": 585,
        "power": 415,
        "shooter": True
    },
    {
        "id": 138,
        "vcminame": "halfling",
        "name": "Halfling",
        "value": 75,
        "power": 60,
        "shooter": True
    },
    {
        "id": 139,
        "vcminame": "peasant",
        "name": "Peasant",
        "value": 15,
        "power": 15,
        "shooter": False
    },
    {
        "id": 140,
        "vcminame": "boar",
        "name": "Boar",
        "value": 145,
        "power": 145,
        "shooter": False
    },
    {
        "id": 141,
        "vcminame": "mummy",
        "name": "Mummy",
        "value": 270,
        "power": 270,
        "shooter": False
    },
    {
        "id": 142,
        "vcminame": "nomad",
        "name": "Nomad",
        "value": 345,
        "power": 285,
        "shooter": False
    },
    {
        "id": 143,
        "vcminame": "rogue",
        "name": "Rogue",
        "value": 135,
        "power": 135,
        "shooter": False
    },
    {
        "id": 144,
        "vcminame": "troll",
        "name": "Troll",
        "value": 1024,
        "power": 1024,
        "shooter": False
    },
    # {
    #     "id": 145,
    #     "vcminame": "catapult",
    #     "name": "Catapult",
    #     "value": 500,
    #     "power": 10,
    # "shooter": False
    # },
    # {
    #     "id": 146,
    #     "vcminame": "ballista",
    #     "name": "Ballista",
    #     "value": 600,
    #     "power": 650,
    # "shooter": True
    # },
    # {
    #     "id": 147,
    #     "vcminame": "firstAidTent",
    #     "name": "First Aid Tent",
    #     "value": 300,
    #     "power": 10,
    # "shooter": False
    # },
    # {
    #     "id": 148,
    #     "vcminame": "ammoCart",
    #     "name": "Ammo Cart",
    #     "value": 400,
    #     "power": 5,
    # "shooter": False
    # },
    # {
    #     "id": 149,
    #     "vcminame": "arrowTower",
    #     "name": "Arrow Tower",
    #     "value": 400,
    #     "power": 5,
    # "shooter": True
    # }
]
