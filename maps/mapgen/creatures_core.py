# Generated from within VCMI (minor post-processing needed):
#
# VLC->creatures()->forEach([](const Creature *c, bool _) {
#     printf(",{\"id\":%d,\"vcminame\":\"%s\",\"name\":\"%s\",\"value\":%d,\"power\":%d}", c->getIndex(), c->getNamePluralTextID().c_str(), c->getNameSingularTranslated().c_str(), c->getAIValue(), c->getFightValue());
# });
#
# NOTE: "special" creatures (war machines, arrow tower)
#       as well as 4 unused creatures are commented out
#

ALL_CREATURES = [
    {
        "id": 0,
        "vcminame": "creatures.core.pikeman.name.plural",
        "name": "Pikeman",
        "value": 80,
        "power": 100
    },
    {
        "id": 1,
        "vcminame": "creatures.core.halberdier.name.plural",
        "name": "Halberdier",
        "value": 115,
        "power": 115
    },
    {
        "id": 2,
        "vcminame": "creatures.core.archer.name.plural",
        "name": "Archer",
        "value": 126,
        "power": 115
    },
    {
        "id": 3,
        "vcminame": "creatures.core.marksman.name.plural",
        "name": "Marksman",
        "value": 184,
        "power": 115
    },
    {
        "id": 4,
        "vcminame": "creatures.core.griffin.name.plural",
        "name": "Griffin",
        "value": 351,
        "power": 324
    },
    {
        "id": 5,
        "vcminame": "creatures.core.royalGriffin.name.plural",
        "name": "Royal Griffin",
        "value": 448,
        "power": 364
    },
    {
        "id": 6,
        "vcminame": "creatures.core.swordsman.name.plural",
        "name": "Swordsman",
        "value": 445,
        "power": 445
    },
    {
        "id": 7,
        "vcminame": "creatures.core.crusader.name.plural",
        "name": "Crusader",
        "value": 588,
        "power": 588
    },
    {
        "id": 8,
        "vcminame": "creatures.core.monk.name.plural",
        "name": "Monk",
        "value": 485,
        "power": 485
    },
    {
        "id": 9,
        "vcminame": "creatures.core.zealot.name.plural",
        "name": "Zealot",
        "value": 750,
        "power": 500
    },
    {
        "id": 10,
        "vcminame": "creatures.core.cavalier.name.plural",
        "name": "Cavalier",
        "value": 1946,
        "power": 1668
    },
    {
        "id": 11,
        "vcminame": "creatures.core.champion.name.plural",
        "name": "Champion",
        "value": 2100,
        "power": 1800
    },
    {
        "id": 12,
        "vcminame": "creatures.core.angel.name.plural",
        "name": "Angel",
        "value": 5019,
        "power": 3585
    },
    {
        "id": 13,
        "vcminame": "creatures.core.archangel.name.plural",
        "name": "Archangel",
        "value": 8776,
        "power": 6033
    },
    {
        "id": 14,
        "vcminame": "creatures.core.centaur.name.plural",
        "name": "Centaur",
        "value": 100,
        "power": 100
    },
    {
        "id": 15,
        "vcminame": "creatures.core.centaurCaptain.name.plural",
        "name": "Centaur Captain",
        "value": 138,
        "power": 115
    },
    {
        "id": 16,
        "vcminame": "creatures.core.dwarf.name.plural",
        "name": "Dwarf",
        "value": 138,
        "power": 194
    },
    {
        "id": 17,
        "vcminame": "creatures.core.battleDwarf.name.plural",
        "name": "Battle Dwarf",
        "value": 209,
        "power": 209
    },
    {
        "id": 18,
        "vcminame": "creatures.core.woodElf.name.plural",
        "name": "Wood Elf",
        "value": 234,
        "power": 195
    },
    {
        "id": 19,
        "vcminame": "creatures.core.grandElf.name.plural",
        "name": "Grand Elf",
        "value": 331,
        "power": 195
    },
    {
        "id": 20,
        "vcminame": "creatures.core.pegasus.name.plural",
        "name": "Pegasus",
        "value": 518,
        "power": 407
    },
    {
        "id": 21,
        "vcminame": "creatures.core.silverPegasus.name.plural",
        "name": "Silver Pegasus",
        "value": 532,
        "power": 418
    },
    {
        "id": 22,
        "vcminame": "creatures.core.dendroidGuard.name.plural",
        "name": "Dendroid Guard",
        "value": 517,
        "power": 690
    },
    {
        "id": 23,
        "vcminame": "creatures.core.dendroidSoldier.name.plural",
        "name": "Dendroid Soldier",
        "value": 803,
        "power": 765
    },
    {
        "id": 24,
        "vcminame": "creatures.core.unicorn.name.plural",
        "name": "Unicorn",
        "value": 1806,
        "power": 1548
    },
    {
        "id": 25,
        "vcminame": "creatures.core.warUnicorn.name.plural",
        "name": "War Unicorn",
        "value": 2030,
        "power": 1740
    },
    {
        "id": 26,
        "vcminame": "creatures.core.greenDragon.name.plural",
        "name": "Green Dragon",
        "value": 4872,
        "power": 3654
    },
    {
        "id": 27,
        "vcminame": "creatures.core.goldDragon.name.plural",
        "name": "Gold Dragon",
        "value": 8613,
        "power": 6220
    },
    {
        "id": 28,
        "vcminame": "creatures.core.gremlin.name.plural",
        "name": "Gremlin",
        "value": 44,
        "power": 55
    },
    {
        "id": 29,
        "vcminame": "creatures.core.masterGremlin.name.plural",
        "name": "Master Gremlin",
        "value": 66,
        "power": 55
    },
    {
        "id": 30,
        "vcminame": "creatures.core.stoneGargoyle.name.plural",
        "name": "Stone Gargoyle",
        "value": 165,
        "power": 150
    },
    {
        "id": 31,
        "vcminame": "creatures.core.obsidianGargoyle.name.plural",
        "name": "Obsidian Gargoyle",
        "value": 201,
        "power": 155
    },
    {
        "id": 32,
        "vcminame": "creatures.core.ironGolem.name.plural",
        "name": "Stone Golem",
        "value": 250,
        "power": 339
    },
    {
        "id": 33,
        "vcminame": "creatures.core.stoneGolem.name.plural",
        "name": "Iron Golem",
        "value": 412,
        "power": 412
    },
    {
        "id": 34,
        "vcminame": "creatures.core.mage.name.plural",
        "name": "Mage",
        "value": 570,
        "power": 418
    },
    {
        "id": 35,
        "vcminame": "creatures.core.archMage.name.plural",
        "name": "Arch Mage",
        "value": 680,
        "power": 467
    },
    {
        "id": 36,
        "vcminame": "creatures.core.genie.name.plural",
        "name": "Genie",
        "value": 884,
        "power": 680
    },
    {
        "id": 37,
        "vcminame": "creatures.core.masterGenie.name.plural",
        "name": "Master Genie",
        "value": 942,
        "power": 748
    },
    {
        "id": 38,
        "vcminame": "creatures.core.naga.name.plural",
        "name": "Naga",
        "value": 2016,
        "power": 2016
    },
    {
        "id": 39,
        "vcminame": "creatures.core.nagaQueen.name.plural",
        "name": "Naga Queen",
        "value": 2840,
        "power": 2485
    },
    {
        "id": 40,
        "vcminame": "creatures.core.giant.name.plural",
        "name": "Giant",
        "value": 3718,
        "power": 3146
    },
    {
        "id": 41,
        "vcminame": "creatures.core.titan.name.plural",
        "name": "Titan",
        "value": 7500,
        "power": 5000
    },
    {
        "id": 42,
        "vcminame": "creatures.core.imp.name.plural",
        "name": "Imp",
        "value": 50,
        "power": 50
    },
    {
        "id": 43,
        "vcminame": "creatures.core.familiar.name.plural",
        "name": "Familiar",
        "value": 60,
        "power": 60
    },
    {
        "id": 44,
        "vcminame": "creatures.core.gog.name.plural",
        "name": "Gog",
        "value": 159,
        "power": 145
    },
    {
        "id": 45,
        "vcminame": "creatures.core.magog.name.plural",
        "name": "Magog",
        "value": 240,
        "power": 210
    },
    {
        "id": 46,
        "vcminame": "creatures.core.hellHound.name.plural",
        "name": "Hell Hound",
        "value": 357,
        "power": 275
    },
    {
        "id": 47,
        "vcminame": "creatures.core.cerberus.name.plural",
        "name": "Cerberus",
        "value": 392,
        "power": 308
    },
    {
        "id": 48,
        "vcminame": "creatures.core.demon.name.plural",
        "name": "Demon",
        "value": 445,
        "power": 445
    },
    {
        "id": 49,
        "vcminame": "creatures.core.hornedDemon.name.plural",
        "name": "Horned Demon",
        "value": 480,
        "power": 480
    },
    {
        "id": 50,
        "vcminame": "creatures.core.pitFiend.name.plural",
        "name": "Pit Fiend",
        "value": 765,
        "power": 765
    },
    {
        "id": 51,
        "vcminame": "creatures.core.pitLord.name.plural",
        "name": "Pit Lord",
        "value": 1224,
        "power": 1071
    },
    {
        "id": 52,
        "vcminame": "creatures.core.efreet.name.plural",
        "name": "Efreeti",
        "value": 1670,
        "power": 1413
    },
    {
        "id": 53,
        "vcminame": "creatures.core.efreetSultan.name.plural",
        "name": "Efreet Sultan",
        "value": 1848,
        "power": 1584
    },
    {
        "id": 54,
        "vcminame": "creatures.core.devil.name.plural",
        "name": "Devil",
        "value": 5101,
        "power": 3759
    },
    {
        "id": 55,
        "vcminame": "creatures.core.archDevil.name.plural",
        "name": "Arch Devil",
        "value": 7115,
        "power": 5243
    },
    {
        "id": 56,
        "vcminame": "creatures.core.skeleton.name.plural",
        "name": "Skeleton",
        "value": 60,
        "power": 75
    },
    {
        "id": 57,
        "vcminame": "creatures.core.skeletonWarrior.name.plural",
        "name": "Skeleton Warrior",
        "value": 85,
        "power": 85
    },
    {
        "id": 58,
        "vcminame": "creatures.core.walkingDead.name.plural",
        "name": "Walking Dead",
        "value": 98,
        "power": 140
    },
    {
        "id": 59,
        "vcminame": "creatures.core.zombieLord.name.plural",
        "name": "Zombie",
        "value": 128,
        "power": 160
    },
    {
        "id": 60,
        "vcminame": "creatures.core.wight.name.plural",
        "name": "Wight",
        "value": 252,
        "power": 231
    },
    {
        "id": 61,
        "vcminame": "creatures.core.wraith.name.plural",
        "name": "Wraith",
        "value": 315,
        "power": 252
    },
    {
        "id": 62,
        "vcminame": "creatures.core.vampire.name.plural",
        "name": "Vampire",
        "value": 555,
        "power": 518
    },
    {
        "id": 63,
        "vcminame": "creatures.core.vampireLord.name.plural",
        "name": "Vampire Lord",
        "value": 783,
        "power": 652
    },
    {
        "id": 64,
        "vcminame": "creatures.core.lich.name.plural",
        "name": "Lich",
        "value": 848,
        "power": 742
    },
    {
        "id": 65,
        "vcminame": "creatures.core.powerLich.name.plural",
        "name": "Power Lich",
        "value": 1079,
        "power": 889
    },
    {
        "id": 66,
        "vcminame": "creatures.core.blackKnight.name.plural",
        "name": "Black Knight",
        "value": 2087,
        "power": 1753
    },
    {
        "id": 67,
        "vcminame": "creatures.core.dreadKnight.name.plural",
        "name": "Dread Knight",
        "value": 2382,
        "power": 2029
    },
    {
        "id": 68,
        "vcminame": "creatures.core.boneDragon.name.plural",
        "name": "Bone Dragon",
        "value": 3388,
        "power": 2420
    },
    {
        "id": 69,
        "vcminame": "creatures.core.ghostDragon.name.plural",
        "name": "Ghost Dragon",
        "value": 4696,
        "power": 3228
    },
    {
        "id": 70,
        "vcminame": "creatures.core.troglodyte.name.plural",
        "name": "Troglodyte",
        "value": 59,
        "power": 73
    },
    {
        "id": 71,
        "vcminame": "creatures.core.infernalTroglodyte.name.plural",
        "name": "Infernal Troglodyte",
        "value": 84,
        "power": 84
    },
    {
        "id": 72,
        "vcminame": "creatures.core.harpy.name.plural",
        "name": "Harpy",
        "value": 154,
        "power": 140
    },
    {
        "id": 73,
        "vcminame": "creatures.core.harpyHag.name.plural",
        "name": "Harpy Hag",
        "value": 238,
        "power": 196
    },
    {
        "id": 74,
        "vcminame": "creatures.core.beholder.name.plural",
        "name": "Beholder",
        "value": 336,
        "power": 240
    },
    {
        "id": 75,
        "vcminame": "creatures.core.evilEye.name.plural",
        "name": "Evil Eye",
        "value": 367,
        "power": 245
    },
    {
        "id": 76,
        "vcminame": "creatures.core.medusa.name.plural",
        "name": "Medusa",
        "value": 517,
        "power": 379
    },
    {
        "id": 77,
        "vcminame": "creatures.core.medusaQueen.name.plural",
        "name": "Medusa Queen",
        "value": 577,
        "power": 423
    },
    {
        "id": 78,
        "vcminame": "creatures.core.minotaur.name.plural",
        "name": "Minotaur",
        "value": 835,
        "power": 835
    },
    {
        "id": 79,
        "vcminame": "creatures.core.minotaurKing.name.plural",
        "name": "Minotaur King",
        "value": 1068,
        "power": 890
    },
    {
        "id": 80,
        "vcminame": "creatures.core.manticore.name.plural",
        "name": "Manticore",
        "value": 1547,
        "power": 1215
    },
    {
        "id": 81,
        "vcminame": "creatures.core.scorpicore.name.plural",
        "name": "Scorpicore",
        "value": 1589,
        "power": 1248
    },
    {
        "id": 82,
        "vcminame": "creatures.core.redDragon.name.plural",
        "name": "Red Dragon",
        "value": 4702,
        "power": 3762
    },
    {
        "id": 83,
        "vcminame": "creatures.core.blackDragon.name.plural",
        "name": "Black Dragon",
        "value": 8721,
        "power": 6783
    },
    {
        "id": 84,
        "vcminame": "creatures.core.goblin.name.plural",
        "name": "Goblin",
        "value": 60,
        "power": 60
    },
    {
        "id": 85,
        "vcminame": "creatures.core.hobgoblin.name.plural",
        "name": "Hobgoblin",
        "value": 78,
        "power": 65
    },
    {
        "id": 86,
        "vcminame": "creatures.core.goblinWolfRider.name.plural",
        "name": "Wolf Rider",
        "value": 130,
        "power": 130
    },
    {
        "id": 87,
        "vcminame": "creatures.core.hobgoblinWolfRider.name.plural",
        "name": "Wolf Raider",
        "value": 203,
        "power": 174
    },
    {
        "id": 88,
        "vcminame": "creatures.core.orc.name.plural",
        "name": "Orc",
        "value": 192,
        "power": 175
    },
    {
        "id": 89,
        "vcminame": "creatures.core.orcChieftain.name.plural",
        "name": "Orc Chieftain",
        "value": 240,
        "power": 200
    },
    {
        "id": 90,
        "vcminame": "creatures.core.ogre.name.plural",
        "name": "Ogre",
        "value": 416,
        "power": 520
    },
    {
        "id": 91,
        "vcminame": "creatures.core.ogreMage.name.plural",
        "name": "Ogre Mage",
        "value": 672,
        "power": 672
    },
    {
        "id": 92,
        "vcminame": "creatures.core.roc.name.plural",
        "name": "Roc",
        "value": 1027,
        "power": 790
    },
    {
        "id": 93,
        "vcminame": "creatures.core.thunderbird.name.plural",
        "name": "Thunderbird",
        "value": 1106,
        "power": 869
    },
    {
        "id": 94,
        "vcminame": "creatures.core.cyclop.name.plural",
        "name": "Cyclops",
        "value": 1266,
        "power": 1055
    },
    {
        "id": 95,
        "vcminame": "creatures.core.cyclopKing.name.plural",
        "name": "Cyclops King",
        "value": 1443,
        "power": 1110
    },
    {
        "id": 96,
        "vcminame": "creatures.core.behemoth.name.plural",
        "name": "Behemoth",
        "value": 3162,
        "power": 3162
    },
    {
        "id": 97,
        "vcminame": "creatures.core.ancientBehemoth.name.plural",
        "name": "Ancient Behemoth",
        "value": 6168,
        "power": 5397
    },
    {
        "id": 98,
        "vcminame": "creatures.core.gnoll.name.plural",
        "name": "Gnoll",
        "value": 56,
        "power": 70
    },
    {
        "id": 99,
        "vcminame": "creatures.core.gnollMarauder.name.plural",
        "name": "Gnoll Marauder",
        "value": 90,
        "power": 90
    },
    {
        "id": 100,
        "vcminame": "creatures.core.lizardman.name.plural",
        "name": "Lizardman",
        "value": 126,
        "power": 115
    },
    {
        "id": 101,
        "vcminame": "creatures.core.lizardWarrior.name.plural",
        "name": "Lizard Warrior",
        "value": 156,
        "power": 130
    },
    {
        "id": 102,
        "vcminame": "creatures.core.gorgon.name.plural",
        "name": "Gorgon",
        "value": 890,
        "power": 890
    },
    {
        "id": 103,
        "vcminame": "creatures.core.mightyGorgon.name.plural",
        "name": "Mighty Gorgon",
        "value": 1028,
        "power": 1028
    },
    {
        "id": 104,
        "vcminame": "creatures.core.serpentFly.name.plural",
        "name": "Serpent Fly",
        "value": 268,
        "power": 215
    },
    {
        "id": 105,
        "vcminame": "creatures.core.fireDragonFly.name.plural",
        "name": "Dragon Fly",
        "value": 312,
        "power": 250
    },
    {
        "id": 106,
        "vcminame": "creatures.core.basilisk.name.plural",
        "name": "Basilisk",
        "value": 552,
        "power": 506
    },
    {
        "id": 107,
        "vcminame": "creatures.core.greaterBasilisk.name.plural",
        "name": "Greater Basilisk",
        "value": 714,
        "power": 561
    },
    {
        "id": 108,
        "vcminame": "creatures.core.wyvern.name.plural",
        "name": "Wyvern",
        "value": 1350,
        "power": 1050
    },
    {
        "id": 109,
        "vcminame": "creatures.core.wyvernMonarch.name.plural",
        "name": "Wyvern Monarch",
        "value": 1518,
        "power": 1181
    },
    {
        "id": 110,
        "vcminame": "creatures.core.hydra.name.plural",
        "name": "Hydra",
        "value": 4120,
        "power": 4120
    },
    {
        "id": 111,
        "vcminame": "creatures.core.chaosHydra.name.plural",
        "name": "Chaos Hydra",
        "value": 5931,
        "power": 5272
    },
    {
        "id": 112,
        "vcminame": "creatures.core.airElemental.name.plural",
        "name": "Air Elemental",
        "value": 356,
        "power": 324
    },
    {
        "id": 113,
        "vcminame": "creatures.core.earthElemental.name.plural",
        "name": "Earth Elemental",
        "value": 330,
        "power": 415
    },
    {
        "id": 114,
        "vcminame": "creatures.core.fireElemental.name.plural",
        "name": "Fire Elemental",
        "value": 345,
        "power": 345
    },
    {
        "id": 115,
        "vcminame": "creatures.core.waterElemental.name.plural",
        "name": "Water Elemental",
        "value": 315,
        "power": 315
    },
    {
        "id": 116,
        "vcminame": "creatures.core.goldGolem.name.plural",
        "name": "Gold Golem",
        "value": 600,
        "power": 600
    },
    {
        "id": 117,
        "vcminame": "creatures.core.diamondGolem.name.plural",
        "name": "Diamond Golem",
        "value": 775,
        "power": 775
    },
    {
        "id": 118,
        "vcminame": "creatures.core.pixie.name.plural",
        "name": "Pixie",
        "value": 55,
        "power": 40
    },
    {
        "id": 119,
        "vcminame": "creatures.core.sprite.name.plural",
        "name": "Sprite",
        "value": 95,
        "power": 70
    },
    {
        "id": 120,
        "vcminame": "creatures.core.psychicElemental.name.plural",
        "name": "Psychic Elemental",
        "value": 1669,
        "power": 1431
    },
    {
        "id": 121,
        "vcminame": "creatures.core.magicElemental.name.plural",
        "name": "Magic Elemental",
        "value": 2012,
        "power": 1724
    },
    {
        "id": 122,
        "vcminame": "creatures.core.unused122.name.plural",
        "name": "NOT USED (1)",
        "value": 0,
        "power": 0
    },
    {
        "id": 123,
        "vcminame": "creatures.core.iceElemental.name.plural",
        "name": "Ice Elemental",
        "value": 380,
        "power": 315
    },
    {
        "id": 124,
        "vcminame": "creatures.core.unused124.name.plural",
        "name": "NOT USED (2) ",
        "value": 0,
        "power": 0
    },
    {
        "id": 125,
        "vcminame": "creatures.core.magmaElemental.name.plural",
        "name": "Magma Elemental",
        "value": 490,
        "power": 490
    },
    {
        "id": 126,
        "vcminame": "creatures.core.unused126.name.plural",
        "name": "NOT USED (3)",
        "value": 0,
        "power": 0
    },
    {
        "id": 127,
        "vcminame": "creatures.core.stormElemental.name.plural",
        "name": "Storm Elemental",
        "value": 486,
        "power": 324
    },
    {
        "id": 128,
        "vcminame": "creatures.core.unused128.name.plural",
        "name": "NOT USED (4)",
        "value": 0,
        "power": 0
    },
    {
        "id": 129,
        "vcminame": "creatures.core.energyElemental.name.plural",
        "name": "Energy Elemental",
        "value": 470,
        "power": 360
    },
    {
        "id": 130,
        "vcminame": "creatures.core.firebird.name.plural",
        "name": "Firebird",
        "value": 4547,
        "power": 3248
    },
    {
        "id": 131,
        "vcminame": "creatures.core.phoenix.name.plural",
        "name": "Phoenix",
        "value": 6721,
        "power": 4929
    },
    {
        "id": 132,
        "vcminame": "creatures.core.azureDragon.name.plural",
        "name": "Azure Dragon",
        "value": 78845,
        "power": 56315
    },
    {
        "id": 133,
        "vcminame": "creatures.core.crystalDragon.name.plural",
        "name": "Crystal Dragon",
        "value": 39338,
        "power": 30260
    },
    {
        "id": 134,
        "vcminame": "creatures.core.fairieDragon.name.plural",
        "name": "Faerie Dragon",
        "value": 19580,
        "power": 16317
    },
    {
        "id": 135,
        "vcminame": "creatures.core.rustDragon.name.plural",
        "name": "Rust Dragon",
        "value": 26433,
        "power": 24030
    },
    {
        "id": 136,
        "vcminame": "creatures.core.enchanter.name.plural",
        "name": "Enchanter",
        "value": 1210,
        "power": 805
    },
    {
        "id": 137,
        "vcminame": "creatures.core.sharpshooter.name.plural",
        "name": "Sharpshooter",
        "value": 585,
        "power": 415
    },
    {
        "id": 138,
        "vcminame": "creatures.core.halfling.name.plural",
        "name": "Halfling",
        "value": 75,
        "power": 60
    },
    {
        "id": 139,
        "vcminame": "creatures.core.peasant.name.plural",
        "name": "Peasant",
        "value": 15,
        "power": 15
    },
    {
        "id": 140,
        "vcminame": "creatures.core.boar.name.plural",
        "name": "Boar",
        "value": 145,
        "power": 145
    },
    {
        "id": 141,
        "vcminame": "creatures.core.mummy.name.plural",
        "name": "Mummy",
        "value": 270,
        "power": 270
    },
    {
        "id": 142,
        "vcminame": "creatures.core.nomad.name.plural",
        "name": "Nomad",
        "value": 345,
        "power": 285
    },
    {
        "id": 143,
        "vcminame": "creatures.core.rogue.name.plural",
        "name": "Rogue",
        "value": 135,
        "power": 135
    },
    {
        "id": 144,
        "vcminame": "creatures.core.troll.name.plural",
        "name": "Troll",
        "value": 1024,
        "power": 1024
    },
    {
        "id": 145,
        "vcminame": "creatures.core.catapult.name.plural",
        "name": "Catapult",
        "value": 500,
        "power": 10
    },
    {
        "id": 146,
        "vcminame": "creatures.core.ballista.name.plural",
        "name": "Ballista",
        "value": 600,
        "power": 650
    },
    {
        "id": 147,
        "vcminame": "creatures.core.firstAidTent.name.plural",
        "name": "First Aid Tent",
        "value": 300,
        "power": 10
    },
    {
        "id": 148,
        "vcminame": "creatures.core.ammoCart.name.plural",
        "name": "Ammo Cart",
        "value": 400,
        "power": 5
    },
    {
        "id": 149,
        "vcminame": "creatures.core.arrowTower.name.plural",
        "name": "Arrow Tower",
        "value": 400,
        "power": 5
    }
]
