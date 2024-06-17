// =============================================================================
// Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AI/MMAI/schema/schema.h"
#include "conncommon.h"

using namespace MMAI::Schema;
using namespace MMAI::Schema::V1;

constexpr int get_n_actions() { return N_ACTIONS; }
constexpr int get_n_nonhex_actions() { return N_NONHEX_ACTIONS; }
constexpr int get_n_hex_actions() { return N_HEX_ACTIONS; }
constexpr int get_state_size() { return BATTLEFIELD_STATE_SIZE; }
constexpr int get_state_size_one_hex() { return BATTLEFIELD_STATE_SIZE_ONE_HEX; }
constexpr int get_state_value_na() { return BATTLEFIELD_STATE_VALUE_NA; }
constexpr int get_side_left() { return static_cast<int>(MMAI::Schema::V1::ISupplementaryData::Side::LEFT); }
constexpr int get_side_right() { return static_cast<int>(MMAI::Schema::V1::ISupplementaryData::Side::RIGHT); }

static const std::vector<std::string> get_dmgmods() {
    auto mods = std::vector<std::string> {};
    std::string modname;

    for (int i=0; i < static_cast<int>(DmgMod::_count); i++) {
        switch (DmgMod(i)) {
        break; case DmgMod::ZERO: modname = "ZERO";
        break; case DmgMod::HALF: modname = "HALF";
        break; case DmgMod::FULL: modname = "FULL";
        break; default:
            throw std::runtime_error("Unexpected DmgMod: " + std::to_string(i));
        }

        mods.push_back(modname);
    }

    return mods;
}

static const std::vector<std::string> get_shootdistances() {
    auto dists = std::vector<std::string> {};
    std::string dist;

    for (int i=0; i < static_cast<int>(ShootDistance::_count); i++) {
        switch (ShootDistance(i)) {
        break; case ShootDistance::NA: dist = "NA";
        break; case ShootDistance::FAR: dist = "FAR";
        break; case ShootDistance::NEAR: dist = "NEAR";
        break; default:
            throw std::runtime_error("Unexpected ShootDistance: " + std::to_string(i));
        }

        dists.push_back(dist);
    }

    return dists;
}

static const std::vector<std::string> get_meleedistances() {
    auto dists = std::vector<std::string> {};
    std::string dist;

    for (int i=0; i < static_cast<int>(MeleeDistance::_count); i++) {
        switch (MeleeDistance(i)) {
        break; case MeleeDistance::NA: dist = "NA";
        break; case MeleeDistance::FAR: dist = "FAR";
        break; case MeleeDistance::NEAR: dist = "NEAR";
        break; default:
            throw std::runtime_error("Unexpected MeleeDistance: " + std::to_string(i));
        }

        dists.push_back(dist);
    }

    return dists;
}

static const std::vector<std::string> get_hexactions() {
    auto actions = std::vector<std::string> {};
    std::string actname;

    for (int i=0; i < static_cast<int>(HexAction::_count); i++) {
        switch (HexAction(i)) {
        break; case HexAction::AMOVE_TR: actname = "AMOVE_TR";
        break; case HexAction::AMOVE_R: actname = "AMOVE_R";
        break; case HexAction::AMOVE_BR: actname = "AMOVE_BR";
        break; case HexAction::AMOVE_BL: actname = "AMOVE_BL";
        break; case HexAction::AMOVE_L: actname = "AMOVE_L";
        break; case HexAction::AMOVE_TL: actname = "AMOVE_TL";
        break; case HexAction::AMOVE_2TR: actname = "AMOVE_2TR";
        break; case HexAction::AMOVE_2R: actname = "AMOVE_2R";
        break; case HexAction::AMOVE_2BR: actname = "AMOVE_2BR";
        break; case HexAction::AMOVE_2BL: actname = "AMOVE_2BL";
        break; case HexAction::AMOVE_2L: actname = "AMOVE_2L";
        break; case HexAction::AMOVE_2TL: actname = "AMOVE_2TL";
        break; case HexAction::MOVE: actname = "MOVE";
        break; case HexAction::SHOOT: actname = "SHOOT";
        break; default:
            throw std::runtime_error("Unexpected HexAction: " + std::to_string(i));
        }

        actions.push_back(actname);
    }

    return actions;
}

static const std::vector<std::string> get_hexstates() {
    auto states = std::vector<std::string> {};
    std::string statename;

    for (int i=0; i < static_cast<int>(HexState::_count); i++) {
        switch (HexState(i)) {
        break; case HexState::OBSTACLE: statename = "OBSTACLE";
        break; case HexState::OCCUPIED: statename = "OCCUPIED";
        break; case HexState::FREE: statename = "FREE";
        break; default:
            throw std::runtime_error("Unexpected HexState: " + std::to_string(i));
        }

        states.push_back(statename);
    }

    return states;
}

static const std::vector<std::tuple<std::string, std::string, int, int, int>> get_attribute_mapping(std::string global_encoding) {
    // attrname => (encname, offset, n, vmax)
    auto res = std::vector<std::tuple<std::string, std::string, int, int, int>> {};
    int offset = 0;

    for (const auto &[a, e_, n_, vmax] : HEX_ENCODING) {
        auto e = e_;
        auto n = n_;

        std::string attrname;

        switch (a) {
        break; case A::PERCENT_CUR_TO_START_TOTAL_VALUE: attrname = "PERCENT_CUR_TO_START_TOTAL_VALUE";
        break; case A::HEX_Y_COORD: attrname = "HEX_Y_COORD";
        break; case A::HEX_X_COORD: attrname = "HEX_X_COORD";
        break; case A::HEX_STATE: attrname = "HEX_STATE";
        break; case A::HEX_ACTION_MASK_FOR_ACT_STACK: attrname = "HEX_ACTION_MASK_FOR_ACT_STACK";
        break; case A::HEX_ACTION_MASK_FOR_L_STACK_0: attrname = "HEX_ACTION_MASK_FOR_L_STACK_0";
        break; case A::HEX_ACTION_MASK_FOR_L_STACK_1: attrname = "HEX_ACTION_MASK_FOR_L_STACK_1";
        break; case A::HEX_ACTION_MASK_FOR_L_STACK_2: attrname = "HEX_ACTION_MASK_FOR_L_STACK_2";
        break; case A::HEX_ACTION_MASK_FOR_L_STACK_3: attrname = "HEX_ACTION_MASK_FOR_L_STACK_3";
        break; case A::HEX_ACTION_MASK_FOR_L_STACK_4: attrname = "HEX_ACTION_MASK_FOR_L_STACK_4";
        break; case A::HEX_ACTION_MASK_FOR_L_STACK_5: attrname = "HEX_ACTION_MASK_FOR_L_STACK_5";
        break; case A::HEX_ACTION_MASK_FOR_L_STACK_6: attrname = "HEX_ACTION_MASK_FOR_L_STACK_6";
        break; case A::HEX_ACTION_MASK_FOR_R_STACK_0: attrname = "HEX_ACTION_MASK_FOR_R_STACK_0";
        break; case A::HEX_ACTION_MASK_FOR_R_STACK_1: attrname = "HEX_ACTION_MASK_FOR_R_STACK_1";
        break; case A::HEX_ACTION_MASK_FOR_R_STACK_2: attrname = "HEX_ACTION_MASK_FOR_R_STACK_2";
        break; case A::HEX_ACTION_MASK_FOR_R_STACK_3: attrname = "HEX_ACTION_MASK_FOR_R_STACK_3";
        break; case A::HEX_ACTION_MASK_FOR_R_STACK_4: attrname = "HEX_ACTION_MASK_FOR_R_STACK_4";
        break; case A::HEX_ACTION_MASK_FOR_R_STACK_5: attrname = "HEX_ACTION_MASK_FOR_R_STACK_5";
        break; case A::HEX_ACTION_MASK_FOR_R_STACK_6: attrname = "HEX_ACTION_MASK_FOR_R_STACK_6";
        break; case A::HEX_MELEEABLE_BY_ACT_STACK: attrname = "HEX_MELEEABLE_BY_ACT_STACK";
        break; case A::HEX_MELEEABLE_BY_L_STACK_0: attrname = "HEX_MELEEABLE_BY_L_STACK_0";
        break; case A::HEX_MELEEABLE_BY_L_STACK_1: attrname = "HEX_MELEEABLE_BY_L_STACK_1";
        break; case A::HEX_MELEEABLE_BY_L_STACK_2: attrname = "HEX_MELEEABLE_BY_L_STACK_2";
        break; case A::HEX_MELEEABLE_BY_L_STACK_3: attrname = "HEX_MELEEABLE_BY_L_STACK_3";
        break; case A::HEX_MELEEABLE_BY_L_STACK_4: attrname = "HEX_MELEEABLE_BY_L_STACK_4";
        break; case A::HEX_MELEEABLE_BY_L_STACK_5: attrname = "HEX_MELEEABLE_BY_L_STACK_5";
        break; case A::HEX_MELEEABLE_BY_L_STACK_6: attrname = "HEX_MELEEABLE_BY_L_STACK_6";
        break; case A::HEX_MELEEABLE_BY_R_STACK_0: attrname = "HEX_MELEEABLE_BY_R_STACK_0";
        break; case A::HEX_MELEEABLE_BY_R_STACK_1: attrname = "HEX_MELEEABLE_BY_R_STACK_1";
        break; case A::HEX_MELEEABLE_BY_R_STACK_2: attrname = "HEX_MELEEABLE_BY_R_STACK_2";
        break; case A::HEX_MELEEABLE_BY_R_STACK_3: attrname = "HEX_MELEEABLE_BY_R_STACK_3";
        break; case A::HEX_MELEEABLE_BY_R_STACK_4: attrname = "HEX_MELEEABLE_BY_R_STACK_4";
        break; case A::HEX_MELEEABLE_BY_R_STACK_5: attrname = "HEX_MELEEABLE_BY_R_STACK_5";
        break; case A::HEX_MELEEABLE_BY_R_STACK_6: attrname = "HEX_MELEEABLE_BY_R_STACK_6";
        break; case A::HEX_SHOOT_DISTANCE_FROM_ACT_STACK: attrname = "HEX_SHOOT_DISTANCE_FROM_ACT_STACK";
        break; case A::HEX_SHOOT_DISTANCE_FROM_L_STACK_0: attrname = "HEX_SHOOT_DISTANCE_FROM_L_STACK_0";
        break; case A::HEX_SHOOT_DISTANCE_FROM_L_STACK_1: attrname = "HEX_SHOOT_DISTANCE_FROM_L_STACK_1";
        break; case A::HEX_SHOOT_DISTANCE_FROM_L_STACK_2: attrname = "HEX_SHOOT_DISTANCE_FROM_L_STACK_2";
        break; case A::HEX_SHOOT_DISTANCE_FROM_L_STACK_3: attrname = "HEX_SHOOT_DISTANCE_FROM_L_STACK_3";
        break; case A::HEX_SHOOT_DISTANCE_FROM_L_STACK_4: attrname = "HEX_SHOOT_DISTANCE_FROM_L_STACK_4";
        break; case A::HEX_SHOOT_DISTANCE_FROM_L_STACK_5: attrname = "HEX_SHOOT_DISTANCE_FROM_L_STACK_5";
        break; case A::HEX_SHOOT_DISTANCE_FROM_L_STACK_6: attrname = "HEX_SHOOT_DISTANCE_FROM_L_STACK_6";
        break; case A::HEX_SHOOT_DISTANCE_FROM_R_STACK_0: attrname = "HEX_SHOOT_DISTANCE_FROM_R_STACK_0";
        break; case A::HEX_SHOOT_DISTANCE_FROM_R_STACK_1: attrname = "HEX_SHOOT_DISTANCE_FROM_R_STACK_1";
        break; case A::HEX_SHOOT_DISTANCE_FROM_R_STACK_2: attrname = "HEX_SHOOT_DISTANCE_FROM_R_STACK_2";
        break; case A::HEX_SHOOT_DISTANCE_FROM_R_STACK_3: attrname = "HEX_SHOOT_DISTANCE_FROM_R_STACK_3";
        break; case A::HEX_SHOOT_DISTANCE_FROM_R_STACK_4: attrname = "HEX_SHOOT_DISTANCE_FROM_R_STACK_4";
        break; case A::HEX_SHOOT_DISTANCE_FROM_R_STACK_5: attrname = "HEX_SHOOT_DISTANCE_FROM_R_STACK_5";
        break; case A::HEX_SHOOT_DISTANCE_FROM_R_STACK_6: attrname = "HEX_SHOOT_DISTANCE_FROM_R_STACK_6";
        break; case A::HEX_MELEE_DISTANCE_FROM_ACT_STACK: attrname = "HEX_MELEE_DISTANCE_FROM_ACT_STACK";
        break; case A::HEX_MELEE_DISTANCE_FROM_L_STACK_0: attrname = "HEX_MELEE_DISTANCE_FROM_L_STACK_0";
        break; case A::HEX_MELEE_DISTANCE_FROM_L_STACK_1: attrname = "HEX_MELEE_DISTANCE_FROM_L_STACK_1";
        break; case A::HEX_MELEE_DISTANCE_FROM_L_STACK_2: attrname = "HEX_MELEE_DISTANCE_FROM_L_STACK_2";
        break; case A::HEX_MELEE_DISTANCE_FROM_L_STACK_3: attrname = "HEX_MELEE_DISTANCE_FROM_L_STACK_3";
        break; case A::HEX_MELEE_DISTANCE_FROM_L_STACK_4: attrname = "HEX_MELEE_DISTANCE_FROM_L_STACK_4";
        break; case A::HEX_MELEE_DISTANCE_FROM_L_STACK_5: attrname = "HEX_MELEE_DISTANCE_FROM_L_STACK_5";
        break; case A::HEX_MELEE_DISTANCE_FROM_L_STACK_6: attrname = "HEX_MELEE_DISTANCE_FROM_L_STACK_6";
        break; case A::HEX_MELEE_DISTANCE_FROM_R_STACK_0: attrname = "HEX_MELEE_DISTANCE_FROM_R_STACK_0";
        break; case A::HEX_MELEE_DISTANCE_FROM_R_STACK_1: attrname = "HEX_MELEE_DISTANCE_FROM_R_STACK_1";
        break; case A::HEX_MELEE_DISTANCE_FROM_R_STACK_2: attrname = "HEX_MELEE_DISTANCE_FROM_R_STACK_2";
        break; case A::HEX_MELEE_DISTANCE_FROM_R_STACK_3: attrname = "HEX_MELEE_DISTANCE_FROM_R_STACK_3";
        break; case A::HEX_MELEE_DISTANCE_FROM_R_STACK_4: attrname = "HEX_MELEE_DISTANCE_FROM_R_STACK_4";
        break; case A::HEX_MELEE_DISTANCE_FROM_R_STACK_5: attrname = "HEX_MELEE_DISTANCE_FROM_R_STACK_5";
        break; case A::HEX_MELEE_DISTANCE_FROM_R_STACK_6: attrname = "HEX_MELEE_DISTANCE_FROM_R_STACK_6";
        break; case A::STACK_QUANTITY: attrname = "STACK_QUANTITY";
        break; case A::STACK_ATTACK: attrname = "STACK_ATTACK";
        break; case A::STACK_DEFENSE: attrname = "STACK_DEFENSE";
        break; case A::STACK_SHOTS: attrname = "STACK_SHOTS";
        break; case A::STACK_DMG_MIN: attrname = "STACK_DMG_MIN";
        break; case A::STACK_DMG_MAX: attrname = "STACK_DMG_MAX";
        break; case A::STACK_HP: attrname = "STACK_HP";
        break; case A::STACK_HP_LEFT: attrname = "STACK_HP_LEFT";
        break; case A::STACK_SPEED: attrname = "STACK_SPEED";
        break; case A::STACK_WAITED: attrname = "STACK_WAITED";
        break; case A::STACK_QUEUE_POS: attrname = "STACK_QUEUE_POS";
        break; case A::STACK_RETALIATIONS_LEFT: attrname = "STACK_RETALIATIONS_LEFT";
        break; case A::STACK_SIDE: attrname = "STACK_SIDE";
        break; case A::STACK_SLOT: attrname = "STACK_SLOT";
        break; case A::STACK_CREATURE_TYPE: attrname = "STACK_CREATURE_TYPE";
        break; case A::STACK_AI_VALUE_TENTH: attrname = "STACK_AI_VALUE_TENTH";
        break; case A::STACK_IS_ACTIVE: attrname = "STACK_IS_ACTIVE";
        break; case A::STACK_IS_WIDE: attrname = "STACK_IS_WIDE";
        break; case A::STACK_FLYING: attrname = "STACK_FLYING";
        break; case A::STACK_NO_MELEE_PENALTY: attrname = "STACK_NO_MELEE_PENALTY";
        break; case A::STACK_TWO_HEX_ATTACK_BREATH: attrname = "STACK_TWO_HEX_ATTACK_BREATH";
        break; case A::STACK_BLOCKS_RETALIATION: attrname = "STACK_BLOCKS_RETALIATION";
        break; case A::STACK_DEFENSIVE_STANCE: attrname = "STACK_DEFENSIVE_STANCE";
        break; default:
            throw std::runtime_error("Unexpected attribute: " + std::to_string(static_cast<int>(a)));
        }

        std::string encname;

        switch(e) {
        break; case Encoding::NUMERIC: encname = "NUMERIC";
        break; case Encoding::NUMERIC_SQRT: encname = "NUMERIC_SQRT";
        break; case Encoding::BINARY: encname = "BINARY";
        break; case Encoding::CATEGORICAL: encname = "CATEGORICAL";
        break; case Encoding::FLOATING: encname = "FLOATING";
        break; default:
            throw std::runtime_error("Unexpected encoding: " + std::to_string(static_cast<int>(e)));
        }

        res.emplace_back(attrname, encname, offset, n, vmax);
        offset += n;
    }

    return res;
}

// This module is designed to be importable multiple times in the same process
// (not the case with connector, which can't be imported twice in the same PID)
PYBIND11_MODULE(connexport, m) {
        m.def("get_n_actions", &get_n_actions);
        m.def("get_n_nonhex_actions", &get_n_nonhex_actions);
        m.def("get_n_hex_actions", &get_n_hex_actions);
        m.def("get_state_size", &get_state_size);
        m.def("get_state_size_one_hex", &get_state_size_one_hex);
        m.def("get_state_value_na", &get_state_value_na);
        m.def("get_side_left", &get_side_left);
        m.def("get_side_right", &get_side_right);
        m.def("get_dmgmods", &get_dmgmods, "Get a list of the DmgMod enum value names");
        m.def("get_shootdistances", &get_shootdistances, "Get a list of the ShootDistance enum value names");
        m.def("get_meleedistances", &get_meleedistances, "Get a list of the MeleeDistance enum value names");
        m.def("get_hexactions", &get_hexactions, "Get a list of the HexAction enum value names");
        m.def("get_hexstates", &get_hexstates, "Get a list of the HexState enum value names");
        m.def("get_attribute_mapping", &get_attribute_mapping, "Get a attrname => (encname, offset, n, vmax)");
}
