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

#include "mmai_export.h" // "vendor" header file
#include "conncommon.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace MMAI::Export;

constexpr int get_n_actions() { return N_ACTIONS; }
constexpr int get_n_nonhex_actions() { return N_NONHEX_ACTIONS; }
constexpr int get_n_hex_actions() { return N_HEX_ACTIONS; }
constexpr int get_state_size_default() { return STATE_SIZE_DEFAULT; }
constexpr int get_state_size_default_one_hex() { return STATE_SIZE_DEFAULT_ONE_HEX; }
constexpr int get_state_size_float() { return STATE_SIZE_FLOAT; }
constexpr int get_state_size_float_one_hex() { return STATE_SIZE_FLOAT_ONE_HEX; }
constexpr int get_state_value_na() { return STATE_VALUE_NA; }
constexpr auto get_encoding_type_default() { return STATE_ENCODING_DEFAULT; }
constexpr auto get_encoding_type_float() { return STATE_ENCODING_FLOAT; }

static const std::map<ErrMask, std::tuple<std::string, std::string>> get_error_mapping() {
    std::map<ErrMask, std::tuple<std::string, std::string>> res;

    for (auto& [_err, tuple] : ERRORS) {
        res[std::get<0>(tuple)] = {std::get<1>(tuple), std::get<2>(tuple)};
    }

    return res;
}

static const std::vector<std::tuple<std::string, std::string, int, int, int>> get_attribute_mapping(std::string global_encoding) {
    // attrname => (encname, offset, n, vmax)
    auto res = std::vector<std::tuple<std::string, std::string, int, int, int>> {};
    int offset = 0;

    for (const auto &[a, e_, n_, vmax] : HEX_ENCODING) {
        auto e = e_;
        auto n = n_;

        if (global_encoding == STATE_ENCODING_FLOAT) {
            e = Encoding::FLOATING;
            n = 1;
        }

        std::string attrname;

        switch(a) {
        break; case Attribute::HEX_Y_COORD: attrname = "HEX_Y_COORD";
        break; case Attribute::HEX_X_COORD: attrname = "HEX_X_COORD";
        break; case Attribute::HEX_STATE: attrname = "HEX_STATE";
        break; case Attribute::HEX_REACHABLE_BY_ACTIVE_STACK: attrname = "HEX_REACHABLE_BY_ACTIVE_STACK";
        break; case Attribute::HEX_REACHABLE_BY_FRIENDLY_STACK_0: attrname = "HEX_REACHABLE_BY_FRIENDLY_STACK_0";
        break; case Attribute::HEX_REACHABLE_BY_FRIENDLY_STACK_1: attrname = "HEX_REACHABLE_BY_FRIENDLY_STACK_1";
        break; case Attribute::HEX_REACHABLE_BY_FRIENDLY_STACK_2: attrname = "HEX_REACHABLE_BY_FRIENDLY_STACK_2";
        break; case Attribute::HEX_REACHABLE_BY_FRIENDLY_STACK_3: attrname = "HEX_REACHABLE_BY_FRIENDLY_STACK_3";
        break; case Attribute::HEX_REACHABLE_BY_FRIENDLY_STACK_4: attrname = "HEX_REACHABLE_BY_FRIENDLY_STACK_4";
        break; case Attribute::HEX_REACHABLE_BY_FRIENDLY_STACK_5: attrname = "HEX_REACHABLE_BY_FRIENDLY_STACK_5";
        break; case Attribute::HEX_REACHABLE_BY_FRIENDLY_STACK_6: attrname = "HEX_REACHABLE_BY_FRIENDLY_STACK_6";
        break; case Attribute::HEX_REACHABLE_BY_ENEMY_STACK_0: attrname = "HEX_REACHABLE_BY_ENEMY_STACK_0";
        break; case Attribute::HEX_REACHABLE_BY_ENEMY_STACK_1: attrname = "HEX_REACHABLE_BY_ENEMY_STACK_1";
        break; case Attribute::HEX_REACHABLE_BY_ENEMY_STACK_2: attrname = "HEX_REACHABLE_BY_ENEMY_STACK_2";
        break; case Attribute::HEX_REACHABLE_BY_ENEMY_STACK_3: attrname = "HEX_REACHABLE_BY_ENEMY_STACK_3";
        break; case Attribute::HEX_REACHABLE_BY_ENEMY_STACK_4: attrname = "HEX_REACHABLE_BY_ENEMY_STACK_4";
        break; case Attribute::HEX_REACHABLE_BY_ENEMY_STACK_5: attrname = "HEX_REACHABLE_BY_ENEMY_STACK_5";
        break; case Attribute::HEX_REACHABLE_BY_ENEMY_STACK_6: attrname = "HEX_REACHABLE_BY_ENEMY_STACK_6";
        break; case Attribute::HEX_MELEEABLE_BY_ACTIVE_STACK: attrname = "HEX_MELEEABLE_BY_ACTIVE_STACK";
        break; case Attribute::HEX_MELEEABLE_BY_FRIENDLY_STACK_0: attrname = "HEX_MELEEABLE_BY_FRIENDLY_STACK_0";
        break; case Attribute::HEX_MELEEABLE_BY_FRIENDLY_STACK_1: attrname = "HEX_MELEEABLE_BY_FRIENDLY_STACK_1";
        break; case Attribute::HEX_MELEEABLE_BY_FRIENDLY_STACK_2: attrname = "HEX_MELEEABLE_BY_FRIENDLY_STACK_2";
        break; case Attribute::HEX_MELEEABLE_BY_FRIENDLY_STACK_3: attrname = "HEX_MELEEABLE_BY_FRIENDLY_STACK_3";
        break; case Attribute::HEX_MELEEABLE_BY_FRIENDLY_STACK_4: attrname = "HEX_MELEEABLE_BY_FRIENDLY_STACK_4";
        break; case Attribute::HEX_MELEEABLE_BY_FRIENDLY_STACK_5: attrname = "HEX_MELEEABLE_BY_FRIENDLY_STACK_5";
        break; case Attribute::HEX_MELEEABLE_BY_FRIENDLY_STACK_6: attrname = "HEX_MELEEABLE_BY_FRIENDLY_STACK_6";
        break; case Attribute::HEX_MELEEABLE_BY_ENEMY_STACK_0: attrname = "HEX_MELEEABLE_BY_ENEMY_STACK_0";
        break; case Attribute::HEX_MELEEABLE_BY_ENEMY_STACK_1: attrname = "HEX_MELEEABLE_BY_ENEMY_STACK_1";
        break; case Attribute::HEX_MELEEABLE_BY_ENEMY_STACK_2: attrname = "HEX_MELEEABLE_BY_ENEMY_STACK_2";
        break; case Attribute::HEX_MELEEABLE_BY_ENEMY_STACK_3: attrname = "HEX_MELEEABLE_BY_ENEMY_STACK_3";
        break; case Attribute::HEX_MELEEABLE_BY_ENEMY_STACK_4: attrname = "HEX_MELEEABLE_BY_ENEMY_STACK_4";
        break; case Attribute::HEX_MELEEABLE_BY_ENEMY_STACK_5: attrname = "HEX_MELEEABLE_BY_ENEMY_STACK_5";
        break; case Attribute::HEX_MELEEABLE_BY_ENEMY_STACK_6: attrname = "HEX_MELEEABLE_BY_ENEMY_STACK_6";
        break; case Attribute::HEX_SHOOTABLE_BY_ACTIVE_STACK: attrname = "HEX_SHOOTABLE_BY_ACTIVE_STACK";
        break; case Attribute::HEX_SHOOTABLE_BY_FRIENDLY_STACK_0: attrname = "HEX_SHOOTABLE_BY_FRIENDLY_STACK_0";
        break; case Attribute::HEX_SHOOTABLE_BY_FRIENDLY_STACK_1: attrname = "HEX_SHOOTABLE_BY_FRIENDLY_STACK_1";
        break; case Attribute::HEX_SHOOTABLE_BY_FRIENDLY_STACK_2: attrname = "HEX_SHOOTABLE_BY_FRIENDLY_STACK_2";
        break; case Attribute::HEX_SHOOTABLE_BY_FRIENDLY_STACK_3: attrname = "HEX_SHOOTABLE_BY_FRIENDLY_STACK_3";
        break; case Attribute::HEX_SHOOTABLE_BY_FRIENDLY_STACK_4: attrname = "HEX_SHOOTABLE_BY_FRIENDLY_STACK_4";
        break; case Attribute::HEX_SHOOTABLE_BY_FRIENDLY_STACK_5: attrname = "HEX_SHOOTABLE_BY_FRIENDLY_STACK_5";
        break; case Attribute::HEX_SHOOTABLE_BY_FRIENDLY_STACK_6: attrname = "HEX_SHOOTABLE_BY_FRIENDLY_STACK_6";
        break; case Attribute::HEX_SHOOTABLE_BY_ENEMY_STACK_0: attrname = "HEX_SHOOTABLE_BY_ENEMY_STACK_0";
        break; case Attribute::HEX_SHOOTABLE_BY_ENEMY_STACK_1: attrname = "HEX_SHOOTABLE_BY_ENEMY_STACK_1";
        break; case Attribute::HEX_SHOOTABLE_BY_ENEMY_STACK_2: attrname = "HEX_SHOOTABLE_BY_ENEMY_STACK_2";
        break; case Attribute::HEX_SHOOTABLE_BY_ENEMY_STACK_3: attrname = "HEX_SHOOTABLE_BY_ENEMY_STACK_3";
        break; case Attribute::HEX_SHOOTABLE_BY_ENEMY_STACK_4: attrname = "HEX_SHOOTABLE_BY_ENEMY_STACK_4";
        break; case Attribute::HEX_SHOOTABLE_BY_ENEMY_STACK_5: attrname = "HEX_SHOOTABLE_BY_ENEMY_STACK_5";
        break; case Attribute::HEX_SHOOTABLE_BY_ENEMY_STACK_6: attrname = "HEX_SHOOTABLE_BY_ENEMY_STACK_6";
        break; case Attribute::HEX_NEXT_TO_ACTIVE_STACK: attrname = "HEX_NEXT_TO_ACTIVE_STACK";
        break; case Attribute::HEX_NEXT_TO_FRIENDLY_STACK_0: attrname = "HEX_NEXT_TO_FRIENDLY_STACK_0";
        break; case Attribute::HEX_NEXT_TO_FRIENDLY_STACK_1: attrname = "HEX_NEXT_TO_FRIENDLY_STACK_1";
        break; case Attribute::HEX_NEXT_TO_FRIENDLY_STACK_2: attrname = "HEX_NEXT_TO_FRIENDLY_STACK_2";
        break; case Attribute::HEX_NEXT_TO_FRIENDLY_STACK_3: attrname = "HEX_NEXT_TO_FRIENDLY_STACK_3";
        break; case Attribute::HEX_NEXT_TO_FRIENDLY_STACK_4: attrname = "HEX_NEXT_TO_FRIENDLY_STACK_4";
        break; case Attribute::HEX_NEXT_TO_FRIENDLY_STACK_5: attrname = "HEX_NEXT_TO_FRIENDLY_STACK_5";
        break; case Attribute::HEX_NEXT_TO_FRIENDLY_STACK_6: attrname = "HEX_NEXT_TO_FRIENDLY_STACK_6";
        break; case Attribute::HEX_NEXT_TO_ENEMY_STACK_0: attrname = "HEX_NEXT_TO_ENEMY_STACK_0";
        break; case Attribute::HEX_NEXT_TO_ENEMY_STACK_1: attrname = "HEX_NEXT_TO_ENEMY_STACK_1";
        break; case Attribute::HEX_NEXT_TO_ENEMY_STACK_2: attrname = "HEX_NEXT_TO_ENEMY_STACK_2";
        break; case Attribute::HEX_NEXT_TO_ENEMY_STACK_3: attrname = "HEX_NEXT_TO_ENEMY_STACK_3";
        break; case Attribute::HEX_NEXT_TO_ENEMY_STACK_4: attrname = "HEX_NEXT_TO_ENEMY_STACK_4";
        break; case Attribute::HEX_NEXT_TO_ENEMY_STACK_5: attrname = "HEX_NEXT_TO_ENEMY_STACK_5";
        break; case Attribute::HEX_NEXT_TO_ENEMY_STACK_6: attrname = "HEX_NEXT_TO_ENEMY_STACK_6";
        break; case Attribute::STACK_QUANTITY: attrname = "STACK_QUANTITY";
        break; case Attribute::STACK_ATTACK: attrname = "STACK_ATTACK";
        break; case Attribute::STACK_DEFENSE: attrname = "STACK_DEFENSE";
        break; case Attribute::STACK_SHOTS: attrname = "STACK_SHOTS";
        break; case Attribute::STACK_DMG_MIN: attrname = "STACK_DMG_MIN";
        break; case Attribute::STACK_DMG_MAX: attrname = "STACK_DMG_MAX";
        break; case Attribute::STACK_HP: attrname = "STACK_HP";
        break; case Attribute::STACK_HP_LEFT: attrname = "STACK_HP_LEFT";
        break; case Attribute::STACK_SPEED: attrname = "STACK_SPEED";
        break; case Attribute::STACK_WAITED: attrname = "STACK_WAITED";
        break; case Attribute::STACK_QUEUE_POS: attrname = "STACK_QUEUE_POS";
        break; case Attribute::STACK_RETALIATIONS_LEFT: attrname = "STACK_RETALIATIONS_LEFT";
        break; case Attribute::STACK_SIDE: attrname = "STACK_SIDE";
        break; case Attribute::STACK_SLOT: attrname = "STACK_SLOT";
        break; case Attribute::STACK_CREATURE_TYPE: attrname = "STACK_CREATURE_TYPE";
        break; case Attribute::STACK_AI_VALUE_TENTH: attrname = "STACK_AI_VALUE_TENTH";
        break; case Attribute::STACK_IS_ACTIVE: attrname = "STACK_IS_ACTIVE";
        break; case Attribute::STACK_IS_WIDE: attrname = "STACK_IS_WIDE";
        break; case Attribute::STACK_FLYING: attrname = "STACK_FLYING";
        break; case Attribute::STACK_NO_MELEE_PENALTY: attrname = "STACK_NO_MELEE_PENALTY";
        break; case Attribute::STACK_TWO_HEX_ATTACK_BREATH: attrname = "STACK_TWO_HEX_ATTACK_BREATH";
        break; case Attribute::STACK_BLOCKS_RETALIATION: attrname = "STACK_BLOCKS_RETALIATION";
        break; case Attribute::STACK_DEFENSIVE_STANCE: attrname = "STACK_DEFENSIVE_STANCE";
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
        m.def("get_state_size_default", &get_state_size_default);
        m.def("get_state_size_default_one_hex", &get_state_size_default_one_hex);
        m.def("get_state_size_float", &get_state_size_float);
        m.def("get_state_size_float_one_hex", &get_state_size_float_one_hex);
        m.def("get_state_value_na", &get_state_value_na);
        m.def("get_encoding_type_default", &get_encoding_type_default);
        m.def("get_encoding_type_float", &get_encoding_type_float);
        m.def("get_error_mapping", &get_error_mapping, "Get available error names and flags");
        m.def("get_attribute_mapping", &get_attribute_mapping, "Get available error names and flags");
}
