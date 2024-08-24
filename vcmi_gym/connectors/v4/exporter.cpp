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

#include "AI/MMAI/schema/v4/constants.h"
#include "AI/MMAI/schema/v4/types.h"
#include "exporter.h"

namespace Connector::V4 {
    const int Exporter::getVersion() const { return 4; }
    const int Exporter::getNActions() const { return N_ACTIONS; }
    const int Exporter::getNNonhexActions() const { return N_NONHEX_ACTIONS; }
    const int Exporter::getNHexActions() const { return N_HEX_ACTIONS; }
    const int Exporter::getStateSize() const { return BATTLEFIELD_STATE_SIZE; }
    const int Exporter::getStateSizeOneHex() const { return BATTLEFIELD_STATE_SIZE_ONE_HEX; }
    const int Exporter::getStateSizeAllHexes() const { return BATTLEFIELD_STATE_SIZE_ALL_HEXES; }
    const int Exporter::getStateSizeOneStack() const { return BATTLEFIELD_STATE_SIZE_ONE_STACK; }
    const int Exporter::getStateSizeAllStacks() const { return BATTLEFIELD_STATE_SIZE_ALL_STACKS; }
    const int Exporter::getStateValueNa() const { return NULL_VALUE_ENCODED; }
    const int Exporter::getSideLeft() const { return static_cast<int>(MMAI::Schema::V4::Side::LEFT); }
    const int Exporter::getSideRight() const { return static_cast<int>(MMAI::Schema::V4::Side::RIGHT); }

    const std::vector<std::string> Exporter::getHexactions() const {
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

    const std::vector<std::string> Exporter::getHexstates() const {
        auto states = std::vector<std::string> {};
        std::string statename;

        for (int i=0; i < static_cast<int>(HexState::_count); i++) {
            switch (HexState(i)) {
            break; case HexState::PASSABLE: statename = "PASSABLE";
            break; case HexState::STOPPING: statename = "STOPPING";
            break; case HexState::DAMAGING_L: statename = "DAMAGING_L";
            break; case HexState::DAMAGING_R: statename = "DAMAGING_R";
            break; case HexState::_count:
            break; default:
                throw std::runtime_error("Unexpected HexState: " + std::to_string(i));
            }

            states.push_back(statename);
        }

        return states;
    }

    const std::vector<AttributeMapping> Exporter::getHexAttributeMapping() const {
        // attrname => (encname, offset, n, vmax)
        auto res = std::vector<AttributeMapping> {};
        int offset = 0;

        for (const auto &[a, e_, n_, vmax] : HEX_ENCODING) {
            auto e = e_;
            auto n = n_;

            std::string attrname;

            using HA = HexAttribute;
            switch (a) {
            break; case HA::Y_COORD:        attrname = "Y_COORD";
            break; case HA::X_COORD:        attrname = "X_COORD";
            break; case HA::STATE_MASK:     attrname = "STATE_MASK";
            break; case HA::ACTION_MASK:    attrname = "ACTION_MASK";
            break; case HA::STACK_ID:       attrname = "STACK_ID";
            break; default:
                throw std::runtime_error("Unexpected attribute: " + std::to_string(static_cast<int>(a)));
            }

            auto encname = getEncodingName(e);
            res.emplace_back(attrname, encname, offset, n, vmax);
            offset += n;
        }

        return res;
    }


    const std::vector<AttributeMapping> Exporter::getStackAttributeMapping() const {
        // attrname => (encname, offset, n, vmax)
        auto res = std::vector<AttributeMapping> {};
        int offset = 0;

        for (const auto &[a, e_, n_, vmax] : STACK_ENCODING) {
            auto e = e_;
            auto n = n_;

            std::string attrname;

            using SA = StackAttribute;
            switch (a) {
            break; case SA::ID: attrname = "ID";
            break; case SA::Y_COORD: attrname = "Y_COORD";
            break; case SA::X_COORD: attrname = "X_COORD";
            break; case SA::SIDE: attrname = "SIDE";
            break; case SA::QUANTITY: attrname = "QUANTITY";
            break; case SA::ATTACK: attrname = "ATTACK";
            break; case SA::DEFENSE: attrname = "DEFENSE";
            break; case SA::SHOTS: attrname = "SHOTS";
            break; case SA::DMG_MIN: attrname = "DMG_MIN";
            break; case SA::DMG_MAX: attrname = "DMG_MAX";
            break; case SA::HP: attrname = "HP";
            break; case SA::HP_LEFT: attrname = "HP_LEFT";
            break; case SA::SPEED: attrname = "SPEED";
            break; case SA::WAITED: attrname = "WAITED";
            break; case SA::QUEUE_POS: attrname = "QUEUE_POS";
            break; case SA::RETALIATIONS_LEFT: attrname = "RETALIATIONS_LEFT";
            break; case SA::IS_WIDE: attrname = "IS_WIDE";
            break; case SA::AI_VALUE: attrname = "AI_VALUE";
            break; case SA::ACTED: attrname = "ACTED";
            break; case SA::SLEEPING: attrname = "SLEEPING";
            break; case SA::BLOCKED: attrname = "BLOCKED";
            break; case SA::BLOCKING: attrname = "BLOCKING";
            break; case SA::ESTIMATED_DMG: attrname = "ESTIMATED_DMG";
            break; case SA::FLYING: attrname = "FLYING";
            break; case SA::BLIND_LIKE_ATTACK: attrname = "BLIND_LIKE_ATTACK";
            break; case SA::ADDITIONAL_ATTACK: attrname = "ADDITIONAL_ATTACK";
            break; case SA::NO_MELEE_PENALTY: attrname = "NO_MELEE_PENALTY";
            break; case SA::TWO_HEX_ATTACK_BREATH: attrname = "TWO_HEX_ATTACK_BREATH";
            break; case SA::BLOCKS_RETALIATION: attrname = "BLOCKS_RETALIATION";
            break; default:
                throw std::runtime_error("Unexpected attribute: " + std::to_string(static_cast<int>(a)));
            }

            auto encname = getEncodingName(e);
            res.emplace_back(attrname, encname, offset, n, vmax);
            offset += n;
        }

        return res;
    }

    const std::string Exporter::getEncodingName(Encoding e) const {
        using E = Encoding;
        switch(e) {
        break; case E::ACCUMULATING_EXPLICIT_NULL:  return "ACCUMULATING_EXPLICIT_NULL";
        break; case E::ACCUMULATING_IMPLICIT_NULL:  return "ACCUMULATING_IMPLICIT_NULL";
        break; case E::ACCUMULATING_MASKING_NULL:   return "ACCUMULATING_MASKING_NULL";
        break; case E::ACCUMULATING_STRICT_NULL:    return "ACCUMULATING_STRICT_NULL";
        break; case E::ACCUMULATING_ZERO_NULL:      return "ACCUMULATING_ZERO_NULL";
        break; case E::BINARY_EXPLICIT_NULL:        return "BINARY_EXPLICIT_NULL";
        break; case E::BINARY_MASKING_NULL:         return "BINARY_MASKING_NULL";
        break; case E::BINARY_STRICT_NULL:          return "BINARY_STRICT_NULL";
        break; case E::BINARY_ZERO_NULL:            return "BINARY_ZERO_NULL";
        break; case E::CATEGORICAL_EXPLICIT_NULL:   return "CATEGORICAL_EXPLICIT_NULL";
        break; case E::CATEGORICAL_IMPLICIT_NULL:   return "CATEGORICAL_IMPLICIT_NULL";
        break; case E::CATEGORICAL_MASKING_NULL:    return "CATEGORICAL_MASKING_NULL";
        break; case E::CATEGORICAL_STRICT_NULL:     return "CATEGORICAL_STRICT_NULL";
        break; case E::CATEGORICAL_ZERO_NULL:       return "CATEGORICAL_ZERO_NULL";
        break; case E::NORMALIZED_EXPLICIT_NULL:    return "NORMALIZED_EXPLICIT_NULL";
        break; case E::NORMALIZED_MASKING_NULL:     return "NORMALIZED_MASKING_NULL";
        break; case E::NORMALIZED_STRICT_NULL:      return "NORMALIZED_STRICT_NULL";
        break; case E::NORMALIZED_ZERO_NULL:        return "NORMALIZED_ZERO_NULL";
        break; default:
            throw std::runtime_error("Unexpected encoding: " + std::to_string(static_cast<int>(e)));
        }
    }

    PYBIND11_MODULE(exporter_v4, m) {
        pybind11::class_<Exporter>(m, "Exporter")
            .def(pybind11::init<>())
            .def("get_version", &Exporter::getVersion)
            .def("get_n_actions", &Exporter::getNActions)
            .def("get_n_nonhex_actions", &Exporter::getNNonhexActions)
            .def("get_n_hex_actions", &Exporter::getNHexActions)
            .def("get_state_size", &Exporter::getStateSize)
            .def("get_state_size_hexes", &Exporter::getStateSizeAllHexes)
            .def("get_state_size_one_hex", &Exporter::getStateSizeOneHex)
            .def("get_state_size_stacks", &Exporter::getStateSizeAllStacks)
            .def("get_state_size_one_stack", &Exporter::getStateSizeOneStack)
            .def("get_state_value_na", &Exporter::getStateValueNa)
            .def("get_side_left", &Exporter::getSideLeft)
            .def("get_side_right", &Exporter::getSideRight)
            .def("get_hexactions", &Exporter::getHexactions, "Get a list of the HexAction enum value names")
            .def("get_hexstates", &Exporter::getHexstates, "Get a list of the HexState enum value names")
            .def("get_hex_attribute_mapping", &Exporter::getHexAttributeMapping, "Get a attrname => (encname, offset, n, vmax)")
            .def("get_stack_attribute_mapping", &Exporter::getStackAttributeMapping, "Get a attrname => (encname, offset, n, vmax)");
    }
}
