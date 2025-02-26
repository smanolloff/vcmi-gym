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

#include "schema/v9/constants.h"
#include "schema/v9/types.h"
#include "exporter.h"

namespace Connector::V9 {
    const int Exporter::getVersion() const { return 9; }
    const int Exporter::getNActions() const { return N_ACTIONS; }
    const int Exporter::getNNonhexActions() const { return N_NONHEX_ACTIONS; }
    const int Exporter::getNHexActions() const { return N_HEX_ACTIONS; }
    const int Exporter::getStateSize() const { return BATTLEFIELD_STATE_SIZE; }
    const int Exporter::getStateSizeOneHex() const { return BATTLEFIELD_STATE_SIZE_ONE_HEX; }
    const int Exporter::getStateSizeAllHexes() const { return BATTLEFIELD_STATE_SIZE_ALL_HEXES; }
    const int Exporter::getStateSizeOnePlayer() const { return BATTLEFIELD_STATE_SIZE_ONE_PLAYER; }
    const int Exporter::getStateSizeGlobal() const { return BATTLEFIELD_STATE_SIZE_GLOBAL; }
    const int Exporter::getStateValueNa() const { return NULL_VALUE_ENCODED; }
    const int Exporter::getSideLeft() const { return static_cast<int>(MMAI::Schema::V9::Side::LEFT); }
    const int Exporter::getSideRight() const { return static_cast<int>(MMAI::Schema::V9::Side::RIGHT); }

    const std::vector<std::string> Exporter::getHexActions() const {
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
            break; case HexAction::MOVE: actname = "MOVE";
            break; case HexAction::SHOOT: actname = "SHOOT";
            break; default:
                throw std::runtime_error("Unexpected HexAction: " + std::to_string(i));
            }

            actions.push_back(actname);
        }

        return actions;
    }

    const std::vector<std::string> Exporter::getHexStates() const {
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
            break; case HA::Y_COORD:                     attrname = "Y_COORD";
            break; case HA::X_COORD:                     attrname = "X_COORD";
            break; case HA::STATE_MASK:                  attrname = "STATE_MASK";
            break; case HA::ACTION_MASK:                 attrname = "ACTION_MASK";
            break; case HA::IS_REAR:                     attrname = "IS_REAR";
            break; case HA::STACK_SIDE:                  attrname = "STACK_SIDE";
            break; case HA::STACK_QUANTITY:              attrname = "STACK_QUANTITY";
            break; case HA::STACK_ATTACK:                attrname = "STACK_ATTACK";
            break; case HA::STACK_DEFENSE:               attrname = "STACK_DEFENSE";
            break; case HA::STACK_SHOTS:                 attrname = "STACK_SHOTS";
            break; case HA::STACK_DMG_MIN:               attrname = "STACK_DMG_MIN";
            break; case HA::STACK_DMG_MAX:               attrname = "STACK_DMG_MAX";
            break; case HA::STACK_HP:                    attrname = "STACK_HP";
            break; case HA::STACK_HP_LEFT:               attrname = "STACK_HP_LEFT";
            break; case HA::STACK_SPEED:                 attrname = "STACK_SPEED";
            break; case HA::STACK_QUEUE_POS:             attrname = "STACK_QUEUE_POS";
            break; case HA::STACK_VALUE_ONE:             attrname = "STACK_VALUE_ONE";
            break; case HA::STACK_FLAGS:                 attrname = "STACK_FLAGS";
            break; case HA::STACK_VALUE_REL:             attrname = "STACK_VALUE_REL";
            break; case HA::STACK_VALUE_REL0:            attrname = "STACK_VALUE_REL0";
            break; case HA::STACK_VALUE_KILLED_REL:      attrname = "STACK_VALUE_KILLED_REL";
            break; case HA::STACK_VALUE_KILLED_ACC_REL0: attrname = "STACK_VALUE_KILLED_ACC_REL0";
            break; case HA::STACK_VALUE_LOST_REL:        attrname = "STACK_VALUE_LOST_REL";
            break; case HA::STACK_VALUE_LOST_ACC_REL0:   attrname = "STACK_VALUE_LOST_ACC_REL0";
            break; case HA::STACK_DMG_DEALT_REL:         attrname = "STACK_DMG_DEALT_REL";
            break; case HA::STACK_DMG_DEALT_ACC_REL0:    attrname = "STACK_DMG_DEALT_ACC_REL0";
            break; case HA::STACK_DMG_RECEIVED_REL:      attrname = "STACK_DMG_RECEIVED_REL";
            break; case HA::STACK_DMG_RECEIVED_ACC_REL0: attrname = "STACK_DMG_RECEIVED_ACC_REL0";
            break; default:
                throw std::runtime_error("Unexpected attribute: " + std::to_string(static_cast<int>(a)));
            }

            auto encname = getEncodingName(e);
            res.emplace_back(attrname, encname, offset, n, vmax);
            offset += n;
        }

        return res;
    }

    const std::vector<FlagMapping> Exporter::getStackFlagMapping() const {
        // attrname => offset
        auto res = std::vector<FlagMapping> {};

        for (int i=0; i<EI(StackFlag::_count); i++) {
            std::string flagname;

            using SF = StackFlag;

            switch (StackFlag(i)) {
            break; case SF::IS_ACTIVE:             flagname = "IS_ACTIVE";
            break; case SF::WILL_ACT:              flagname = "WILL_ACT";
            break; case SF::CAN_WAIT:              flagname = "CAN_WAIT";
            break; case SF::CAN_RETALIATE:         flagname = "CAN_RETALIATE";
            break; case SF::SLEEPING:              flagname = "SLEEPING";
            break; case SF::BLOCKED:               flagname = "BLOCKED";
            break; case SF::BLOCKING:              flagname = "BLOCKING";
            break; case SF::IS_WIDE:               flagname = "IS_WIDE";
            break; case SF::FLYING:                flagname = "FLYING";
            break; case SF::BLIND_LIKE_ATTACK:     flagname = "BLIND_LIKE_ATTACK";
            break; case SF::ADDITIONAL_ATTACK:     flagname = "ADDITIONAL_ATTACK";
            break; case SF::NO_MELEE_PENALTY:      flagname = "NO_MELEE_PENALTY";
            break; case SF::TWO_HEX_ATTACK_BREATH: flagname = "TWO_HEX_ATTACK_BREATH";
            break; case SF::BLOCKS_RETALIATION:    flagname = "BLOCKS_RETALIATION";
            break; default:
                throw std::runtime_error("Unexpected stack flag: " + std::to_string(i));
            }

            res.emplace_back(flagname, i);
        }

        return res;
    }

    const std::vector<AttributeMapping> Exporter::getGlobalAttributeMapping() const {
        // attrname => (encname, offset, n, vmax)
        auto res = std::vector<AttributeMapping> {};
        int offset = 0;

        for (const auto &[a, e_, n_, vmax] : GLOBAL_ENCODING) {
            auto e = e_;
            auto n = n_;

            std::string attrname;

            using GA = GlobalAttribute;
            switch (a) {
            break; case GA::BATTLE_SIDE:                 attrname = "BATTLE_SIDE";
            break; case GA::BATTLE_WINNER:               attrname = "BATTLE_WINNER";
            break; case GA::BFIELD_VALUE_NOW_REL0:       attrname = "BFIELD_VALUE_NOW_REL0";
            break; default:
                throw std::runtime_error("Unexpected attribute: " + std::to_string(static_cast<int>(a)));
            }

            auto encname = getEncodingName(e);
            res.emplace_back(attrname, encname, offset, n, vmax);
            offset += n;
        }

        return res;
    }

    const std::vector<AttributeMapping> Exporter::getPlayerAttributeMapping() const {
        // attrname => (encname, offset, n, vmax)
        auto res = std::vector<AttributeMapping> {};
        int offset = 0;

        for (const auto &[a, e_, n_, vmax] : PLAYER_ENCODING) {
            auto e = e_;
            auto n = n_;

            std::string attrname;

            using PA = PlayerAttribute;
            switch (a) {
            break; case PA::ARMY_VALUE_NOW_REL:     attrname = "ARMY_VALUE_NOW_REL";
            break; case PA::ARMY_VALUE_NOW_REL0:    attrname = "ARMY_VALUE_NOW_REL0";
            break; case PA::VALUE_KILLED_REL:       attrname = "VALUE_KILLED_REL";
            break; case PA::VALUE_KILLED_ACC_REL0:  attrname = "VALUE_KILLED_ACC_REL0";
            break; case PA::VALUE_LOST_REL:         attrname = "VALUE_LOST_REL";
            break; case PA::VALUE_LOST_ACC_REL0:    attrname = "VALUE_LOST_ACC_REL0";
            break; case PA::DMG_DEALT_REL:          attrname = "DMG_DEALT_REL";
            break; case PA::DMG_DEALT_ACC_REL0:     attrname = "DMG_DEALT_ACC_REL0";
            break; case PA::DMG_RECEIVED_REL:       attrname = "DMG_RECEIVED_REL";
            break; case PA::DMG_RECEIVED_ACC_REL0:  attrname = "DMG_RECEIVED_ACC_REL0";
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
        break; case E::EXPNORM_EXPLICIT_NULL:       return "EXPNORM_EXPLICIT_NULL";
        break; case E::EXPNORM_MASKING_NULL:        return "EXPNORM_MASKING_NULL";
        break; case E::EXPNORM_STRICT_NULL:         return "EXPNORM_STRICT_NULL";
        break; case E::EXPNORM_ZERO_NULL:           return "EXPNORM_ZERO_NULL";
        break; case E::LINNORM_EXPLICIT_NULL:       return "LINNORM_EXPLICIT_NULL";
        break; case E::LINNORM_MASKING_NULL:        return "LINNORM_MASKING_NULL";
        break; case E::LINNORM_STRICT_NULL:         return "LINNORM_STRICT_NULL";
        break; case E::LINNORM_ZERO_NULL:           return "LINNORM_ZERO_NULL";
        break; default:
            throw std::runtime_error("Unexpected encoding: " + std::to_string(static_cast<int>(e)));
        }
    }

    PYBIND11_MODULE(exporter_v9, m) {
        pybind11::class_<Exporter>(m, "Exporter")
            .def(pybind11::init<>())
            .def("get_version", &Exporter::getVersion)
            .def("get_n_actions", &Exporter::getNActions)
            .def("get_n_nonhex_actions", &Exporter::getNNonhexActions)
            .def("get_n_hex_actions", &Exporter::getNHexActions)
            .def("get_state_size", &Exporter::getStateSize)
            .def("get_state_size_hexes", &Exporter::getStateSizeAllHexes)
            .def("get_state_size_one_hex", &Exporter::getStateSizeOneHex)
            .def("get_state_size_one_player", &Exporter::getStateSizeOnePlayer)
            .def("get_state_size_global", &Exporter::getStateSizeGlobal)
            .def("get_state_value_na", &Exporter::getStateValueNa)
            .def("get_side_left", &Exporter::getSideLeft)
            .def("get_side_right", &Exporter::getSideRight)
            .def("get_hex_actions", &Exporter::getHexActions, "Get a list of the HexAction enum value names")
            .def("get_hex_states", &Exporter::getHexStates, "Get a list of the HexState enum value names")
            .def("get_hex_attribute_mapping", &Exporter::getHexAttributeMapping, "Get attrname => (encname, offset, n, vmax)")
            .def("get_player_attribute_mapping", &Exporter::getPlayerAttributeMapping, "Get attrname => (encname, offset, n, vmax)")
            .def("get_global_attribute_mapping", &Exporter::getGlobalAttributeMapping, "Get attrname => (encname, offset, n, vmax)")
            .def("get_stack_flag_mapping", &Exporter::getStackFlagMapping, "Get flagname => offset");
    }
}
