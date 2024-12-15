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

#include "schema/v5/constants.h"
#include "schema/v5/types.h"
#include "exporter.h"

namespace Connector::V5 {
    const int Exporter::getVersion() const { return 5; }
    const int Exporter::getNActions() const { return N_ACTIONS; }
    const int Exporter::getStateSize() const { return BATTLEFIELD_STATE_SIZE; }
    const int Exporter::getStateSizeMisc() const { return BATTLEFIELD_STATE_SIZE_MISC; }
    const int Exporter::getStateSizeOneHex() const { return BATTLEFIELD_STATE_SIZE_ONE_HEX; }
    const int Exporter::getStateSizeAllHexes() const { return BATTLEFIELD_STATE_SIZE_ALL_HEXES; }
    const int Exporter::getStateSizeOneStack() const { return BATTLEFIELD_STATE_SIZE_ONE_STACK; }
    const int Exporter::getStateSizeAllStacks() const { return BATTLEFIELD_STATE_SIZE_ALL_STACKS; }
    const int Exporter::getStateValueNa() const { return NULL_VALUE_ENCODED; }
    const int Exporter::getSideLeft() const { return static_cast<int>(MMAI::Schema::V5::Side::LEFT); }
    const int Exporter::getSideRight() const { return static_cast<int>(MMAI::Schema::V5::Side::RIGHT); }

    const std::vector<std::string> Exporter::getPrimaryActions() const {
        auto actions = std::vector<std::string> {};
        std::string actname;

        using PA = PrimaryAction;

        for (int i=0; i < static_cast<int>(PrimaryAction::_count); i++) {
            switch (PrimaryAction(i)) {
            break; case PA::RETREAT:  actname = "RETREAT";
            break; case PA::WAIT:     actname = "WAIT";
            break; case PA::MOVE:     actname = "MOVE";
            break; case PA::ATTACK_0: actname = "ATTACK_0";
            break; case PA::ATTACK_1: actname = "ATTACK_1";
            break; case PA::ATTACK_2: actname = "ATTACK_2";
            break; case PA::ATTACK_3: actname = "ATTACK_3";
            break; case PA::ATTACK_4: actname = "ATTACK_4";
            break; case PA::ATTACK_5: actname = "ATTACK_5";
            break; case PA::ATTACK_6: actname = "ATTACK_6";
            break; case PA::ATTACK_7: actname = "ATTACK_7";
            break; case PA::ATTACK_8: actname = "ATTACK_8";
            break; case PA::ATTACK_9: actname = "ATTACK_9";
            break; default:
                throw std::runtime_error("Unexpected AMoveAction: " + std::to_string(i));
            }

            actions.push_back(actname);
        }

        return actions;
    }

    const std::vector<std::string> Exporter::getAMoveActions() const {
        auto actions = std::vector<std::string> {};
        std::string actname;

        using AMA = AMoveAction;

        for (int i=0; i < static_cast<int>(AMoveAction::_count); i++) {
            switch (AMoveAction(i)) {
            break; case AMA::MOVE_ONLY:         actname = "MOVE_ONLY";
            break; case AMA::MOVE_AND_ATTACK_0: actname = "MOVE_AND_ATTACK_0";
            break; case AMA::MOVE_AND_ATTACK_1: actname = "MOVE_AND_ATTACK_1";
            break; case AMA::MOVE_AND_ATTACK_2: actname = "MOVE_AND_ATTACK_2";
            break; case AMA::MOVE_AND_ATTACK_3: actname = "MOVE_AND_ATTACK_3";
            break; case AMA::MOVE_AND_ATTACK_4: actname = "MOVE_AND_ATTACK_4";
            break; case AMA::MOVE_AND_ATTACK_5: actname = "MOVE_AND_ATTACK_5";
            break; case AMA::MOVE_AND_ATTACK_6: actname = "MOVE_AND_ATTACK_6";
            break; case AMA::MOVE_AND_ATTACK_7: actname = "MOVE_AND_ATTACK_7";
            break; case AMA::MOVE_AND_ATTACK_8: actname = "MOVE_AND_ATTACK_8";
            break; case AMA::MOVE_AND_ATTACK_9: actname = "MOVE_AND_ATTACK_9";
            break; default:
                throw std::runtime_error("Unexpected AMoveAction: " + std::to_string(i));
            }

            actions.push_back(actname);
        }

        return actions;
    }

    const std::vector<std::string> Exporter::getHexStates() const {
        auto states = std::vector<std::string> {};
        std::string statename;

        using HS = HexState;

        for (int i=0; i < static_cast<int>(HexState::_count); i++) {
            switch (HexState(i)) {
            break; case HS::PASSABLE: statename = "PASSABLE";
            break; case HS::STOPPING: statename = "STOPPING";
            break; case HS::DAMAGING_L: statename = "DAMAGING_L";
            break; case HS::DAMAGING_R: statename = "DAMAGING_R";
            break; case HS::_count:
            break; default:
                throw std::runtime_error("Unexpected HexState: " + std::to_string(i));
            }

            states.push_back(statename);
        }

        return states;
    }

    const std::vector<AttributeMapping> Exporter::getMiscAttributeMapping() const {
        // attrname => (encname, offset, n, vmax)
        auto res = std::vector<AttributeMapping> {};
        int offset = 0;

        for (const auto &[a, e_, n_, vmax] : MISC_ENCODING) {
            auto e = e_;
            auto n = n_;

            std::string attrname;

            using MA = MiscAttribute;
            switch (a) {
            break; case MA::PRIMARY_ACTION_MASK:      attrname = "PRIMARY_ACTION_MASK";
            break; case MA::SHOOTING:                 attrname = "SHOOTING";
            break; case MA::INITIAL_ARMY_VALUE_LEFT:  attrname = "INITIAL_ARMY_VALUE_LEFT";
            break; case MA::INITIAL_ARMY_VALUE_RIGHT: attrname = "INITIAL_ARMY_VALUE_RIGHT";
            break; case MA::CURRENT_ARMY_VALUE_LEFT:  attrname = "CURRENT_ARMY_VALUE_LEFT";
            break; case MA::CURRENT_ARMY_VALUE_RIGHT: attrname = "CURRENT_ARMY_VALUE_RIGHT";
            break; default:
                throw std::runtime_error("Unexpected MiscAttribute: " + std::to_string(static_cast<int>(a)));
            }

            auto encname = getEncodingName(e);
            res.emplace_back(attrname, encname, offset, n, vmax);
            offset += n;
        }

        return res;
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
            break; case HA::STACK_ID:       attrname = "STACK_ID";
            break; case HA::STACK_SIDE:     attrname = "STACK_SIDE";
            break; case HA::ACTION_MASK:    attrname = "ACTION_MASK";
            break; default:
                throw std::runtime_error("Unexpected HexAttribute: " + std::to_string(static_cast<int>(a)));
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
            break; case SA::ID:        attrname = "ID";
            break; case SA::SIDE:      attrname = "SIDE";
            break; case SA::Y_COORD:   attrname = "Y_COORD";
            break; case SA::X_COORD:   attrname = "X_COORD";
            break; case SA::CREATURE_ID: attrname = "CREATURE_ID";
            break; case SA::QUANTITY:  attrname = "QUANTITY";
            break; case SA::ATTACK:    attrname = "ATTACK";
            break; case SA::DEFENSE:   attrname = "DEFENSE";
            break; case SA::SHOTS:     attrname = "SHOTS";
            break; case SA::DMG_MIN:   attrname = "DMG_MIN";
            break; case SA::DMG_MAX:   attrname = "DMG_MAX";
            break; case SA::HP:        attrname = "HP";
            break; case SA::HP_LEFT:   attrname = "HP_LEFT";
            break; case SA::SPEED:     attrname = "SPEED";
            break; case SA::QUEUE_POS: attrname = "QUEUE_POS";
            break; case SA::AI_VALUE:  attrname = "AI_VALUE";
            break; case SA::FLAGS:     attrname = "FLAGS";
            break; default:
                throw std::runtime_error("Unexpected StackAttribute: " + std::to_string(static_cast<int>(a)));
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

    PYBIND11_MODULE(exporter_v5, m) {
        pybind11::class_<Exporter>(m, "Exporter")
            .def(pybind11::init<>())
            .def("get_version", &Exporter::getVersion)
            .def("get_n_actions", &Exporter::getNActions)
            .def("get_state_size", &Exporter::getStateSize)
            .def("get_state_size_misc", &Exporter::getStateSizeMisc)
            .def("get_state_size_hexes", &Exporter::getStateSizeAllHexes)
            .def("get_state_size_one_hex", &Exporter::getStateSizeOneHex)
            .def("get_state_size_stacks", &Exporter::getStateSizeAllStacks)
            .def("get_state_size_one_stack", &Exporter::getStateSizeOneStack)
            .def("get_state_value_na", &Exporter::getStateValueNa)
            .def("get_side_left", &Exporter::getSideLeft)
            .def("get_side_right", &Exporter::getSideRight)
            .def("get_primary_actions", &Exporter::getPrimaryActions, "Get a list of the PrimaryAction enum value names")
            .def("get_amove_actions", &Exporter::getAMoveActions, "Get a list of the AMoveAction enum value names")
            .def("get_hex_states", &Exporter::getHexStates, "Get a list of the HexState enum value names")
            .def("get_misc_attribute_mapping", &Exporter::getMiscAttributeMapping, "Get attrname => (encname, offset, n, vmax)")
            .def("get_hex_attribute_mapping", &Exporter::getHexAttributeMapping, "Get attrname => (encname, offset, n, vmax)")
            .def("get_stack_attribute_mapping", &Exporter::getStackAttributeMapping, "Get attrname => (encname, offset, n, vmax)")
            .def("get_stack_flag_mapping", &Exporter::getStackFlagMapping, "Get flagname => offset");
    }
}
