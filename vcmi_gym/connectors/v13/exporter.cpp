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

#include "exporter.h"

namespace Connector::V13 {
    const int Exporter::getVersion() const { return 12; }
    const int Exporter::getNActions() const { return N_ACTIONS; }
    const int Exporter::getNNonhexActions() const { return N_NONHEX_ACTIONS; }
    const int Exporter::getNHexActions() const { return N_HEX_ACTIONS; }
    const int Exporter::getStateSize() const { return BATTLEFIELD_STATE_SIZE; }
    const int Exporter::getStateSizeOneHex() const { return BATTLEFIELD_STATE_SIZE_ONE_HEX; }
    const int Exporter::getStateSizeAllHexes() const { return BATTLEFIELD_STATE_SIZE_ALL_HEXES; }
    const int Exporter::getStateSizeOnePlayer() const { return BATTLEFIELD_STATE_SIZE_ONE_PLAYER; }
    const int Exporter::getStateSizeGlobal() const { return BATTLEFIELD_STATE_SIZE_GLOBAL; }
    const int Exporter::getStateValueNa() const { return NULL_VALUE_ENCODED; }
    const int Exporter::getSideLeft() const { return static_cast<int>(MMAI::Schema::Side::LEFT); }
    const int Exporter::getSideRight() const { return static_cast<int>(MMAI::Schema::Side::RIGHT); }

    const std::vector<std::string> Exporter::getGlobalActions() const {
        auto actions = std::vector<std::string> {};
        std::string actname;

        for (int i=0; i < static_cast<int>(GlobalAction::_count); i++) {
            switch (GlobalAction(i)) {
            break; case GlobalAction::RETREAT: actname = "RETREAT";
            break; case GlobalAction::WAIT: actname = "WAIT";
            break; default:
                throw std::runtime_error("Unexpected HexAction: " + std::to_string(i));
            }

            actions.push_back(actname);
        }

        return actions;
    }

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

        for (const auto &[a, e_, n_, vmax, slope_] : HEX_ENCODING) {
            auto e = e_;
            auto n = n_;
            auto slope = slope_;

            std::string attrname;

            using HA = HexAttribute;
            switch (a) {
            break; case HA::Y_COORD:                     attrname = "Y_COORD";
            break; case HA::X_COORD:                     attrname = "X_COORD";
            break; case HA::STATE_MASK:                  attrname = "STATE_MASK";
            break; case HA::ACTION_MASK:                 attrname = "ACTION_MASK";
            break; case HA::IS_REAR:                     attrname = "IS_REAR";
            break; case HA::STACK_SIDE:                  attrname = "STACK_SIDE";
            break; case HA::STACK_SLOT:                  attrname = "STACK_SLOT";
            break; case HA::STACK_QUANTITY:              attrname = "STACK_QUANTITY";
            // break; case HA::STACK_QUANTITY_BINS:         attrname = "STACK_QUANTITY_BINS";
            break; case HA::STACK_ATTACK:                attrname = "STACK_ATTACK";
            // break; case HA::STACK_ATTACK_BINS:           attrname = "STACK_ATTACK_BINS";
            break; case HA::STACK_DEFENSE:               attrname = "STACK_DEFENSE";
            // break; case HA::STACK_DEFENSE_BINS:          attrname = "STACK_DEFENSE_BINS";
            break; case HA::STACK_SHOTS:                 attrname = "STACK_SHOTS";
            break; case HA::STACK_DMG_MIN:               attrname = "STACK_DMG_MIN";
            // break; case HA::STACK_DMG_MIN_BINS:          attrname = "STACK_DMG_MIN_BINS";
            break; case HA::STACK_DMG_MAX:               attrname = "STACK_DMG_MAX";
            // break; case HA::STACK_DMG_MAX_BINS:          attrname = "STACK_DMG_MAX_BINS";
            break; case HA::STACK_HP:                    attrname = "STACK_HP";
            // break; case HA::STACK_HP_BINS:               attrname = "STACK_HP_BINS";
            break; case HA::STACK_HP_LEFT:               attrname = "STACK_HP_LEFT";
            // break; case HA::STACK_HP_LEFT_REL:           attrname = "STACK_HP_LEFT_REL";
            break; case HA::STACK_SPEED:                 attrname = "STACK_SPEED";
            break; case HA::STACK_QUEUE:                 attrname = "STACK_QUEUE";
            break; case HA::STACK_VALUE_ONE:             attrname = "STACK_VALUE_ONE";
            // break; case HA::STACK_VALUE_ONE_BINS:        attrname = "STACK_VALUE_ONE_BINS";
            break; case HA::STACK_FLAGS1:                attrname = "STACK_FLAGS1";
            break; case HA::STACK_FLAGS2:                attrname = "STACK_FLAGS2";
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
            res.emplace_back(attrname, encname, offset, n, vmax, slope);
            offset += n;
        }

        return res;
    }

    const std::vector<FlagMapping> Exporter::getStackFlag1Mapping() const {
        // attrname => offset
        auto res = std::vector<FlagMapping> {};

        for (int i=0; i<EI(StackFlag1::_count); i++) {
            std::string flagname;

            using SF1 = StackFlag1;

            switch (StackFlag1(i)) {
            break; case SF1::IS_ACTIVE:                 flagname = "IS_ACTIVE";
            break; case SF1::WILL_ACT:                  flagname = "WILL_ACT";
            break; case SF1::CAN_WAIT:                  flagname = "CAN_WAIT";
            break; case SF1::CAN_RETALIATE:             flagname = "CAN_RETALIATE";
            break; case SF1::SLEEPING:                  flagname = "SLEEPING";
            break; case SF1::BLOCKED:                   flagname = "BLOCKED";
            break; case SF1::BLOCKING:                  flagname = "BLOCKING";
            break; case SF1::IS_WIDE:                   flagname = "IS_WIDE";
            break; case SF1::FLYING:                    flagname = "FLYING";
            break; case SF1::ADDITIONAL_ATTACK:         flagname = "ADDITIONAL_ATTACK";
            break; case SF1::NO_MELEE_PENALTY:          flagname = "NO_MELEE_PENALTY";
            break; case SF1::TWO_HEX_ATTACK_BREATH:     flagname = "TWO_HEX_ATTACK_BREATH";
            break; case SF1::BLOCKS_RETALIATION:        flagname = "BLOCKS_RETALIATION";
            break; case SF1::SHOOTER:                   flagname = "SHOOTER";
            break; case SF1::NON_LIVING:                flagname = "NON_LIVING";
            break; case SF1::WAR_MACHINE:               flagname = "WAR_MACHINE";
            break; case SF1::FIREBALL:                  flagname = "FIREBALL";
            break; case SF1::DEATH_CLOUD:               flagname = "DEATH_CLOUD";
            break; case SF1::THREE_HEADED_ATTACK:       flagname = "THREE_HEADED_ATTACK";
            break; case SF1::ALL_AROUND_ATTACK:         flagname = "ALL_AROUND_ATTACK";
            break; case SF1::RETURN_AFTER_STRIKE:       flagname = "RETURN_AFTER_STRIKE";
            break; case SF1::ENEMY_DEFENCE_REDUCTION:   flagname = "ENEMY_DEFENCE_REDUCTION";
            break; case SF1::LIFE_DRAIN:                flagname = "LIFE_DRAIN";
            break; case SF1::DOUBLE_DAMAGE_CHANCE:      flagname = "DOUBLE_DAMAGE_CHANCE";
            break; case SF1::DEATH_STARE:               flagname = "DEATH_STARE";
            break; default:
                throw std::runtime_error("Unexpected stack flag: " + std::to_string(i));
            }

            res.emplace_back(flagname, i);
        }

        return res;
    }

    const std::vector<FlagMapping> Exporter::getStackFlag2Mapping() const {
        // attrname => offset
        auto res = std::vector<FlagMapping> {};

        for (int i=0; i<EI(StackFlag2::_count); i++) {
            std::string flagname;

            using SF2 = StackFlag2;

            switch (StackFlag2(i)) {
            break; case SF2::AGE:               flagname = "AGE";
            break; case SF2::AGE_ATTACK:        flagname = "AGE_ATTACK";
            break; case SF2::BIND:              flagname = "BIND";
            break; case SF2::BIND_ATTACK:       flagname = "BIND_ATTACK";
            break; case SF2::BLIND:             flagname = "BLIND";
            break; case SF2::BLIND_ATTACK:      flagname = "BLIND_ATTACK";
            break; case SF2::CURSE:             flagname = "CURSE";
            break; case SF2::CURSE_ATTACK:      flagname = "CURSE_ATTACK";
            break; case SF2::DISPEL_ATTACK:     flagname = "DISPEL_ATTACK";
            break; case SF2::PETRIFY:           flagname = "PETRIFY";
            break; case SF2::PETRIFY_ATTACK:    flagname = "PETRIFY_ATTACK";
            break; case SF2::POISON:            flagname = "POISON";
            break; case SF2::POISON_ATTACK:     flagname = "POISON_ATTACK";
            break; case SF2::WEAKNESS:          flagname = "WEAKNESS";
            break; case SF2::WEAKNESS_ATTACK:   flagname = "WEAKNESS_ATTACK";
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

        for (const auto &[a, e_, n_, vmax, slope_] : GLOBAL_ENCODING) {
            auto e = e_;
            auto n = n_;
            auto slope = slope_;

            std::string attrname;

            using GA = GlobalAttribute;
            switch (a) {
            break; case GA::BATTLE_SIDE:                 attrname = "BATTLE_SIDE";
            break; case GA::BATTLE_SIDE_ACTIVE_PLAYER:   attrname = "BATTLE_SIDE_ACTIVE_PLAYER";
            break; case GA::BATTLE_WINNER:               attrname = "BATTLE_WINNER";
            break; case GA::BFIELD_VALUE_START_ABS:      attrname = "BFIELD_VALUE_START_ABS";
            // break; case GA::BFIELD_VALUE_START_ABS_BINS: attrname = "BFIELD_VALUE_START_ABS_BINS";
            break; case GA::BFIELD_VALUE_NOW_ABS:        attrname = "BFIELD_VALUE_NOW_ABS";
            // break; case GA::BFIELD_VALUE_NOW_ABS_BINS:   attrname = "BFIELD_VALUE_NOW_ABS_BINS";
            break; case GA::BFIELD_VALUE_NOW_REL0:       attrname = "BFIELD_VALUE_NOW_REL0";
            break; case GA::BFIELD_HP_START_ABS:         attrname = "BFIELD_HP_START_ABS";
            // break; case GA::BFIELD_HP_START_ABS_BINS:    attrname = "BFIELD_HP_START_ABS_BINS";
            break; case GA::BFIELD_HP_NOW_ABS:           attrname = "BFIELD_HP_NOW_ABS";
            // break; case GA::BFIELD_HP_NOW_ABS_BINS:      attrname = "BFIELD_HP_NOW_ABS_BINS";
            break; case GA::BFIELD_HP_NOW_REL0:          attrname = "BFIELD_HP_NOW_REL0";
            break; case GA::ACTION_MASK:                 attrname = "ACTION_MASK";
            break; default:
                throw std::runtime_error("Unexpected attribute: " + std::to_string(static_cast<int>(a)));
            }

            auto encname = getEncodingName(e);
            res.emplace_back(attrname, encname, offset, n, vmax, slope);
            offset += n;
        }

        return res;
    }

    const std::vector<AttributeMapping> Exporter::getPlayerAttributeMapping() const {
        // attrname => (encname, offset, n, vmax)
        auto res = std::vector<AttributeMapping> {};
        int offset = 0;

        for (const auto &[a, e_, n_, vmax, slope_] : PLAYER_ENCODING) {
            auto e = e_;
            auto n = n_;
            auto slope = slope_;

            std::string attrname;

            using PA = PlayerAttribute;
            switch (a) {
            break; case PA::BATTLE_SIDE:                attrname = "BATTLE_SIDE";
            break; case PA::ARMY_VALUE_NOW_ABS:         attrname = "ARMY_VALUE_NOW_ABS";
            // break; case PA::ARMY_VALUE_NOW_ABS_BINS:    attrname = "ARMY_VALUE_NOW_ABS_BINS";
            break; case PA::ARMY_VALUE_NOW_REL:         attrname = "ARMY_VALUE_NOW_REL";
            break; case PA::ARMY_VALUE_NOW_REL0:        attrname = "ARMY_VALUE_NOW_REL0";
            break; case PA::ARMY_HP_NOW_ABS:            attrname = "ARMY_HP_NOW_ABS";
            // break; case PA::ARMY_HP_NOW_ABS_BINS:       attrname = "ARMY_HP_NOW_ABS_BINS";
            break; case PA::ARMY_HP_NOW_REL:            attrname = "ARMY_HP_NOW_REL";
            break; case PA::ARMY_HP_NOW_REL0:           attrname = "ARMY_HP_NOW_REL0";
            break; case PA::VALUE_KILLED_NOW_ABS:       attrname = "VALUE_KILLED_NOW_ABS";
            // break; case PA::VALUE_KILLED_NOW_ABS_BINS:  attrname = "VALUE_KILLED_NOW_ABS_BINS";
            break; case PA::VALUE_KILLED_NOW_REL:       attrname = "VALUE_KILLED_NOW_REL";
            break; case PA::VALUE_KILLED_ACC_ABS:       attrname = "VALUE_KILLED_ACC_ABS";
            // break; case PA::VALUE_KILLED_ACC_ABS_BINS:  attrname = "VALUE_KILLED_ACC_ABS_BINS";
            break; case PA::VALUE_KILLED_ACC_REL0:      attrname = "VALUE_KILLED_ACC_REL0";
            break; case PA::VALUE_LOST_NOW_ABS:         attrname = "VALUE_LOST_NOW_ABS";
            // break; case PA::VALUE_LOST_NOW_ABS_BINS:    attrname = "VALUE_LOST_NOW_ABS_BINS";
            break; case PA::VALUE_LOST_NOW_REL:         attrname = "VALUE_LOST_NOW_REL";
            break; case PA::VALUE_LOST_ACC_ABS:         attrname = "VALUE_LOST_ACC_ABS";
            // break; case PA::VALUE_LOST_ACC_ABS_BINS:    attrname = "VALUE_LOST_ACC_ABS_BINS";
            break; case PA::VALUE_LOST_ACC_REL0:        attrname = "VALUE_LOST_ACC_REL0";
            break; case PA::DMG_DEALT_NOW_ABS:          attrname = "DMG_DEALT_NOW_ABS";
            // break; case PA::DMG_DEALT_NOW_ABS_BINS:     attrname = "DMG_DEALT_NOW_ABS_BINS";
            break; case PA::DMG_DEALT_NOW_REL:          attrname = "DMG_DEALT_NOW_REL";
            break; case PA::DMG_DEALT_ACC_ABS:          attrname = "DMG_DEALT_ACC_ABS";
            // break; case PA::DMG_DEALT_ACC_ABS_BINS:     attrname = "DMG_DEALT_ACC_ABS_BINS";
            break; case PA::DMG_DEALT_ACC_REL0:         attrname = "DMG_DEALT_ACC_REL0";
            break; case PA::DMG_RECEIVED_NOW_ABS:       attrname = "DMG_RECEIVED_NOW_ABS";
            // break; case PA::DMG_RECEIVED_NOW_ABS_BINS:  attrname = "DMG_RECEIVED_NOW_ABS_BINS";
            break; case PA::DMG_RECEIVED_NOW_REL:       attrname = "DMG_RECEIVED_NOW_REL";
            break; case PA::DMG_RECEIVED_ACC_ABS:       attrname = "DMG_RECEIVED_ACC_ABS";
            // break; case PA::DMG_RECEIVED_ACC_ABS_BINS:  attrname = "DMG_RECEIVED_ACC_ABS_BINS";
            break; case PA::DMG_RECEIVED_ACC_REL0:      attrname = "DMG_RECEIVED_ACC_REL0";
            break; default:
                throw std::runtime_error("Unexpected attribute: " + std::to_string(static_cast<int>(a)));
            }

            auto encname = getEncodingName(e);
            res.emplace_back(attrname, encname, offset, n, vmax, slope);
            offset += n;
        }

        return res;
    }

    const std::string Exporter::getEncodingName(Encoding e) const {
        using E = Encoding;
        switch(e) {
        break; case E::ACCUMULATING_EXPLICIT_NULL:          return "ACCUMULATING_EXPLICIT_NULL";
        break; case E::ACCUMULATING_IMPLICIT_NULL:          return "ACCUMULATING_IMPLICIT_NULL";
        break; case E::ACCUMULATING_MASKING_NULL:           return "ACCUMULATING_MASKING_NULL";
        break; case E::ACCUMULATING_STRICT_NULL:            return "ACCUMULATING_STRICT_NULL";
        break; case E::ACCUMULATING_ZERO_NULL:              return "ACCUMULATING_ZERO_NULL";
        break; case E::BINARY_EXPLICIT_NULL:                return "BINARY_EXPLICIT_NULL";
        break; case E::BINARY_MASKING_NULL:                 return "BINARY_MASKING_NULL";
        break; case E::BINARY_STRICT_NULL:                  return "BINARY_STRICT_NULL";
        break; case E::BINARY_ZERO_NULL:                    return "BINARY_ZERO_NULL";
        break; case E::CATEGORICAL_EXPLICIT_NULL:           return "CATEGORICAL_EXPLICIT_NULL";
        break; case E::CATEGORICAL_IMPLICIT_NULL:           return "CATEGORICAL_IMPLICIT_NULL";
        break; case E::CATEGORICAL_MASKING_NULL:            return "CATEGORICAL_MASKING_NULL";
        break; case E::CATEGORICAL_STRICT_NULL:             return "CATEGORICAL_STRICT_NULL";
        break; case E::CATEGORICAL_ZERO_NULL:               return "CATEGORICAL_ZERO_NULL";
        break; case E::EXPNORM_EXPLICIT_NULL:               return "EXPNORM_EXPLICIT_NULL";
        break; case E::EXPNORM_MASKING_NULL:                return "EXPNORM_MASKING_NULL";
        break; case E::EXPNORM_STRICT_NULL:                 return "EXPNORM_STRICT_NULL";
        break; case E::EXPNORM_ZERO_NULL:                   return "EXPNORM_ZERO_NULL";
        break; case E::LINNORM_EXPLICIT_NULL:               return "LINNORM_EXPLICIT_NULL";
        break; case E::LINNORM_MASKING_NULL:                return "LINNORM_MASKING_NULL";
        break; case E::LINNORM_STRICT_NULL:                 return "LINNORM_STRICT_NULL";
        break; case E::LINNORM_ZERO_NULL:                   return "LINNORM_ZERO_NULL";
        break; case E::RAW:                                 return "RAW";
        break; default:
            throw std::runtime_error("Unexpected encoding: " + std::to_string(static_cast<int>(e)));
        }
    }

    const std::vector<std::string> Exporter::getLinkTypes() const {
        auto res = std::vector<std::string> {};
        std::string type;

        for (int i=0; i < static_cast<int>(LinkType::_count); i++) {
            switch (LinkType(i)) {
            break; case LinkType::ADJACENT:         type = "ADJACENT";
            break; case LinkType::REACH:            type = "REACH";
            break; case LinkType::RANGED_MOD:       type = "RANGED_MOD";
            break; case LinkType::ACTS_BEFORE:      type = "ACTS_BEFORE";
            break; case LinkType::MELEE_DMG_REL:    type = "MELEE_DMG_REL";
            break; case LinkType::RETAL_DMG_REL:    type = "RETAL_DMG_REL";
            break; case LinkType::RANGED_DMG_REL:   type = "RANGED_DMG_REL";
            break; default:
                throw std::runtime_error("Unexpected LinkType: " + std::to_string(i));
            }

            res.push_back(type);
        }

        return res;
    }

    PYBIND11_MODULE(exporter_v13, m) {
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
            .def("get_global_actions", &Exporter::getGlobalActions, "Get a list of the GlobalAction enum value names")
            .def("get_hex_actions", &Exporter::getHexActions, "Get a list of the HexAction enum value names")
            .def("get_hex_states", &Exporter::getHexStates, "Get a list of the HexState enum value names")
            .def("get_hex_attribute_mapping", &Exporter::getHexAttributeMapping, "Get attrname => (encname, offset, n, vmax, slope)")
            .def("get_player_attribute_mapping", &Exporter::getPlayerAttributeMapping, "Get attrname => (encname, offset, n, vmax, slope)")
            .def("get_global_attribute_mapping", &Exporter::getGlobalAttributeMapping, "Get attrname => (encname, offset, n, vmax, slope)")
            .def("get_stack_flag1_mapping", &Exporter::getStackFlag1Mapping, "Get flagname => offset")
            .def("get_stack_flag2_mapping", &Exporter::getStackFlag2Mapping, "Get flagname => offset")
            .def("get_link_types", &Exporter::getLinkTypes, "Get a list of LinkType enum value names");
    }
}
