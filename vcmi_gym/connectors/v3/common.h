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

#pragma once

#include <filesystem>
#include <cstdio>
#include <iostream>
#include <thread>
#include <filesystem>

#include "AI/MMAI/schema/base.h"
#include "AI/MMAI/schema/v3/types.h"

#include <pybind11/numpy.h>

#define VERBOSE true  // no effect on release builds

#ifdef DEBUG_BUILD
    #define LOG(msg) if(VERBOSE) { std::cout << "<" << std::this_thread::get_id() << ">[" << std::filesystem::path(__FILE__).filename().string() << "][" << (PyGILState_Check() ? "GIL=1" : "GIL=0") << "] (" << __FUNCTION__ << ") " << msg << "\n"; }
    #define LOGSTR(msg, a1) if (VERBOSE) { std::cout << "<" << std::this_thread::get_id() << ">[" << std::filesystem::path(__FILE__).filename().string() << "][" << (PyGILState_Check() ? "GIL=1" : "GIL=0") << "] (" << __FUNCTION__ << ") " << msg << a1 << "\n"; }
#else
    #define LOG(msg) // noop
    #define LOGSTR(msg, a1) // noop
#endif

namespace Connector::V3 {
    namespace py = pybind11;

    using P_BattlefieldState = py::array_t<float>;
    using P_ActionMask = py::array_t<bool>;
    using P_AttentionMask = py::array_t<float>;

    class P_State {
    public:
        P_State(
            MMAI::Schema::V3::ISupplementaryData::Type type_,
            P_BattlefieldState state_,
            P_ActionMask actmask_,
            P_AttentionMask attnmask_,
            const MMAI::Schema::V3::ErrorCode errcode_,
            const MMAI::Schema::V3::Side side_,
            const int dmg_dealt_,
            const int dmg_received_,
            const int units_lost_,
            const int units_killed_,
            const int value_lost_,
            const int value_killed_,
            const int initial_side0_army_value_,
            const int initial_side1_army_value_,
            const int current_side0_army_value_,
            const int current_side1_army_value_,
            const bool is_battle_over_,
            const bool is_victorious_,
            const std::string ansiRender_
        ) : type(type_)
          , errcode(static_cast<int>(errcode_))
          , actmask(actmask_)
          , attnmask(attnmask_)
          , state(state_)
          , side(static_cast<int>(side_))
          , dmg_dealt(dmg_dealt_)
          , dmg_received(dmg_received_)
          , is_battle_over(is_battle_over_)
          , units_lost(units_lost_)
          , units_killed(units_killed_)
          , value_lost(value_lost_)
          , value_killed(value_killed_)
          , initial_side0_army_value(initial_side0_army_value_)
          , initial_side1_army_value(initial_side1_army_value_)
          , current_side0_army_value(current_side0_army_value_)
          , current_side1_army_value(current_side1_army_value_)
          , is_victorious(is_victorious_)
          , ansiRender(ansiRender_) {}

        const MMAI::Schema::V3::ISupplementaryData::Type type;
        const P_BattlefieldState state;
        const P_ActionMask actmask;
        const P_AttentionMask attnmask;
        const int errcode;
        const int side;
        const int dmg_dealt;
        const int dmg_received;
        const int units_lost;
        const int units_killed;
        const int value_lost;
        const int value_killed;
        const int initial_side0_army_value;
        const int initial_side1_army_value;
        const int current_side0_army_value;
        const int current_side1_army_value;
        const bool is_battle_over;
        const bool is_victorious;
        const std::string ansiRender;

        const P_BattlefieldState get_state() const { return state; }
        const P_ActionMask get_actmask() const { return actmask; }
        const P_AttentionMask get_attnmask() const { return attnmask; }
        const int &get_errcode() const { return errcode; }
        const int &get_side() const { return side; }
        const int &get_dmg_dealt() const { return dmg_dealt; }
        const int &get_dmg_received() const { return dmg_received; }
        const int &get_units_lost() const { return units_lost; }
        const int &get_units_killed() const { return units_killed; }
        const int &get_value_lost() const { return value_lost; }
        const int &get_value_killed() const { return value_killed; }
        const int &get_initial_side0_army_value() const { return initial_side0_army_value; }
        const int &get_initial_side1_army_value() const { return initial_side1_army_value; }
        const int &get_current_side0_army_value() const { return current_side0_army_value; }
        const int &get_current_side1_army_value() const { return current_side1_army_value; }
        const bool &get_is_battle_over() const { return is_battle_over; }
        const bool &get_is_victorious() const { return is_victorious; }
    };
}
