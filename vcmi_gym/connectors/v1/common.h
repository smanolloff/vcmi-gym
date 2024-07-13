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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <filesystem>

#include "AI/MMAI/schema/base.h"
#include "AI/MMAI/schema/v1/types.h"

#define VERBOSE true  // no effect on release builds

#ifdef DEBUG_BUILD
    #define LOG(msg) if(VERBOSE) { std::cout << "<" << std::this_thread::get_id() << ">[" << std::filesystem::path(__FILE__).filename().string() << "][" << (PyGILState_Check() ? "GIL=1" : "GIL=0") << "] (" << __FUNCTION__ << ") " << msg << "\n"; }
    #define LOGSTR(msg, a1) if (VERBOSE) { std::cout << "<" << std::this_thread::get_id() << ">[" << std::filesystem::path(__FILE__).filename().string() << "][" << (PyGILState_Check() ? "GIL=1" : "GIL=0") << "] (" << __FUNCTION__ << ") " << msg << a1 << "\n"; }
#else
    #define LOG(msg) // noop
    #define LOGSTR(msg, a1) // noop
#endif

namespace Connector::V1 {
    namespace py = pybind11;

    using P_BattlefieldState = py::array_t<float>;
    using P_ActionMask = py::array_t<bool>;
    using P_AttentionMask = py::array_t<float>;

    class P_State {
    public:
        P_State(
            MMAI::Schema::V1::ISupplementaryData::Type type_,
            P_BattlefieldState state_,
            P_ActionMask actmask_,
            P_AttentionMask attnmask_,
            MMAI::Schema::V1::ErrorCode errcode_,
            MMAI::Schema::V1::Side side_,
            int dmg_dealt_,
            int dmg_received_,
            int units_lost_,
            int units_killed_,
            int value_lost_,
            int value_killed_,
            int side0_army_value_,
            int side1_army_value_,
            bool is_battle_over_,
            bool is_victorious_,
            std::string ansiRender_
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
          , side0_army_value(side0_army_value_)
          , side1_army_value(side1_army_value_)
          , is_victorious(is_victorious_)
          , ansiRender(ansiRender_) {}

        const MMAI::Schema::V1::ISupplementaryData::Type type;
        const py::array_t<float> state;
        const py::array_t<bool> actmask;
        const py::array_t<float> attnmask;
        const int errcode;
        const int side;
        const int dmg_dealt;
        const int dmg_received;
        const int units_lost;
        const int units_killed;
        const int value_lost;
        const int value_killed;
        const int side0_army_value;
        const int side1_army_value;
        const bool is_battle_over;
        const bool is_victorious;
        const std::string ansiRender;

        const py::array_t<float> &get_state() const { return state; }
        const py::array_t<bool> &get_actmask() const { return actmask; }
        const py::array_t<float> &get_attnmask() const { return attnmask; }
        const int &get_errcode() const { return errcode; }
        const int &get_side() const { return side; }
        const int &get_dmg_dealt() const { return dmg_dealt; }
        const int &get_dmg_received() const { return dmg_received; }
        const int &get_units_lost() const { return units_lost; }
        const int &get_units_killed() const { return units_killed; }
        const int &get_value_lost() const { return value_lost; }
        const int &get_value_killed() const { return value_killed; }
        const int &get_side0_army_value() const { return side0_army_value; }
        const int &get_side1_army_value() const { return side1_army_value; }
        const bool &get_is_battle_over() const { return is_battle_over; }
        const bool &get_is_victorious() const { return is_victorious; }
    };
}
