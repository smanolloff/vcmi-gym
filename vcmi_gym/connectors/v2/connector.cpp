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

#include <pybind11/detail/common.h>

#include "connector.h"

namespace Connector::V2 {
    const int Connector::version() {
        return 2;
    }

    PYBIND11_MODULE(connector_v2, m) {
        pybind11::class_<V1::P_State>(m, "P_State")
            .def("get_state", &V1::P_State::get_state)
            .def("get_actmask", &V1::P_State::get_actmask)
            .def("get_attnmask", &V1::P_State::get_attnmask)
            .def("get_errcode", &V1::P_State::get_errcode)
            .def("get_side", &V1::P_State::get_side)
            .def("get_dmg_dealt", &V1::P_State::get_dmg_dealt)
            .def("get_dmg_received", &V1::P_State::get_dmg_received)
            .def("get_units_lost", &V1::P_State::get_units_lost)
            .def("get_units_killed", &V1::P_State::get_units_killed)
            .def("get_value_lost", &V1::P_State::get_value_lost)
            .def("get_value_killed", &V1::P_State::get_value_killed)
            .def("get_side0_army_value", &V1::P_State::get_side0_army_value)
            .def("get_side1_army_value", &V1::P_State::get_side1_army_value)
            .def("get_is_battle_over", &V1::P_State::get_is_battle_over)
            .def("get_is_victorious", &V1::P_State::get_is_victorious);

        pybind11::class_<Connector, std::unique_ptr<Connector>>(m, "Connector")
            .def(pybind11::init<
                const std::string &, // mapname
                const int &,         // seed
                const int &,         // randomHeroes
                const int &,         // randomObstacles
                const int &,         // swapSides
                const std::string &, // loglevelGlobal
                const std::string &, // loglevelAI
                const std::string &, // loglevelStats
                const std::string &, // red
                const std::string &, // blue
                const std::string &, // redModel
                const std::string &, // blueModel
                const std::string &, // statsMode
                const std::string &, // statsStorage
                const int &,         // statsPersistFreq
                const int &,         // statsSampling
                const float &,       // statsScoreVar
                const bool &         // trueRng
            >())
            .def("start", &Connector::start)
            .def("reset", &Connector::reset)
            .def("act", &Connector::act)
            .def("renderAnsi", &Connector::renderAnsi);
    }
}
