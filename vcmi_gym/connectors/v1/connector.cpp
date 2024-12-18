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

#include "connector.h"
#include "schema/v1/constants.h"
#include "schema/v1/types.h"
#include "ML/MLClient.h"
#include "ML/model_wrappers/function.h"
#include "ML/model_wrappers/scripted.h"
#include "ML/model_wrappers/torchpath.h"
#include <algorithm>
#include <pybind11/detail/common.h>
#include <stdexcept>
#include <string>

namespace Connector::V1 {
    Connector::Connector(
        const std::string mapname_,
        const int seed_,
        const int randomHeroes_,
        const int randomObstacles_,
        const int swapSides_,
        const std::string loglevelGlobal_,
        const std::string loglevelAI_,
        const std::string loglevelStats_,
        const std::string red_,
        const std::string blue_,
        const std::string redModel_,
        const std::string blueModel_,
        const std::string statsMode_,
        const std::string statsStorage_,
        const int statsPersistFreq_,
        const int statsSampling_,
        const float statsScoreVar_,
        const bool trueRng_
    ) : mapname(mapname_),
        seed(seed_),
        randomHeroes(randomHeroes_),
        randomObstacles(randomObstacles_),
        swapSides(swapSides_),
        loglevelGlobal(loglevelGlobal_),
        loglevelAI(loglevelAI_),
        loglevelStats(loglevelStats_),
        red(red_),
        blue(blue_),
        redModel(redModel_),
        blueModel(blueModel_),
        statsMode(statsMode_),
        statsStorage(statsStorage_),
        statsPersistFreq(statsPersistFreq_),
        statsSampling(statsSampling_),
        statsScoreVar(statsScoreVar_),
        trueRng(trueRng_) {};

    const int Connector::version() {
        return 1;
    }

    const MMAI::Schema::V1::ISupplementaryData* Connector::extractSupplementaryData(const MMAI::Schema::IState *s) {
        // LOG("Extracting supplementary data...");
        // auto &any = s->getSupplementaryData();
        // if(!any.has_value()) throw std::runtime_error("supdata is empty");
        // auto &t = typeid(const MMAI::Schema::V1::ISupplementaryData*);
        // if(any.type() != t) LOG(
        //     std::string("Bad std::any payload type from getSupplementaryData()") \
        //     + ": want: " + t.name() + "/" + std::to_string(t.hash_code()) \
        //     + ", have: " + any.type().name() + "/" + std::to_string(any.type().hash_code())
        // );

        return std::any_cast<const MMAI::Schema::V1::ISupplementaryData*>(s->getSupplementaryData());
    };

    const P_State Connector::convertState(const MMAI::Schema::IState* s) {
        LOG("Convert IState -> P_State");

        auto &bs = s->getBattlefieldState();
        P_BattlefieldState pbs(bs.size());
        auto pbsmd = pbs.mutable_data();
        for (int i=0; i<bs.size(); i++)
            pbsmd[i] = bs[i];

        auto &actmask = s->getActionMask();

        auto pam = P_ActionMask(actmask.size());
        auto pammd = pam.mutable_data();
        for (int i=0; i < actmask.size(); i++)
            pammd[i] = actmask[i];

        auto &attnmask = s->getAttentionMask();
        auto patm = P_AttentionMask(attnmask.size());
        auto patmmd = patm.mutable_data();
        for (int i=0; i < attnmask.size(); i++)
            patmmd[i] = attnmask[i];

        // XXX: these do not improve performance, better avoid the const_cast
        // auto pbs = P_BattlefieldState(bs.size(), const_cast<float*>(bs.data()));
        // auto patm = P_AttentionMask(attnmask.size(), const_cast<float*>(attnmask.data()));

        auto sup = extractSupplementaryData(s);
        assert(sup->getType() == MMAI::Schema::V1::ISupplementaryData::Type::REGULAR);

        auto res = P_State(
             sup->getType(), pbs, pam, patm, sup->getErrorCode(), sup->getSide(),
             sup->getDmgDealt(), sup->getDmgReceived(),
             sup->getUnitsLost(), sup->getUnitsKilled(),
             sup->getValueLost(), sup->getValueKilled(),
             sup->getSide0ArmyValue(), sup->getSide1ArmyValue(),
             sup->getIsBattleEnded(), sup->getIsVictorious(), sup->getAnsiRender()
        );

        return res;
    }

    const P_State Connector::reset() {
        assert(connstate == ConnectorState::AWAITING_ACTION);

        std::unique_lock lock(m);
        LOG("obtain lock: done");

        LOGSTR("set this->action = ", std::to_string(MMAI::Schema::ACTION_RESET));
        action = MMAI::Schema::ACTION_RESET;

        LOG("set state = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOG("cond.notify_one()");
        cond.notify_one();

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOG("cond.wait(lock)");
            cond.wait(lock);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
            // py::gil_scoped_acquire acquire2;
        }

        assert(connstate == ConnectorState::AWAITING_ACTION);

        LOG("release lock (return)");
        LOG("return P_State");
        const auto pstate = convertState(state);
        assert(pstate.type == MMAI::Schema::V1::ISupplementaryData::Type::REGULAR);
        return pstate;
    }

    const std::string Connector::renderAnsi() {
        assert(connstate == ConnectorState::AWAITING_ACTION);

        LOG("obtain lock");
        std::unique_lock lock(m);
        LOG("obtain lock: done");

        LOGSTR("set this->action = ", std::to_string(MMAI::Schema::ACTION_RENDER_ANSI));
        action = MMAI::Schema::ACTION_RENDER_ANSI;

        LOG("set connstate = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOG("cond.notify_one()");
        cond.notify_one();

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOG("cond.wait(lock)");
            cond.wait(lock);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
        }

        assert(connstate == ConnectorState::AWAITING_ACTION);
        auto sup = extractSupplementaryData(state);
        assert(sup->getType() == MMAI::Schema::V1::ISupplementaryData::Type::ANSI_RENDER);

        LOG("release lock (return)");
        LOG("return state->ansiRender");
        return sup->getAnsiRender();
    }

    const P_State Connector::act(MMAI::Schema::Action a) {
        assert(connstate == ConnectorState::AWAITING_ACTION);

        // Prevent control actions via `step`
        assert(a > 0);

        LOG("obtain lock");
        std::unique_lock lock(m);
        LOG("obtain lock: done");

        LOGSTR("set this->action = ", std::to_string(a));
        action = a;

        LOG("set connstate = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOG("cond.notify_one()");
        cond.notify_one();

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOG("cond.wait(lock)");
            cond.wait(lock);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
        }

        assert(connstate == ConnectorState::AWAITING_ACTION);

        LOG("release lock (return)");
        LOG("return P_State");
        return convertState(state);
    }

    const P_State Connector::start() {
        assert(connstate == ConnectorState::NEW);
        printf("VCMI Connector v%d initialized\n", version());

        setvbuf(stdout, NULL, _IONBF, 0);
        LOG("start");

        LOG("obtain lock");
        std::unique_lock lock(m);
        LOG("obtain lock: done");

        auto f_getAction = [this](const MMAI::Schema::IState* s) {
            return this->getAction(s);
        };

        auto f_getValueDummy = [](const MMAI::Schema::IState* s) {
            throw std::runtime_error("getValue not implemented in connector");
            return 0.0;
        };

        std::function<int(const MMAI::Schema::IState* s)> getActionRed;
        std::function<int(const MMAI::Schema::IState* s)> getActionBlue;
        std::function<double(const MMAI::Schema::IState* s)> getValueRed;
        std::function<double(const MMAI::Schema::IState* s)> getValueBlue;

        if (red == "MMAI_USER") {
            leftModel = new ML::ModelWrappers::Function(version(), "MMAI_MODEL", f_getAction, f_getValueDummy);
        } else if (red == "MMAI_MODEL") {
            // BAI will load the actual model based on leftModel->getName()
            leftModel = new ML::ModelWrappers::TorchPath(redModel);
        } else {
            leftModel = new ML::ModelWrappers::Scripted(redModel);
        }

        if (blue == "MMAI_USER") {
            rightModel = new ML::ModelWrappers::Function(version(), "MMAI_MODEL", f_getAction, f_getValueDummy);
        } else if (blue == "MMAI_MODEL") {
            // BAI will load the actual model based on leftModel->getName()
            rightModel = new ML::ModelWrappers::TorchPath(blueModel);
        } else {
            rightModel = new ML::ModelWrappers::Scripted(blueModel);
        }

        // auto oldcwd = std::filesystem::current_path();

        // This must happen in the main thread (SDL requires it)
        auto initargs = ML::InitArgs(
            mapname,            // mapname
            leftModel,          // leftModel
            rightModel,         // rightModel
            0,                  // maxBattles
            seed,               // seed
            randomHeroes,       // randomHeroes
            randomObstacles,    // randomObstacles
            0,                  // townChance
            0,                  // warmachineChance
            0,                  // tightFormationChance
            0,                  // randomTerrainChance
            "",                 // battlefieldPattern
            0,                  // manaMin
            0,                  // manaMax
            swapSides,          // swapSides
            loglevelGlobal,     // loglevelGlobal
            loglevelAI,         // loglevelAI
            loglevelStats,      // loglevelStats
            statsMode,          // statsMode
            statsStorage,       // statsStorage
            60000,              // statsTimeout
            statsPersistFreq,   // statsPersistFreq
            true                // headless (disable the GUI, as it cannot run in a non-main thread)
        );

        LOG("call init_vcmi(...)");
        init_vcmi(initargs);

        LOG("set connstate = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOG("launch new thread");
        vcmithread = std::thread([] {
            LOG("[thread] Start VCMI");
            ML::start_vcmi();
            assert(false); // should never happen
        });

        // LOG("detach the newly created thread...")
        // vcmithread.detach();

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOG("cond.wait(lock)");
            cond.wait(lock);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
            // py::gil_scoped_acquire acquire2;
        }

        // auto newcwd = std::filesystem::current_path();
        // std::cout << "OLDCWD: " << oldcwd << "\nNEWCWD: " << newcwd << "\n";

        // NOTE: changing CWD here *sometimes* fails with exception:
        // std::__1::ios_base::failure: could not open file: unspecified iostream_category error
        // (sometimes = fails on benchmark, works on test...)
        //
        // LOGSTR("Change cwd back to", oldcwd.string());
        // std::filesystem::current_path(oldcwd);

        assert(connstate == ConnectorState::AWAITING_ACTION);

        LOG("release lock (return)");
        LOG("return P_Result");

        return convertState(state);
    }

    MMAI::Schema::Action Connector::getAction(const MMAI::Schema::IState* s) {
        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("obtain lock");
        std::unique_lock lock(m);
        LOG("obtain lock: done");

        assert(connstate == ConnectorState::AWAITING_STATE);

        LOG("set this->istate = s");
        state = s;

        LOG("set connstate = AWAITING_ACTION");
        connstate = ConnectorState::AWAITING_ACTION;

        LOG("cond.notify_one()");
        cond.notify_one();

        assert(connstate == ConnectorState::AWAITING_ACTION);

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            // Now wait again (will unblock once step/reset have been called)
            LOG("cond.wait(lock)");
            cond.wait(lock);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
            // py::gil_scoped_acquire acquire2;
        }

        assert(connstate == ConnectorState::AWAITING_STATE);

        LOG("release lock (return)");
        LOGSTR("return Action: ", std::to_string(action));
        return action;
    }

    PYBIND11_MODULE(connector_v1, m) {
        py::class_<P_State>(m, "P_State")
            .def("get_state", &P_State::get_state)
            .def("get_actmask", &P_State::get_actmask)
            .def("get_attnmask", &P_State::get_attnmask)
            .def("get_errcode", &P_State::get_errcode)
            .def("get_side", &P_State::get_side)
            .def("get_dmg_dealt", &P_State::get_dmg_dealt)
            .def("get_dmg_received", &P_State::get_dmg_received)
            .def("get_units_lost", &P_State::get_units_lost)
            .def("get_units_killed", &P_State::get_units_killed)
            .def("get_value_lost", &P_State::get_value_lost)
            .def("get_value_killed", &P_State::get_value_killed)
            .def("get_side0_army_value", &P_State::get_side0_army_value)
            .def("get_side1_army_value", &P_State::get_side1_army_value)
            .def("get_is_battle_over", &P_State::get_is_battle_over)
            .def("get_is_victorious", &P_State::get_is_victorious);

        py::class_<Connector, std::unique_ptr<Connector>>(m, "Connector")
            .def(py::init<
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
