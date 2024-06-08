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
#include "mmai_export.h"
#include "myclient.h"
#include <stdexcept>

Connector::Connector(
    const std::string encoding_,
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
) : encoding(encoding_),
    mapname(mapname_),
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
    trueRng(trueRng_),
    baggage(std::make_unique<MMAI::Export::Baggage>(initBaggage())) {}

MMAI::Export::Baggage Connector::initBaggage() {
    return MMAI::Export::Baggage([this](const MMAI::Export::Result* r) {
        return this->getAction(r);
    });
}

const P_Result Connector::convertResult(const MMAI::Export::Result* r) {
    LOG("Convert Result -> P_Result");

    P_State ps;

    if (encoding == MMAI::Export::STATE_ENCODING_DEFAULT) {
        auto vec = MMAI::Export::State{};
        vec.reserve(MMAI::Export::STATE_SIZE_DEFAULT);

        for (auto &u : r->stateUnencoded)
            u.encode(vec);

        if (vec.size() != MMAI::Export::STATE_SIZE_DEFAULT)
            throw std::runtime_error("STATE_ENCODING_DEFAULT: Unexpected state size: " + std::to_string(vec.size()));

        ps = P_State(MMAI::Export::STATE_SIZE_DEFAULT);
        auto psmd = ps.mutable_data();

        for (int i=0; i<MMAI::Export::STATE_SIZE_DEFAULT; i++)
            psmd[i] = vec[i];
    } else if (encoding == MMAI::Export::STATE_ENCODING_FLOAT) {
        if (r->stateUnencoded.size() != MMAI::Export::STATE_SIZE_FLOAT)
            throw std::runtime_error("STATE_ENCODING_FLOAT: Unexpected state size: " + std::to_string(r->stateUnencoded.size()));

        ps = P_State(MMAI::Export::STATE_SIZE_FLOAT);
        auto psmd = ps.mutable_data();

        for (int i=0; i<MMAI::Export::STATE_SIZE_FLOAT; i++)
            psmd[i] = r->stateUnencoded[i].encode2Floating();
    } else {
        throw std::runtime_error("Unexpected encoding: " + encoding);
    };

    auto pam = P_ActMask(r->actmask.size());
    auto pammd = pam.mutable_data();
    for (int i=0; i < r->actmask.size(); i++)
        pammd[i] = r->actmask[i];

    auto patm = P_AttnMasks(r->attnmasks.size());
    auto patmmd = patm.mutable_data();
    for (int i=0; i < r->attnmasks.size(); i++)
        patmmd[i] = r->attnmasks[i];

    return P_Result(
         r->type, ps, pam, patm, r->errmask, r->side,
         r->dmgDealt, r->dmgReceived,
         r->unitsLost, r->unitsKilled,
         r->valueLost, r->valueKilled,
         r->side0ArmyValue, r->side1ArmyValue,
         r->ended, r->victory, r->ansiRender
    );
}

const P_Result Connector::reset() {
    assert(state == ConnectorState::AWAITING_ACTION);

    std::unique_lock lock(m);
    LOG("obtain lock: done");

    LOGSTR("set this->action = ", std::to_string(MMAI::Export::ACTION_RESET));
    action = MMAI::Export::ACTION_RESET;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

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

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result->type == MMAI::Export::ResultType::REGULAR);

    LOG("release lock (return)");
    LOG("return P_Result");
    return convertResult(result);
}

const std::string Connector::renderAnsi() {
    assert(state == ConnectorState::AWAITING_ACTION);

    LOG("obtain lock");
    std::unique_lock lock(m);
    LOG("obtain lock: done");

    LOGSTR("set this->action = ", std::to_string(MMAI::Export::ACTION_RENDER_ANSI));
    action = MMAI::Export::ACTION_RENDER_ANSI;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

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

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result->type == MMAI::Export::ResultType::ANSI_RENDER);

    LOG("release lock (return)");
    LOG("return P_Result");
    return result->ansiRender;
}

const P_Result Connector::act(MMAI::Export::Action a) {
    assert(state == ConnectorState::AWAITING_ACTION);

    // Prevent control actions via `step`
    assert(a > 0);

    LOG("obtain lock");
    std::unique_lock lock(m);
    LOG("obtain lock: done");

    LOGSTR("set this->action = ", std::to_string(a));
    action = a;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

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

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result->type == MMAI::Export::ResultType::REGULAR);

    LOG("release lock (return)");
    LOG("return P_Result");
    return convertResult(result);
}

const P_Result Connector::start() {
    assert(state == ConnectorState::NEW);

    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    LOG("obtain lock");
    std::unique_lock lock(m);
    LOG("obtain lock: done");

    // auto oldcwd = std::filesystem::current_path();

    // This must happen in the main thread (SDL requires it)
    LOG("call init_vcmi(...)");
    init_vcmi(
        baggage.get(),
        encoding,
        mapname,
        0,  // maxbattles
        seed,
        randomHeroes,
        randomObstacles,
        swapSides,
        loglevelGlobal,
        loglevelAI,
        loglevelStats,
        red,
        blue,
        redModel,
        blueModel,
        statsMode,
        statsStorage,
        statsPersistFreq,
        statsSampling,
        statsScoreVar,
        false,  // printModelPredictions
        trueRng,
        true  // headless (disable the GUI, as it cannot run in a non-main thread)
    );

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("launch new thread");
    vcmithread = std::thread([] {
        LOG("[thread] Start VCMI");
        start_vcmi();
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

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result->type == MMAI::Export::ResultType::REGULAR);

    LOG("release lock (return)");
    LOG("return P_Result");

    return convertResult(result);
}

MMAI::Export::Action Connector::getAction(const MMAI::Export::Result* r) {

    LOG("acquire Python GIL");
    py::gil_scoped_acquire acquire;

    LOG("obtain lock");
    std::unique_lock lock(m);
    LOG("obtain lock: done");

    assert(state == ConnectorState::AWAITING_RESULT);

    LOG("set this->result = r");
    result = r;

    LOG("set state = AWAITING_ACTION");
    state = ConnectorState::AWAITING_ACTION;

    LOG("cond.notify_one()");
    cond.notify_one();

    assert(state == ConnectorState::AWAITING_ACTION);

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

    assert(state == ConnectorState::AWAITING_RESULT);

    LOG("release lock (return)");
    LOGSTR("return Action: ", std::to_string(action));
    return action;
}

PYBIND11_MODULE(connector, m) {
    py::class_<P_Result>(m, "P_Result")
        .def("get_state", &P_Result::get_state)
        .def("get_actmask", &P_Result::get_actmask)
        .def("get_attnmasks", &P_Result::get_attnmasks)
        .def("get_errmask", &P_Result::get_errmask)
        .def("get_side", &P_Result::get_side)
        .def("get_dmg_dealt", &P_Result::get_dmg_dealt)
        .def("get_dmg_received", &P_Result::get_dmg_received)
        .def("get_units_lost", &P_Result::get_units_lost)
        .def("get_units_killed", &P_Result::get_units_killed)
        .def("get_value_lost", &P_Result::get_value_lost)
        .def("get_value_killed", &P_Result::get_value_killed)
        .def("get_side0_army_value", &P_Result::get_side0_army_value)
        .def("get_side1_army_value", &P_Result::get_side1_army_value)
        .def("get_is_battle_over", &P_Result::get_is_battle_over)
        .def("get_is_victorious", &P_Result::get_is_victorious);

    py::class_<Connector, std::unique_ptr<Connector>>(m, "Connector")
        .def(py::init<
            const std::string &, // encoding
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
