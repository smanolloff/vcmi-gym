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

#include "threadconnector.h"
#include "AI/MMAI/schema/v4/constants.h"
#include "AI/MMAI/schema/v4/types.h"
#include "ML/MLClient.h"
#include "ML/model_wrappers/function.h"
#include "ML/model_wrappers/scripted.h"
#include "MMAILoader/TorchModel.h"
#include "exporter.h"
#include <algorithm>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <thread>

#define ASSERT_STATE(id, want) { \
    if((want) != (connstate)) \
        throw VCMIConnectorException(std::string(id) + ": unexpected connector state: want: " + std::to_string(EI(want)) + ", have: " + std::to_string(EI(connstate))); \
}

namespace Connector::V4::Thread {
    // Python does not know about some threads and exceptions thrown there
    // result in abrupt program termination.
    // => use this to set a member var `_error`
    //    (the exception must be constructed and thrown in python-aware thread)
    void Connector::setError(std::string msg) {
        std::cerr << "ERROR: " << msg;
        _error = msg;
    }

    void Connector::maybeThrowError() {
        if (_error.empty()) return;
        // if (_shutdown) return;
        auto msg = _error;
        _error = "";
        throw VCMIConnectorException(msg);
    }

    int Connector::_cond_wait(const char* funcname, int id, std::condition_variable &cond, std::unique_lock<std::mutex> &l, int timeoutSeconds, std::function<bool()> &checker) {
        int res;
        int i = 0;
        auto start = std::chrono::high_resolution_clock::now();

        while (true) {
            // LOGFMT("[%s] cond%d.wait/2 ...", funcname % id);
            auto fres = checker();
            // LOGFMT("[%s] cond%d.wait/2 -> %d", funcname % id % static_cast<int>(res));

            // XXX: shutdown is not really supported
            if (_shutdown) {
                LOG("shutdown requested");
                res = RETURN_CODE_SHUTDOWN;
                break;
            } else if (fres) {
                res = RETURN_CODE_OK;
                break;
            }

            std::chrono::duration<double, std::chrono::seconds::period> elapsed =
                std::chrono::high_resolution_clock::now() - start;

            if (timeoutSeconds != -1 && elapsed.count() > timeoutSeconds) {
                res = RETURN_CODE_TIMEOUT;
                break;
            }
        }

        LOGFMT("[%s] cond%d.wait/2 -> EXIT %d", funcname % id % static_cast<int>(res));
        return res;
    }

    int Connector::cond_wait(const char* funcname, int id, std::condition_variable &cond, std::unique_lock<std::mutex> &l, int timeoutSeconds) {
        std::function<bool()> checker = [&cond, &l]() -> bool {
            return cond.wait_for(l, std::chrono::milliseconds(1000)) == std::cv_status::no_timeout;
        };

        return _cond_wait(funcname, id, cond, l, timeoutSeconds, checker);
    }

    int Connector::cond_wait(const char* funcname, int id, std::condition_variable &cond, std::unique_lock<std::mutex> &l, int timeoutSeconds, std::function<bool()> &pred) {
        std::function<bool()> checker = [&cond, &l, &pred]() -> bool {
            return cond.wait_for(l, std::chrono::milliseconds(1000), pred);
        };

        return _cond_wait(funcname, id, cond, l, timeoutSeconds, checker);
    }

    // EOF TEST SIGNAL HANDLING

    Connector::Connector(
        const std::string mapname_,
        const int seed_,
        const int randomHeroes_,
        const int randomObstacles_,
        const int townChance_,
        const int warmachineChance_,
        const int manaMin_,
        const int manaMax_,
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
        const int statsPersistFreq_
    ) : mapname(mapname_),
        seed(seed_),
        randomHeroes(randomHeroes_),
        randomObstacles(randomObstacles_),
        townChance(townChance_),
        warmachineChance(warmachineChance_),
        manaMin(manaMin_),
        manaMax(manaMax_),
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
        statsPersistFreq(statsPersistFreq_)
        {};

    const int Connector::version() {
        return 4;
    }

    const MMAI::Schema::V4::ISupplementaryData* Connector::extractSupplementaryData(const MMAI::Schema::IState *s) {
        LOG("Extracting supplementary data...");
        auto &any = s->getSupplementaryData();
        if(!any.has_value()) throw std::runtime_error("extractSupplementaryData: supdata is empty");
        auto &t = typeid(const MMAI::Schema::V4::ISupplementaryData*);
        auto err = MMAI::Schema::AnyCastError(any, typeid(const MMAI::Schema::V4::ISupplementaryData*));

        if(!err.empty()) {
            LOGFMT("anycast for getSumpplementaryData error: %s", err);
        }

        return std::any_cast<const MMAI::Schema::V4::ISupplementaryData*>(s->getSupplementaryData());
    };

    const P_State Connector::convertState(const MMAI::Schema::IState* s) {
        LOG("Convert IState -> P_State");
        auto sup = extractSupplementaryData(s);
        assert(sup->getType() == MMAI::Schema::V4::ISupplementaryData::Type::REGULAR);

        // XXX: these do not improve performance, better avoid the const_cast
        // auto pbs = P_BattlefieldState(bs.size(), const_cast<float*>(bs.data()));
        // auto patm = P_AttentionMask(attnmask.size(), const_cast<float*>(attnmask.data()));

        // XXX: manually copying the state into py::array_t<float> is
        //      ~10% faster than storing the BattlefieldState& reference in
        //      P_State as pybind's STL automatically converts it to a python
        //      list of python floats, which needs to be converted to a numpy
        //      array of float42 floats in pyconnector.set_v_result_act()
        //      (which copies the data anyway and is ultimately slower).

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

        auto stats = sup->getStats();
        auto res = P_State(
             sup->getType(),
             pbs,
             pam,
             patm,
             sup->getErrorCode(),
             sup->getSide(),
             sup->getDmgDealt(),
             sup->getDmgReceived(),
             sup->getUnitsLost(),
             sup->getUnitsKilled(),
             sup->getValueLost(),
             sup->getValueKilled(),
             stats->getInitialArmyValueLeft(),
             stats->getInitialArmyValueRight(),
             stats->getCurrentArmyValueLeft(),
             stats->getCurrentArmyValueRight(),
             sup->getIsBattleEnded(),
             sup->getIsVictorious(),
             sup->getAnsiRender()
        );

        return res;
    }

    const std::tuple<int, std::string> Connector::renderAnsi(int side) {
        LOG("renderAnsi called with side=" + std::to_string(side));
        ASSERT_STATE("renderAnsi", side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0);

        // throw any errors set during getAction
        maybeThrowError();

        auto &m = side ? m1 : m0;
        auto &cond = side ? cond1 : cond0;

        LOG("obtain lock");
        std::unique_lock lock(m);
        LOG("obtain lock: done");

        LOGFMT("set this->action = %d", MMAI::Schema::ACTION_RENDER_ANSI);
        action = MMAI::Schema::ACTION_RENDER_ANSI;

        LOG("set connstate = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOG("cond.notify_one()");
        cond.notify_one();

        int res = 0;

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOG("cond.wait(lock)");
            res = cond_wait(__func__, side, cond, lock, vcmiTimeout);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
        }

        ASSERT_STATE("renderAnsi.2", side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0);

        // throw any errors set during getAction
        maybeThrowError();

        auto sup = extractSupplementaryData(state);
        assert(sup->getType() == MMAI::Schema::V4::ISupplementaryData::Type::ANSI_RENDER);

        LOG("release lock (return)");
        LOG("return state->ansiRender");
        return {res, sup->getAnsiRender()};
    }

    const std::tuple<int, P_State> Connector::reset(int side) {
        LOG("reset called with side=" + std::to_string(side));
        ASSERT_STATE("reset", side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0);

        // throw any errors set during getAction
        maybeThrowError();

        auto &m = side ? m1 : m0;
        auto &cond = side ? cond1 : cond0;

        std::unique_lock lock(m);
        LOG("obtain lock: done");

        LOGFMT("set this->action = %d", MMAI::Schema::ACTION_RESET);
        action = MMAI::Schema::ACTION_RESET;

        LOG("set state = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOG("cond.notify_one()");
        cond.notify_one();

        int res;

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOG("cond.wait(lock)");
            res = cond_wait(__func__, side, cond, lock, vcmiTimeout);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
            // py::gil_scoped_acquire acquire2;
        }

        ASSERT_STATE("reset.2", side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0);

        // throw any errors set during getAction
        maybeThrowError();

        LOG("release lock (return)");
        LOG("return P_State");
        const auto pstate = convertState(state);
        assert(pstate.type == MMAI::Schema::V4::ISupplementaryData::Type::REGULAR);
        return {res, pstate};
    }

    const std::tuple<int, P_State> Connector::getState(int side, MMAI::Schema::Action a) {
        LOG("getState called with side=" + std::to_string(side));
        ASSERT_STATE("getState", side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0);

        // throw any errors set during getAction
        maybeThrowError();

        // Prevent control actions via `step`
        assert(a > 0);

        auto &m = side ? m1 : m0;
        auto &cond = side ? cond1 : cond0;

        LOG("obtain lock");
        std::unique_lock lock(m);
        LOG("obtain lock: done");

        LOGFMT("set this->action = %d", a);
        action = a;

        LOG("set connstate = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOG("cond.notify_one()");
        cond.notify_one();

        int res;

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOG("cond.wait(lock)");
            res = cond_wait(__func__, side, cond, lock, vcmiTimeout);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
        }

        ASSERT_STATE("getState.2", side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0);

        // throw any errors set during getAction
        maybeThrowError();

        LOG("release lock (return)");
        LOG("return P_State");
        return {res, convertState(state)};
    }

    // this is called by a VCMI thread (the runNetwork thread)
    // Python does not know about this thread and exceptions thrown here
    // result in abrupt program termination.
    // => set a member var `_error` to be thrown by python-aware threads
    // Only throw here if this var was previously set and still not thrown
    MMAI::Schema::Action Connector::getAction(const MMAI::Schema::IState* s, int side) {
        LOG("getAction called with side=" + std::to_string(side));

        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        auto &m = side ? m1 : m0;
        auto &cond = side ? cond1 : cond0;

        LOG("obtain lock");
        std::unique_lock lock(m);
        LOG("obtain lock: done");

        if (connstate != ConnectorState::AWAITING_STATE)
            setError(FMT("getAction: Unexpected connector state: want: %d, have: %d", EI(ConnectorState::AWAITING_STATE) % EI(connstate)));

        LOG("set this->istate = s");
        state = s;

        LOG("set connstate = AWAITING_ACTION_" + std::to_string(side));
        connstate = side
            ? ConnectorState::AWAITING_ACTION_1
            : ConnectorState::AWAITING_ACTION_0;

        LOG("cond.notify_one()");
        cond.notify_one();

        int res;

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            // Now wait again (will unblock once step/reset have been called)
            LOG("cond.wait(lock)");
            res = cond_wait(__func__, side, cond, lock, userTimeout);
            LOG("cond.wait(lock): done");

            LOG("acquire Python GIL (scope-auto)");
            // py::gil_scoped_acquire acquire2;
        }

        // the above cond_wait gave priority to a python thread which was
        // waiting in getState. It was supposed to throw any stored errors
        // If it did not (bug) => throw here
        maybeThrowError();

        if (res == RETURN_CODE_TIMEOUT) {
            setError(boost::str(boost::format("getAction: timeout after %ds while waiting for user\n") % userTimeout));
        } else if (res != RETURN_CODE_OK) {
            setError(boost::str(boost::format("getAction: unexpected return code from cond_wait: %d\n") % res));
        } else if (connstate != ConnectorState::AWAITING_STATE) {
            setError(FMT("getAction: unexpected connector state: want: %d, have: %d", EI(ConnectorState::AWAITING_STATE) % EI(connstate)));
        }

        LOG("release lock (return)");
        LOGFMT("return Action: %d", action);
        return action;
    }

    const std::tuple<int, P_State> Connector::connect(int side) {
        LOG("connect called with side=" + std::to_string(side));

        LOG("release Python GIL");
        py::gil_scoped_release release;

        std::this_thread::sleep_for(std::chrono::seconds(1));
        LOG("obtain lock2");
        std::unique_lock lock2(m2);
        LOG("obtain lock2: done");

        ASSERT_STATE("connect", ConnectorState::NEW);

        if (side)
            connectedClient1 = true;
        else
            connectedClient0 = true;

        LOG("cond2.notify_one()");
        cond2.notify_one();

        auto &m = side ? m1 : m0;
        auto &cond = side ? cond1 : cond0;

        LOG("obtain lock");
        std::unique_lock lock(m);
        LOG("obtain lock: done");

        LOG("release lock2");
        lock2.unlock();

        LOG("cond.wait(lock)");
        auto res = cond_wait(__func__, side, cond, lock, bootTimeout);
        LOG("cond.wait(lock): done");

        ASSERT_STATE("connect.2", side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0);

        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("release lock (return)");
        LOG("release Python GIL (return)");
        LOG("return P_Result");

        return {res, convertState(state)};
    }

    void Connector::start(int bootTimeout_, int vcmiTimeout_, int userTimeout_) {
        ASSERT_STATE("start", ConnectorState::NEW);

        bootTimeout = bootTimeout_;
        vcmiTimeout = vcmiTimeout_;
        userTimeout = userTimeout_;

        setvbuf(stdout, NULL, _IONBF, 0);
        LOG("start");

        // struct sigaction sa;
        // sa.sa_handler = signal_handler;
        // sa.sa_flags = 0;
        // sigemptyset(&sa.sa_mask);

        // if (sigaction(SIGINT, &sa, nullptr) == -1)
        //     throw std::runtime_error("Error installing signal handler.");

        LOG("obtain lock2");
        std::unique_lock lock2(m2);
        LOG("obtain lock2: done");

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            std::function<bool()> predicate = [this] {
                return (connectedClient0 || red != "MMAI_USER")
                    && (connectedClient1 || blue != "MMAI_USER");
            };

            if (predicate()) {
                LOGFMT("clients already connected: (%s || %s) && (%s || %s)", connectedClient0 % red % connectedClient1 % blue);
            } else {
                LOGFMT("cond2.wait(lock2, %1%s, predicate)", bootTimeout);
                auto res = cond_wait(__func__, 2, cond2, lock2, bootTimeout, predicate);
                if (res == RETURN_CODE_TIMEOUT) {
                    throw VCMIConnectorException(boost::str(boost::format(
                        "timeout after %ds while waiting for client (red:%s, blue:%s, connectedClient0: %d, connectedClient1: %d)\n") \
                        % bootTimeout % red % blue % connectedClient0 % connectedClient1
                    ));
                    return;
                } else if (res != RETURN_CODE_OK) {
                    throw VCMIConnectorException(boost::str(boost::format(
                        "unexpected return code from cond_wait: %d\n") % res
                    ));
                    return;
                }
            }

            // Successfully obtaining these locks means the
            // clients are ready and waiting for state
            LOG("obtain lock0");
            std::unique_lock lock0(m0);
            LOG("obtain lock0: done");

            LOG("obtain lock1");
            std::unique_lock lock1(m1);
            LOG("obtain lock1: done");

            LOG("acquire Python GIL (scope-auto)");
            // py::gil_scoped_acquire acquire2;
        }

        auto f_getAction0 = [this](const MMAI::Schema::IState* s) {
            return this->getAction(s, 0);
        };

        auto f_getAction1 = [this](const MMAI::Schema::IState* s) {
            return this->getAction(s, 1);
        };

        auto f_getValueDummy = [](const MMAI::Schema::IState* s) {
            std::cerr << "WARNING: getValue called, but is not implemented in connector\n";
            return 0.0;
        };

        std::function<int(const MMAI::Schema::IState* s)> getActionRed;
        std::function<int(const MMAI::Schema::IState* s)> getActionBlue;
        std::function<double(const MMAI::Schema::IState* s)> getValueRed;
        std::function<double(const MMAI::Schema::IState* s)> getValueBlue;

        if (red == "MMAI_USER") {
            leftModel = new ML::ModelWrappers::Function(version(), "MMAI_MODEL", f_getAction0, f_getValueDummy);
        } else if (red == "MMAI_MODEL") {
            leftModel = new MMAI::TorchModel(redModel, false);
        } else {
            leftModel = new ML::ModelWrappers::Scripted(red);
        }

        if (blue == "MMAI_USER") {
            rightModel = new ML::ModelWrappers::Function(version(), "MMAI_MODEL", f_getAction1, f_getValueDummy);
        } else if (blue == "MMAI_MODEL") {
            rightModel = new MMAI::TorchModel(blueModel, false);
        } else {
            rightModel = new ML::ModelWrappers::Scripted(blue);
        }

        // auto oldcwd = std::filesystem::current_path();

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

        // This must happen in the main thread (SDL requires it)
        LOG("call init_vcmi(...)");
        init_vcmi(initargs);

        LOG("set connstate = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOG("release lock2");
        lock2.unlock();

        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("launch VCMI (will never return)");
        ML::start_vcmi();
        assert(false); // should never be here
    }
}
