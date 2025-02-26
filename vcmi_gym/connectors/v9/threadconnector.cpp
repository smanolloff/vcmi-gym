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
#include "schema/base.h"
#include "schema/v9/constants.h"
#include "schema/v9/types.h"
#include "ML/MLClient.h"
#include "ML/model_wrappers/function.h"
#include "ML/model_wrappers/scripted.h"
#include "ML/model_wrappers/torchpath.h"
#include "exporter.h"
#include <algorithm>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <thread>
#include <boost/date_time/posix_time/posix_time.hpp>

#define ASSERT_STATE(id, want) { \
    if((want) != (connstate)) \
        throw VCMIConnectorException(std::string(id) + ": unexpected connector state: want: " + std::to_string(EI(want)) + ", have: " + std::to_string(EI(connstate))); \
}

// Python does not know about some threads and exceptions thrown there
// result in abrupt program termination.
// => use this to set a member var `_error`
//    (the exception must be constructed and thrown in python-aware thread)
#define SET_ERROR(msg) { \
    if (!_shutdown) { \
        std::cerr << boost::str(boost::format("ERROR (only recorded): %1%") % msg); \
        LOG(boost::str(boost::format("ERROR (only recorded): %1%") % msg)); \
        _error = msg; \
    } \
}

#define SHUTDOWN_VCMI_RETURN(ret) { \
    if (_shutdown) { \
        LOG("connector is shutting down"); \
        return ret; \
    } \
}

#define SHUTDOWN_PYTHON_RETURN(ret) { \
    if (_shutdown) { \
        LOG("connector is shutting down"); \
        return {static_cast<int>(ReturnCode::SHUTDOWN), ret}; \
    } \
}

namespace Connector::V9::Thread {

    const std::vector<std::string> Connector::getLogs() {
        return std::vector<std::string>(logs.begin(), logs.end());
    }

    void Connector::log(std::string funcname, std::string msg) {
#if VERBOSE || LOGCOLLECT
        boost::posix_time::ptime t = boost::posix_time::microsec_clock::universal_time();

        // std::string entry = boost::str(boost::format("++ %1% <%2%>[%3%][%4%] <%5%> %6%") \
        //     % boost::posix_time::to_iso_extended_string(t)
        //     % std::this_thread::get_id()
        //     % std::filesystem::path(__FILE__).filename().string()
        //     % (PyGILState_Check() ? "GIL=1" : "GIL=0")
        //     % funcname
        //     % msg
        // );

        std::string entry = boost::str(boost::format("++ %s <%s>[GIL=%d] <%s> %s")
            % boost::posix_time::to_iso_extended_string(t)
            % std::this_thread::get_id()
            % PyGILState_Check()
            % funcname
            % msg
        );

#if LOGCOLLECT
        {
            std::unique_lock lock(mlog);
            if (logs.size() == maxlogs)
                logs.pop_front();
            logs.push_back(entry);
        }
#endif // LOGCOLLECT

#if VERBOSE
        {
            std::unique_lock lock(mlog);
            std::cout << entry << "\n";
        }
#endif // VERBOSE
#endif // VERBOSE || LOGCOLLECT
    }

    ReturnCode Connector::_cond_wait(const char* funcname, int id, std::condition_variable &cond, std::unique_lock<std::mutex> &l, int timeoutSeconds, std::function<bool()> &checker) {
        ReturnCode res;
        int i = 0;
        auto start = std::chrono::high_resolution_clock::now();

        while (true) {
            // LOG(boost::str(boost::format("[%s] cond%d.wait/2: checker()...") % funcname % id));
            auto fres = checker();
            // LOG(boost::str(boost::format("[%s] cond%d.wait/2: checker -> %d") % funcname % id % static_cast<int>(fres)));

            // XXX: shutdown is not really supported
            if (_shutdown) {
                LOG("shutdown requested");
                res = ReturnCode::SHUTDOWN;
                break;
            } else if (fres) {
                res = ReturnCode::OK;
                break;
            }

            std::chrono::duration<double, std::chrono::seconds::period> elapsed =
                std::chrono::high_resolution_clock::now() - start;

            if (timeoutSeconds != -1 && elapsed.count() > timeoutSeconds) {
                LOG(boost::str(boost::format("%s: cond%d.wait/2 timed out after %d seconds\n") % funcname % id % elapsed.count()));
                res = ReturnCode::TIMEOUT;
                break;
            }
        }

        LOGFMT("[%s] cond%d.wait/2 -> EXIT %d", funcname % id % static_cast<int>(res));
        return res;
    }

    ReturnCode Connector::cond_wait(const char* funcname, int id, std::condition_variable &cond, std::unique_lock<std::mutex> &l, int timeoutSeconds, std::function<bool()> &pred) {
        std::function<bool()> checker = [&cond, &l, &pred]() -> bool {
            return cond.wait_for(l, std::chrono::milliseconds(100), pred);
        };

        return _cond_wait(funcname, id, cond, l, timeoutSeconds, checker);
    }

    // EOF TEST SIGNAL HANDLING

    Connector::Connector(
        const int maxlogs_,
        const int bootTimeout_,
        const int vcmiTimeout_,
        const int userTimeout_,
        const std::string mapname_,
        const int seed_,
        const int randomHeroes_,
        const int randomObstacles_,
        const int townChance_,
        const int warmachineChance_,
        const int tightFormationChance_,
        const int randomTerrainChance_,
        const std::string battlefieldPattern_,
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
    ) : maxlogs(maxlogs_),
        bootTimeout(bootTimeout_),
        vcmiTimeout(vcmiTimeout_),
        userTimeout(userTimeout_),
        mapname(mapname_),
        seed(seed_),
        randomHeroes(randomHeroes_),
        randomObstacles(randomObstacles_),
        townChance(townChance_),
        warmachineChance(warmachineChance_),
        tightFormationChance(tightFormationChance_),
        randomTerrainChance(randomTerrainChance_),
        battlefieldPattern(battlefieldPattern_),
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

    const MMAI::Schema::V9::ISupplementaryData* Connector::extractSupplementaryData(const MMAI::Schema::IState *s) {
        LOG("Extracting supplementary data...");
        auto &any = s->getSupplementaryData();
        if(!any.has_value()) throw std::runtime_error("extractSupplementaryData: supdata is empty");
        auto &t = typeid(const MMAI::Schema::V9::ISupplementaryData*);
        auto err = MMAI::Schema::AnyCastError(any, typeid(const MMAI::Schema::V9::ISupplementaryData*));

        if(!err.empty()) {
            LOGFMT("anycast for getSumpplementaryData error: %s", err);
        }

        return std::any_cast<const MMAI::Schema::V9::ISupplementaryData*>(s->getSupplementaryData());
    };

    const P_State Connector::convertState(const MMAI::Schema::IState* s) {
        LOG("Convert IState -> P_State");
        auto sup = extractSupplementaryData(s);
        assert(sup->getType() == MMAI::Schema::V9::ISupplementaryData::Type::REGULAR);

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

        auto res = P_State(
             sup->getType(),
             pbs,
             pam,
             sup->getErrorCode(),
             sup->getAnsiRender()
        );

        return res;
    }

    ReturnCode Connector::getState(const char* funcname, int side, MMAI::Schema::Action action_) {
        LOGFMT("%s called with side=%d", funcname % side);

        auto expstate = side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0;
        auto &m = side ? m1 : m0;
        auto &cond = side ? cond1 : cond0;

        ASSERT_STATE(funcname, expstate);

        LOGFMT("obtain lock%d", side);
        std::unique_lock lock(m);
        LOGFMT("obtain lock%d: done", side);

        LOGFMT("set this->action = %d", action_);
        action = action_;

        LOG("set connstate = AWAITING_STATE");
        connstate = ConnectorState::AWAITING_STATE;

        LOGFMT("cond%d.notify_one()", side);
        cond.notify_one();

        std::function<bool()> pred = [this, expstate] { return connstate == expstate || _shutdown; };

        LOGFMT("cond%1%.wait(lock%1%)", side);
        auto res = cond_wait(funcname, side, cond, lock, vcmiTimeout, pred);
        LOGFMT("cond%1%.wait(lock%1%): done", side);

        if (!_error.empty()) {
            throw VCMIConnectorException(_error);
        }

        LOGFMT("release lock%d (return)", side);
        return res;
    }

    const std::tuple<int, std::string> Connector::render(int side) {
        SHUTDOWN_PYTHON_RETURN("");
        auto code = getState(__func__, side, MMAI::Schema::ACTION_RENDER_ANSI);
        auto sup = extractSupplementaryData(state);
        assert(sup->getType() == MMAI::Schema::V9::ISupplementaryData::Type::ANSI_RENDER);
        LOG("return state->ansiRender");
        return {static_cast<int>(code), sup->getAnsiRender()};
    }

    const std::tuple<int, P_State> Connector::reset(int side) {
        SHUTDOWN_PYTHON_RETURN(convertState(state)); // reuse last state if shutting down
        auto code = getState(__func__, side, MMAI::Schema::ACTION_RESET);
        auto pstate = convertState(state);
        LOG("return P_State");
        return {static_cast<int>(code), pstate};
    }

    const std::tuple<int, P_State> Connector::step(int side, MMAI::Schema::Action a) {
        SHUTDOWN_PYTHON_RETURN(convertState(state)); // reuse last state if shutting down
        auto code = getState(__func__, side, a);
        auto pstate = convertState(state);
        LOG("return P_State");
        return {static_cast<int>(code), pstate};
    }

    // this is called by a VCMI thread (the runNetwork thread)
    // Python does not know about this thread and exceptions thrown here
    // result in abrupt program termination.
    // => set a member var `_error` to be thrown by python-aware threads
    // Only throw here if this var was previously set and still not thrown
    MMAI::Schema::Action Connector::getAction(const MMAI::Schema::IState* s, int side) {
        SHUTDOWN_VCMI_RETURN(MMAI::Schema::ACTION_RESET);

        LOG("getAction called with side=" + std::to_string(side));

        auto &m = side ? m1 : m0;
        auto &cond = side ? cond1 : cond0;

        LOGFMT("obtain lock%d", side);
        std::unique_lock lock(m);
        LOGFMT("obtain lock%d: done", side);

        if (connstate != ConnectorState::AWAITING_STATE)
            SET_ERROR(boost::str(boost::format("%s: Unexpected connector state: want: %d, have: %d") % __func__ % EI(ConnectorState::AWAITING_STATE) % EI(connstate)));

        LOG("set this->istate = s");
        state = s;

        LOG("set connstate = AWAITING_ACTION_" + std::to_string(side));
        connstate = side
            ? ConnectorState::AWAITING_ACTION_1
            : ConnectorState::AWAITING_ACTION_0;

        LOGFMT("cond%d.notify_one()", side);
        cond.notify_one();

        std::function<bool()> pred = [this] { return connstate == ConnectorState::AWAITING_STATE || _shutdown; };

        // Now wait again (will unblock once step/reset have been called)
        LOGFMT("cond%1%.wait(lock%1%)", side);
        auto res = cond_wait(__func__, side, cond, lock, userTimeout, pred);
        LOGFMT("cond%1%.wait(lock%1%): done", side);

        SHUTDOWN_VCMI_RETURN(MMAI::Schema::ACTION_RESET);

        // the above cond_wait gave priority to a python thread which was
        // waiting in getState. It was supposed to throw any stored errors
        // If it did not (bug) => throw here
        if (!_error.empty()) {
            // need to explicitly print the logs here
            // (this exception won't be handled by python)
            for (auto &msg : logs)
                std::cerr << msg << "\n";
            throw VCMIConnectorException(_error);
        }

        if (res == ReturnCode::TIMEOUT) {
            SET_ERROR(boost::str(boost::format("timeout after %ds while waiting for user\n") % userTimeout));
        } else if (res == ReturnCode::SHUTDOWN) {
            LOG("connector is shutting down...");
        } else if (res != ReturnCode::OK) {
            SET_ERROR(boost::str(boost::format("unexpected return code from cond_wait: %d\n") % EI(res)));
        } else if (connstate != ConnectorState::AWAITING_STATE) {
            SET_ERROR(boost::str(boost::format("unexpected connector state: want: %d, have: %d") % EI(ConnectorState::AWAITING_STATE) % EI(connstate)));
        }

        LOGFMT("release lock%d (return)", side);
        LOGFMT("return Action: %d", action);
        return action;
    }

    // initial connect is a special case and cannot reuse getState()
    const std::tuple<int, P_State> Connector::connect(int side) {
        LOG("connect called with side=" + std::to_string(side));

        LOG("obtain lock2");
        std::unique_lock lock2(m2);
        LOG("obtain lock2: done");

        if (side)
            connectedClient1 = true;
        else
            connectedClient0 = true;

        LOG("cond2.notify_one()");
        cond2.notify_one();

        auto expstate = side ? ConnectorState::AWAITING_ACTION_1 : ConnectorState::AWAITING_ACTION_0;
        auto &m = side ? m1 : m0;
        auto &cond = side ? cond1 : cond0;

        LOGFMT("obtain lock%d", side);
        std::unique_lock lock(m);
        LOGFMT("obtain lock%d: done", side);

        LOG("release lock2");
        lock2.unlock();

        ReturnCode res;

        std::function<bool()> pred = [this, expstate] { return connstate == expstate; };
        LOGFMT("cond%1%.wait(lock%1%)", side);
        res = cond_wait(__func__, side, cond, lock, bootTimeout, pred);
        LOGFMT("cond%1%.wait(lock%1%): done", side);

        auto pstate = convertState(state);
        LOGFMT("release lock%d (return)", side);
        LOG("return P_State");
        return {static_cast<int>(res), pstate};
    }

    void Connector::start() {
        ASSERT_STATE("start", ConnectorState::NEW);

        setvbuf(stdout, NULL, _IONBF, 0);
        LOG("start");

        LOG("release Python GIL");
        py::gil_scoped_release release;

        // struct sigaction sa;
        // sa.sa_handler = signal_handler;
        // sa.sa_flags = 0;
        // sigemptyset(&sa.sa_mask);

        // if (sigaction(SIGINT, &sa, nullptr) == -1)
        //     throw std::runtime_error("Error installing signal handler.");

        LOG("obtain lock2");
        std::unique_lock lock2(m2);
        LOG("obtain lock2: done");

        std::function<bool()> predicate = [this] {
            return (connectedClient0 || red != "MMAI_USER")
                && (connectedClient1 || blue != "MMAI_USER");
        };

        LOGFMT("cond2.wait(lock2, %1%s, predicate)", bootTimeout);
        auto res = cond_wait(__func__, 2, cond2, lock2, bootTimeout, predicate);
        if (res == ReturnCode::TIMEOUT) {
            throw VCMIConnectorException(boost::str(boost::format(
                "timeout after %ds while waiting for client (red:%s, blue:%s, connectedClient0: %d, connectedClient1: %d)\n") \
                % bootTimeout % red % blue % connectedClient0 % connectedClient1
            ));
            return;
        } else if (res == ReturnCode::SHUTDOWN) {
            LOG("connector is shutting down...");
            return;
        } else if (res != ReturnCode::OK) {
            throw VCMIConnectorException(boost::str(boost::format(
                "unexpected return code from cond_wait: %d\n") % EI(res)
            ));
            return;
        }

        {
            // Successfully obtaining these locks means the
            // clients are ready and waiting for state
            LOG("obtain lock0");
            std::unique_lock lock0(m0);
            LOG("obtain lock0: done");

            LOG("obtain lock1");
            std::unique_lock lock1(m1);
            LOG("obtain lock1: done");

            LOG("release lock0 and lock1");
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

        auto f_getRandomAction = [](const MMAI::Schema::IState* s) {
            return RandomValidAction(s);
        };

        if (red == "MMAI_RANDOM") {
            leftModel = new ML::ModelWrappers::Function(version(), "MMAI_RANDOM", f_getRandomAction, f_getValueDummy);
        } else if (red == "MMAI_USER") {
            leftModel = new ML::ModelWrappers::Function(version(), "MMAI_MODEL", f_getAction0, f_getValueDummy);
        } else if (red == "MMAI_MODEL") {
            // BAI will load the actual model based on leftModel->getName()
            leftModel = new ML::ModelWrappers::TorchPath(redModel);
        } else {
            leftModel = new ML::ModelWrappers::Scripted(red);
        }

        if (blue == "MMAI_RANDOM") {
            rightModel = new ML::ModelWrappers::Function(version(), "MMAI_RANDOM", f_getRandomAction, f_getValueDummy);
        } else if (blue == "MMAI_USER") {
            rightModel = new ML::ModelWrappers::Function(version(), "MMAI_MODEL", f_getAction1, f_getValueDummy);
        } else if (blue == "MMAI_MODEL") {
            // BAI will load the actual model based on rightModel->getName()
            rightModel = new ML::ModelWrappers::TorchPath(blueModel);
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
            townChance,         // townChance
            warmachineChance,   // warmachineChance
            tightFormationChance,  // tightFormationChance
            randomTerrainChance,  // randomTerrainChance
            battlefieldPattern,  // battlefieldPattern
            manaMin,            // manaMin
            manaMax,            // manaMax
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

        LOG("launch VCMI (will never return)");
        ML::start_vcmi();

        if (!_shutdown)
            std::cerr << "ERROR: ML::start_vcmi() returned, but shutdown is false";
    }

    void Connector::shutdown() {
        _shutdown = true;
        ML::shutdown_vcmi();
    }
}
