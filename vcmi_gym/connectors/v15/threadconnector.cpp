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
#include "common.h"
#include "schema/base.h"
#include "schema/v15/constants.h"
#include "schema/v15/graph.h"
#include "schema/v15/types.h"
#include "ML/MLClient.h"
#include "ML/model_wrappers/function.h"
#include "ML/model_wrappers/scripted.h"
#include "ML/model_wrappers/path.h"
#include "exporter.h"

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <mutex>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <random>
#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <unistd.h>

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

namespace {
    int RandomValidAction(const MMAI::Schema::IState * s) {
        auto any = s->getSupplementaryData();
        const auto * sup = std::any_cast<const MMAI::Schema::V15::ISupplementaryData*>(any);
        const auto * G = sup->getGraph();

        auto activeIds = G->getActiveActionIds();

        if (activeIds.empty()) {
            std::cout << "No valid actions => reset\n";
            return MMAI::Schema::ACTION_RESET;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, static_cast<int>(activeIds.size()) - 1);
        int randomIndex = dist(gen);
        const auto id = activeIds[randomIndex];
        return static_cast<int>(id);
    }
}

namespace Connector::V15::Thread {
    namespace S15 = MMAI::Schema::V15;

    std::vector<std::string> Connector::getLogs() {
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

        std::string entry = boost::str(boost::format("++ %s <%ld/%s>[GIL=%d] <%s> %s")
            % boost::posix_time::to_iso_extended_string(t)
            % static_cast<int64_t>(getpid())
            // % std::this_thread::get_id()
            % to_base36(native_thread_id())
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
        int maxlogs_,
        int bootTimeout_,
        int vcmiTimeout_,
        int userTimeout_,
        const std::string & mapname_,
        int seed_,
        int randomHeroes_,
        int randomObstacles_,
        int townChance_,
        int warmachineChance_,
        int randomStackChance_,
        int tightFormationChance_,
        int randomTerrainChance_,
        int leftVipChance_,
        int rightVipChance_,
        const std::string & battlefieldPattern_,
        int manaMin_,
        int manaMax_,
        int randomPrimarySkills_,
        int swapSides_,
        const std::string & loglevelGlobal_,
        const std::string & loglevelAI_,
        const std::string & loglevelStats_,
        const std::string & red_,
        const std::string & blue_,
        const std::string & redModel_,
        const std::string & blueModel_,
        bool redAllowMlBot_,
        bool blueAllowMlBot_,
        const std::string & statsMode_,
        const std::string & statsStorage_,
        int statsPersistFreq_
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
        randomStackChance(randomStackChance_),
        tightFormationChance(tightFormationChance_),
        randomTerrainChance(randomTerrainChance_),
        leftVipChance(leftVipChance_),
        rightVipChance(rightVipChance_),
        battlefieldPattern(battlefieldPattern_),
        manaMin(manaMin_),
        manaMax(manaMax_),
        randomPrimarySkills(randomPrimarySkills_),
        swapSides(swapSides_),
        loglevelGlobal(loglevelGlobal_),
        loglevelAI(loglevelAI_),
        loglevelStats(loglevelStats_),
        red(red_),
        blue(blue_),
        redModel(redModel_),
        blueModel(blueModel_),
        redAllowMlBot(redAllowMlBot_),
        blueAllowMlBot(blueAllowMlBot_),
        statsMode(statsMode_),
        statsStorage(statsStorage_),
        statsPersistFreq(statsPersistFreq_)
        {};

    const MMAI::Schema::V15::ISupplementaryData* Connector::extractSupplementaryData(const MMAI::Schema::IState *s) {
        LOG("Extracting supplementary data...");
        auto any = s->getSupplementaryData();
        if(!any.has_value()) throw std::runtime_error("extractSupplementaryData: supdata is empty");
        const auto & t = typeid(const MMAI::Schema::V15::ISupplementaryData*);
        auto err = MMAI::Schema::AnyCastError(any, typeid(const MMAI::Schema::V15::ISupplementaryData*));

        if(!err.empty()) {
            LOGFMT("anycast for getSumpplementaryData error: %s", err);
        }

        return std::any_cast<const MMAI::Schema::V15::ISupplementaryData*>(s->getSupplementaryData());
    };

    py::dict Connector::buildObsDict(const MMAI::Schema::IState * s) {
        LOG("Convert IState -> p_dict");
        const auto * sup = extractSupplementaryData(s);
        assert(sup->getType() == MMAI::Schema::V15::ISupplementaryData::Type::REGULAR);

        const auto * G = sup->getGraph();

        auto nodeTypeMap = std::unordered_map<S15::Graph::ElementType, std::string>{};

        auto p_nodesdict = py::dict();
        for (const auto & [type, name, size] : MMAI::Schema::V15::NODE_TYPES)
        {
            const auto & nodes = G->getNodes(type);
            const auto N = static_cast<py::ssize_t>(nodes.size());
            const auto D = static_cast<py::ssize_t>(size);

            auto p_attrs = py::array_t<float, py::array::c_style>({N, D});
            auto info = p_attrs.request();
            auto * ptr = static_cast<float*>(info.ptr);
            auto out = std::span<float>(ptr, N * D);

            std::ranges::fill(out, 0.0f);

            for (std::size_t j = 0; j < N; ++j)
            {
                auto row = out.subspan(j * D, D);
                int written = nodes[j]->encode(row);
                assert(written == D);
            }

            p_nodesdict[py::str(name)] = p_attrs;
            nodeTypeMap.emplace(type, name);
        }

        auto p_edgesdict = py::dict();
        for (const auto & [type, name, endpoints, size] : MMAI::Schema::V15::EDGE_TYPES)
        {
            auto p_edict = py::dict();

            const auto & edges = G->getEdges(type);
            const auto E = static_cast<py::ssize_t>(edges.size());
            const auto D = static_cast<py::ssize_t>(size);

            auto p_attrs = py::array_t<float, py::array::c_style>({E, D});
            auto p_index = py::array_t<int64_t, py::array::c_style>({static_cast<py::ssize_t>(2), E});
            auto mp_index = p_index.mutable_unchecked<2>();

            auto info = p_attrs.request();
            auto * ptr = static_cast<float*>(info.ptr);
            auto out = std::span<float>(ptr, E * D);
            std::ranges::fill(out, 0.0f);

            for (std::size_t j = 0; j < E; ++j)
            {
                auto row = out.subspan(j * D, D);
                const auto * edge = edges[j];
                int written = edge->encode(row);
                if (written != D)
                    throw std::runtime_error("written: " + std::to_string(written) + ": expected: " + std::to_string(D) + " ET=" + std::to_string(EI(edge->getType())));
                assert(written == D);

                const auto & [srcNode, dstNode] = edge->endpoints();
                int64_t isrc = G->getNodeIndex(srcNode);
                int64_t idst = G->getNodeIndex(dstNode);
                mp_index(0, static_cast<py::ssize_t>(j)) = isrc;
                mp_index(1, static_cast<py::ssize_t>(j)) = idst;
            }

            p_edict[py::str("index")] = p_index;
            p_edict[py::str("attrs")] = p_attrs;

            const auto & [src_type, dst_type] = endpoints;
            const auto & src_name = nodeTypeMap.at(src_type);
            const auto & dst_name = nodeTypeMap.at(dst_type);
            const auto key = py::make_tuple(
                py::str(src_name),
                py::str(name),
                py::str(dst_name)
            );
            p_edgesdict[key] = p_edict;
        }

        const auto & activeActionIds = G->getActiveActionIds();
        const auto A = static_cast<py::ssize_t>(activeActionIds.size());
        auto p_activeids = py::array_t<int64_t, py::array::c_style>(A);
        auto mp_activeids = p_activeids.mutable_unchecked<1>();

        for (ssize_t i = 0; i < activeActionIds.size(); ++i)
            mp_activeids(i) = activeActionIds[i];

        LOG("Creating p_dict...");

        return py::dict(
            py::arg("nodes") = p_nodesdict,
            py::arg("edges") = p_edgesdict,
            py::arg("active_action_ids") = p_activeids
        );
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

        ReturnCode res;

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOGFMT("cond%1%.wait(lock%1%)", side);
            res = cond_wait(funcname, side, cond, lock, vcmiTimeout, pred);
            LOGFMT("cond%1%.wait(lock%1%): done", side);
        }

        if (!_error.empty()) {
            throw VCMIConnectorException(_error);
        }

        LOGFMT("release lock%d (return)", side);
        return res;
    }

    std::tuple<int, const std::string> Connector::render(int side) {
        SHUTDOWN_PYTHON_RETURN("");
        auto code = getState(__func__, side, MMAI::Schema::ACTION_RENDER_ANSI);
        const auto * sup = extractSupplementaryData(state);
        assert(sup->getType() == MMAI::Schema::V15::ISupplementaryData::Type::ANSI_RENDER);
        LOG("return state->ansiRender");
        return {static_cast<int>(code), sup->getAnsiRender()};
    }

    std::tuple<int, const py::dict> Connector::reset(int side) {
        SHUTDOWN_PYTHON_RETURN(buildObsDict(state)); // reuse last state if shutting down
        auto code = getState(__func__, side, MMAI::Schema::ACTION_RESET);
        const auto p_dict = buildObsDict(state);
        LOG("return p_dict");
        return {static_cast<int>(code), p_dict};
    }

    std::tuple<int, const py::dict> Connector::step(int side, MMAI::Schema::Action a) {
        SHUTDOWN_PYTHON_RETURN(buildObsDict(state)); // reuse last state if shutting down
        auto code = getState(__func__, side, a);
        const auto p_dict = buildObsDict(state);
        LOG("return p_dict");
        return {static_cast<int>(code), p_dict};
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
    std::tuple<int, const py::dict> Connector::connect(int side) {
        LOG("connect called with side=" + std::to_string(side));

        LOG("obtain lock2");
        std::unique_lock lock2(m2);
        LOG("obtain lock2: done");

        if (side) {
            if (connectedClient1)
                throw std::runtime_error("A client with side 1 is already connected.");
            connectedClient1 = true;
        } else {
            if (connectedClient0)
                throw std::runtime_error("A client with side 0 is already connected.");
            connectedClient0 = true;
        }

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

        {
            LOG("release Python GIL");
            py::gil_scoped_release release;

            LOGFMT("cond%1%.wait(lock%1%)", side);
            res = cond_wait(__func__, side, cond, lock, bootTimeout, pred);
            LOGFMT("cond%1%.wait(lock%1%): done", side);
        }

        auto py_dict = buildObsDict(state);
        LOGFMT("release lock%d (return)", side);
        LOG("return p_dict");
        return {static_cast<int>(res), py_dict};
    }

    void Connector::start() {
        ASSERT_STATE("start", ConnectorState::NEW);

        setvbuf(stdout, nullptr, _IONBF, 0);
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

        LOG("release Python GIL");
        py::gil_scoped_release release;

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

        using Side = MMAI::Schema::Side;

        if (red == "MMAI_RANDOM") {
            leftModel = new ML::ModelWrappers::Function(version(), "MMAI_RANDOM", Side::LEFT, f_getRandomAction, f_getValueDummy);
        } else if (red == "MMAI_USER") {
            leftModel = new ML::ModelWrappers::Function(version(), "MMAI_USER_GYM", Side::LEFT, f_getAction0, f_getValueDummy);
        } else if (red == "MMAI_MODEL") {
            // BAI will load the actual model based on leftModel->getName()
            leftModel = new ML::ModelWrappers::Path(redModel);
        } else {
            leftModel = new ML::ModelWrappers::Scripted(red, Side::LEFT);
        }

        if (blue == "MMAI_RANDOM") {
            rightModel = new ML::ModelWrappers::Function(version(), "MMAI_RANDOM", Side::RIGHT, f_getRandomAction, f_getValueDummy);
        } else if (blue == "MMAI_USER") {
            rightModel = new ML::ModelWrappers::Function(version(), "MMAI_USER_GYM", Side::RIGHT, f_getAction1, f_getValueDummy);
        } else if (blue == "MMAI_MODEL") {
            // BAI will load the actual model based on rightModel->getName()
            rightModel = new ML::ModelWrappers::Path(blueModel);
        } else {
            rightModel = new ML::ModelWrappers::Scripted(blue, Side::RIGHT);
        }

        // auto oldcwd = std::filesystem::current_path();

        auto initargs = ML::InitArgs{
            .leftAllowMlBot=redAllowMlBot,
            .rightAllowMlBot=blueAllowMlBot,
            .mapname=mapname,
            .maxBattles=0,
            .seed=seed,
            .randomHeroes=randomHeroes,
            .randomObstacles=randomObstacles,
            .townChance=townChance,
            .warmachineChance=warmachineChance,
            .randomStackChance=randomStackChance,
            .tightFormationChance=tightFormationChance,
            .randomTerrainChance=randomTerrainChance,
            .leftVipChance=leftVipChance,
            .rightVipChance=rightVipChance,
            .battlefieldPattern=battlefieldPattern,
            .manaMin=manaMin,
            .manaMax=manaMax,
            .randomPrimarySkills=randomPrimarySkills,
            .swapSides=swapSides,
            .loglevelGlobal=loglevelGlobal,
            .loglevelAI=loglevelAI,
            .loglevelStats=loglevelStats,
            .statsMode=statsMode,
            .statsStorage=statsStorage,
            .statsTimeout=60000,
            .statsPersistFreq=statsPersistFreq,
            .headless=true,
        };

        // This must happen in the main thread (SDL requires it)
        LOG("call init_vcmi(...)");
        init_vcmi(leftModel, rightModel, initargs);

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
