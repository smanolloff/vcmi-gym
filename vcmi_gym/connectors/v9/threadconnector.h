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
#include <condition_variable>
#include <thread>
#include <cstdio>
#include <iostream>
#include <atomic>
#include <deque>

#include "schema/base.h"
#include "schema/v9/types.h"
#include "common.h"
#include "exporter.h"

namespace Connector::V9::Thread {
    enum ConnectorState {
        NEW,
        AWAITING_ACTION_0,
        AWAITING_ACTION_1,
        AWAITING_STATE,
    };

    class VCMIConnectorException : public std::exception {
        std::string msg_;
    public:
        VCMIConnectorException(const std::string& message) : msg_(message) {}
        virtual const char* what() const noexcept override { return msg_.c_str(); }
    };

    class Connector {
        std::mutex m0;
        std::mutex m1;
        std::mutex m2;
        std::mutex mlog;
        std::condition_variable cond0;
        std::condition_variable cond1;
        std::condition_variable cond2;

        bool connectedClient0;
        bool connectedClient1;

        std::atomic<bool> _shutdown = false;

        // This var is set in getAction (call from VCMI) instead of throwing
        // directly, as throws are not handled well in that case
        std::string _error;

        ConnectorState connstate = ConnectorState::NEW;
        std::deque<std::string> logs {};

        const int maxlogs;
        const int bootTimeout;
        const int vcmiTimeout;
        const int userTimeout;
        const std::string mapname;
        const int seed;
        const int randomHeroes;
        const int randomObstacles;
        const int townChance;
        const int warmachineChance;
        const int tightFormationChance;
        const int randomTerrainChance;
        const std::string battlefieldPattern;
        const int manaMin;
        const int manaMax;
        const int swapSides;
        const std::string loglevelGlobal;
        const std::string loglevelAI;
        const std::string loglevelStats;
        const std::string red;
        const std::string blue;
        const std::string redModel;
        const std::string blueModel;
        const std::string statsMode;
        const std::string statsStorage;
        const int statsPersistFreq;

        std::thread vcmithread;
        MMAI::Schema::IModel* leftModel;
        MMAI::Schema::IModel* rightModel;
        MMAI::Schema::Action action;
        const MMAI::Schema::IState * state;

        const P_State convertState(const MMAI::Schema::IState * r);
        MMAI::Schema::Action getAction(const MMAI::Schema::IState * r, int side);
        const MMAI::Schema::Action getActionDummy(MMAI::Schema::IState);
        const MMAI::Schema::V9::ISupplementaryData* extractSupplementaryData(const MMAI::Schema::IState *s);

        // essentially, all of .reset(), .render() and .step() are a form of getState
        ReturnCode getState(const char* funcname, int side, int action);

        // XXX: cond_wait without a predicate is prone to race conditions
        // int cond_wait(
        //     const char* func,
        //     int id,
        //     std::condition_variable &cond,
        //     std::unique_lock<std::mutex> &l,
        //     int timeoutSeconds  // -1 = no timeout
        // );

        ReturnCode cond_wait(
            const char* func,
            int id,
            std::condition_variable &cond,
            std::unique_lock<std::mutex> &l,
            int timeoutSeconds, // -1 = no timeout
            std::function<bool()> &pred
        );

        ReturnCode _cond_wait(
            const char* func,
            int id,
            std::condition_variable &cond,
            std::unique_lock<std::mutex> &l,
            int timeoutSeconds, // -1 = no timeout
            std::function<bool()> &checker
        );

        void maybeThrowError();
        void log(std::string funcname, std::string msg);

        // void signal_handler(int signal);
    public:
        Connector(
            const int maxlogs,
            const int bootTimeout,
            const int vcmiTimeout,
            const int userTimeout,
            const std::string mapname,
            const int seed,
            const int randomHeroes,
            const int randomObstacles,
            const int townChance,
            const int warmachineChance,
            const int tightFormationChance,
            const int randomTerrainChance,
            const std::string battlefieldPattern,
            const int manaMin,
            const int manaMax,
            const int swapSides,
            const std::string loglevelGlobal,
            const std::string loglevelAI,
            const std::string loglevelStats,
            const std::string red,
            const std::string blue,
            const std::string redModel,
            const std::string blueModel,
            const std::string statsMode,
            const std::string statsStorage,
            const int statsPersistFreq
        );

        // timeouts are in seconds
        void start();
        const std::tuple<int, P_State> connect(int side);
        const std::tuple<int, P_State> reset(int side);
        const std::tuple<int, P_State> step(int side, const MMAI::Schema::Action a);
        const std::tuple<int, std::string> render(int side);
        void shutdown();
        const std::vector<std::string> getLogs();

        virtual const int version() { return 9; };
        virtual ~Connector() = default;
    };
}
