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

#include "conncommon.h"

enum ConnectorState {
    NEW,
    AWAITING_ACTION,
    AWAITING_RESULT,
};

class Connector {
    std::mutex m;
    std::condition_variable cond;

    ConnectorState state = ConnectorState::NEW;

    const std::string encoding;
    const std::string mapname;
    const int seed;
    const int randomHeroes;
    const int randomObstacles;
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
    const int statsSampling;
    const float statsScoreVar;

    std::thread vcmithread;
    std::unique_ptr<MMAI::Export::Baggage> baggage;
    MMAI::Export::Action action;
    const MMAI::Export::Result * result;

    const P_Result convertResult(const MMAI::Export::Result * r);
    MMAI::Export::Action getAction(const MMAI::Export::Result * r);
    const MMAI::Export::Action getActionDummy(MMAI::Export::Result);

    MMAI::Export::Baggage initBaggage();
public:
    Connector(
        const std::string encoding,
        const std::string mapname,
        const int seed,
        const int randomHeroes,
        const int randomObstacles,
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
        const int statsPersistFreq,
        const int statsSampling,
        const float statsScoreVar
    );

    const P_Result start();
    const P_Result reset();
    const P_Result act(const MMAI::Export::Action a);
    const std::string renderAnsi();

    // Called when VcmiGym is started from within VCMI itself
    // (ie. VCMI is started normally, and vcmi-gym is started as its AI)
    const MMAI::Export::Baggage* getCBProvider();
};
