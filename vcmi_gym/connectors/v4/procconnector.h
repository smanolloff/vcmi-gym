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

#include "AI/MMAI/schema/base.h"
#include "AI/MMAI/schema/v4/types.h"
#include "common.h"

namespace Connector::V4::Proc {
    enum ConnectorState {
        NEW,
        AWAITING_ACTION,
        AWAITING_STATE,
    };

    class Connector {
        std::mutex m;
        std::condition_variable cond;

        ConnectorState connstate = ConnectorState::NEW;

        const std::string mapname;
        const int seed;
        const int randomHeroes;
        const int randomObstacles;
        const int townChance;
        const int warmachineChance;
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
        MMAI::Schema::Action getAction(const MMAI::Schema::IState * r);
        const MMAI::Schema::Action getActionDummy(MMAI::Schema::IState);
        const MMAI::Schema::V4::ISupplementaryData* extractSupplementaryData(const MMAI::Schema::IState *s);
    public:
        Connector(
            const std::string mapname,
            const int seed,
            const int randomHeroes,
            const int randomObstacles,
            const int townChance,
            const int warmachineChance,
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

        const P_State start();
        const P_State reset();
        const P_State getState(const MMAI::Schema::Action a);
        const std::string renderAnsi();

        virtual const int version();
        virtual ~Connector() = default;
    };
}
