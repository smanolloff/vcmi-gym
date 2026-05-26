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

#include <random>
#include <iostream>
#include "common.h"

namespace Connector::V15 {
    MMAI::Schema::Action RandomValidAction(const MMAI::Schema::IState * s) {
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
        std::uniform_int_distribution<> dist(0, activeIds.size() - 1);
        int randomIndex = dist(gen);
        const auto id = activeIds[randomIndex];
        return id;
    }
}
