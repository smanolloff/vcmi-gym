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

#include "common.h"

namespace Connector::V7 {
    MMAI::Schema::Action RandomValidAction(const MMAI::Schema::IState * s) {
        auto validActions = std::vector<MMAI::Schema::Action>{};
        auto mask = s->getActionMask();

        for (int j = 1; j < mask.size(); j++) {
            if (mask[j])
                validActions.push_back(j);
        }

        if (validActions.empty()) {
            return MMAI::Schema::ACTION_RESET;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, validActions.size() - 1);
        int randomIndex = dist(gen);
        return validActions[randomIndex];
    }
}
