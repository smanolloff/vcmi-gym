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

#include "common.h"

namespace Connector::V5 {
    MMAI::Schema::Action RandomValidAction(const MMAI::Schema::IState * s) {
        using PA = MMAI::Schema::V5::PrimaryAction;
        auto mask = s->getActionMask();

        // // DEBUG - move to self (1-stack army only)
        // return (75 << 8) | EI(PA::MOVE);

        auto validPrimaryActions = std::vector<int>{};

        // start from (skip RETREAT)
        static_assert(EI(PA::RETREAT) == 0);
        for (int i = 1; i < EI(PA::_count); i++) {
            if (mask.at(i))
                validPrimaryActions.push_back(i);
        }

        if (validPrimaryActions.empty()) {
            return MMAI::Schema::ACTION_RESET;
        }

        int primaryAction;

        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, validPrimaryActions.size() - 1);
            int randomIndex = dist(gen);
            primaryAction = validPrimaryActions.at(randomIndex);
        }

        if (primaryAction < EI(PA::MOVE))
            // no hex needed
            return primaryAction;

        using MA = MMAI::Schema::V5::MiscAttribute;

        constexpr auto n0 = std::get<2>(MMAI::Schema::V5::MISC_ENCODING.at(EI(MA::PRIMARY_ACTION_MASK)));
        constexpr auto n1 = std::get<2>(MMAI::Schema::V5::MISC_ENCODING.at(EI(MA::SHOOTING)));
        static_assert(EI(MA::PRIMARY_ACTION_MASK) == 0);
        static_assert(EI(MA::SHOOTING) == 1);
        static_assert(n1 == 1);
        // either 0.0 or 1.0, but compare with 0.5
        // (to avoid floating-point issues)
        auto shooting = s->getBattlefieldState().at(n0) > 0.5;

        if (primaryAction > EI(PA::MOVE) && shooting)
            // no hex needed
            return primaryAction;

        using AMA = MMAI::Schema::V5::AMoveAction;
        auto validHexes = std::vector<int>{};

        for (int ihex = 0; ihex < 165; ihex++) {
            auto ibase = EI(PA::_count) + ihex * EI(AMA::_count);
            auto i = ibase + primaryAction - EI(PA::MOVE);
            if (mask.at(i))
                validHexes.push_back(ihex);
        }

        int hex;

        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, validHexes.size() - 1);
            int randomIndex = dist(gen);
            hex = validHexes.at(randomIndex);
        }

        return (hex << 8) | primaryAction;
    }
}
