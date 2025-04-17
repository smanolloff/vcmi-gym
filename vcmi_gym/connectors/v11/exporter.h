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

// Linux builds fail without this include (used in types.h)
#include <bitset>

#include "schema/base.h"
#include "schema/v11/types.h"
#include "schema/v11/constants.h"

namespace Connector::V11 {
    using namespace MMAI::Schema;
    using namespace MMAI::Schema::V11;

    using AttributeMapping = std::tuple<std::string, std::string, int, int, int>;

    // attrname, offset
    using FlagMapping = std::tuple<std::string, int>;

    enum class ReturnCode {
        OK = 0,
        SHUTDOWN,
        TIMEOUT,
        INTERRUPTED,
        ERROR
    };

    class Exporter {
    public:
        virtual const int getVersion() const;
        virtual const int getNActions() const;
        virtual const int getNNonhexActions() const;
        virtual const int getNHexActions() const;
        virtual const int getStateSize() const;
        virtual const int getStateSizeOneHex() const;
        virtual const int getStateSizeAllHexes() const;
        virtual const int getStateSizeOnePlayer() const;
        virtual const int getStateSizeGlobal() const;
        virtual const int getStateValueNa() const;
        virtual const int getSideLeft() const;
        virtual const int getSideRight() const;
        virtual const std::vector<std::string> getGlobalActions() const;
        virtual const std::vector<std::string> getHexActions() const;
        virtual const std::vector<std::string> getHexStates() const;
        virtual const std::vector<AttributeMapping> getHexAttributeMapping() const;
        virtual const std::vector<AttributeMapping> getPlayerAttributeMapping() const;
        virtual const std::vector<AttributeMapping> getGlobalAttributeMapping() const;
        virtual const std::vector<FlagMapping> getStackFlag1Mapping() const;
        virtual const std::vector<FlagMapping> getStackFlag2Mapping() const;

    protected:
        const std::string getEncodingName(Encoding e) const;
    };
}
