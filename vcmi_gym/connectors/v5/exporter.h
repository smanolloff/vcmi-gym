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

#include "schema/base.h"
#include "schema/v5/types.h"
#include "schema/v5/constants.h"

namespace Connector::V5 {
    using namespace MMAI::Schema;
    using namespace MMAI::Schema::V5;

    // attrname, encname, offset, n, vmax
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
        virtual const int getStateSizeMisc() const;
        virtual const int getStateSize() const;
        virtual const int getStateSizeOneHex() const;
        virtual const int getStateSizeAllHexes() const;
        virtual const int getStateSizeOneStack() const;
        virtual const int getStateSizeAllStacks() const;
        virtual const int getStateValueNa() const;
        virtual const int getSideLeft() const;
        virtual const int getSideRight() const;
        virtual const std::vector<std::string> getPrimaryActions() const;
        virtual const std::vector<std::string> getAMoveActions() const;
        virtual const std::vector<std::string> getHexStates() const;
        virtual const std::vector<AttributeMapping> getMiscAttributeMapping() const;
        virtual const std::vector<AttributeMapping> getHexAttributeMapping() const;
        virtual const std::vector<AttributeMapping> getStackAttributeMapping() const;
        virtual const std::vector<FlagMapping> getStackFlagMapping() const;

    protected:
        const std::string getEncodingName(Encoding e) const;
    };
}
