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

#include "../v1/exporter.h"

namespace Connector::V2 {
    class Exporter : public V1::Exporter {
    public:
        using V1::Exporter::Exporter;
        const int getVersion() const override;
        const int getStateSize() const override;
        const int getStateSizeOneHex() const override;
        const std::vector<V1::AttributeMapping> getAttributeMapping() const override;
    };
}
