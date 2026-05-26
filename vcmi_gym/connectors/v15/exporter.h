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

#include <pybind11/pybind11.h>

// pybind11 modules need to be bound in a .cpp file
namespace Connector::V15 {
    namespace py = pybind11;

    enum class ReturnCode : uint8_t {
        OK = 0,
        SHUTDOWN,
        TIMEOUT,
        INTERRUPTED,
        ERROR
    };

    int getVersion();
    int getMaxRounds();
    py::dict getNodeTypes();
    py::dict getEdgeTypes();
    py::dict getActionTypes();
    py::dict getCombatResults();

    void bindExporterV15(py::module_ & m);
}
