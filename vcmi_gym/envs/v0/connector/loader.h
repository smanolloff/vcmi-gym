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

/*****
****** THIS FILE LIVES IN:
******
****** connector/loader.h
******
*****/

#ifdef __LOADED_FROM_MMAI
#include "export.h"
#else
#include "mmai_export.h" // "vendor" header file
#endif

// internal
void init(std::string encoding, MMAI::Export::Side side, std::string gymdir, std::string modelfile);
MMAI::Export::Action getAction(MMAI::Export::Side &side, const MMAI::Export::Result* &r);

// external
// use long names to avoid symbol collisions

extern "C" __attribute__((visibility("default"))) void ConnectorLoader_initAttacker(
    std::string encoding,
    std::string gymdir,
    std::string modelfile
);

extern "C" __attribute__((visibility("default"))) void ConnectorLoader_initDefender(
    std::string encoding,
    std::string gymdir,
    std::string modelfile
);

extern "C" __attribute__((visibility("default"))) MMAI::Export::Action ConnectorLoader_getActionAttacker(
    const MMAI::Export::Result* r
);

extern "C" __attribute__((visibility("default"))) MMAI::Export::Action ConnectorLoader_getActionDefender(
    const MMAI::Export::Result* r
);
