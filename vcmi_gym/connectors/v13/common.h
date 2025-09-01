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
#include "schema/v13/types.h"

#include <pybind11/numpy.h>

#ifdef DEBUG_BUILD
    #define VERBOSE 1       // whether to output logs to STDOUT
    #define LOGCOLLECT 0    // whether to record logs in memory
#else
    #define VERBOSE 0
    // #define LOGCOLLECT 1
    #define LOGCOLLECT 0
#endif

#if VERBOSE || LOGCOLLECT
    #define LOG(msg) log(__func__, msg);
    #define LOGFMT(fmt, elems) LOG(boost::str(boost::format(fmt) % elems));
#else
    #define LOG(msg) // noop
    #define LOGFMT(fmt, elems) // noop
#endif

namespace Connector::V13 {
    namespace py = pybind11;

    using P_BattlefieldState = py::array_t<float>;
    using P_ActionMask = py::array_t<bool>;


    // XXX: these are also py::array_t, but are 2D
    using P_IntermediateStates = py::array_t<float>;        // shape (d0, STATE_SIZE)
    using P_IntermediateActionMasks = py::array_t<bool>;    // shape (d0, N_ACTIONS)
    using P_IntermediateActions = py::array_t<int64_t>;     // shape (d0,)

    using P_LinksDict = py::dict;
    // {
    //      "ADJACENT": (
    //          "index": [[src0, src1, ...], [dst0, dst1, ...]],    // shape (2, num_links)
    //          "attrs": [[attr0], [attr1], ...],                   // shape (1, num_links,)
    //      ),
    //      "REACH": ...
    // }

    MMAI::Schema::Action RandomValidAction(const MMAI::Schema::IState * s);

    class P_State {
    public:
        P_State(
            MMAI::Schema::V13::ISupplementaryData::Type type_,
            P_IntermediateStates intstates_,
            P_IntermediateActionMasks intmasks_,
            P_IntermediateActions intactions_,
            P_LinksDict linksDict_,
            const MMAI::Schema::V13::ErrorCode errcode_,
            const std::string ansiRender_
        ) : type(type_)
          , intstates(intstates_)
          , intmasks(intmasks_)
          , intactions(intactions_)
          , linksDict(linksDict_)
          , errcode(static_cast<int>(errcode_))
          , ansiRender(ansiRender_) {}

        const MMAI::Schema::V13::ISupplementaryData::Type type;
        const P_BattlefieldState state;
        const P_ActionMask actmask;
        const P_IntermediateStates intstates;
        const P_IntermediateActionMasks intmasks;
        const P_IntermediateActions intactions;
        const P_LinksDict linksDict;
        const int errcode;
        const std::string ansiRender;

        const P_IntermediateStates get_intermediate_states() const { return intstates; }
        const P_IntermediateActionMasks get_intermediate_action_masks() const { return intmasks; }
        const P_IntermediateActions get_intermediate_actions() const { return intactions; }
        const P_LinksDict get_links_dict() const { return linksDict; }

        const int get_errcode() const { return errcode; }
    };
}
