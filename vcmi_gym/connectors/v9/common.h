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
#include "schema/v9/types.h"

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

namespace Connector::V9 {
    namespace py = pybind11;

    using P_BattlefieldState = py::array_t<float>;
    using P_ActionMask = py::array_t<bool>;
    // using P_EdgeSources = py::array_t<int>;         // index src (int)
    // using P_EdgeTargets = py::array_t<int>;         // index dst (int)
    // using P_EdgeValues = py::array_t<float>;        // value (float)
    // using P_EdgeTypesOH = py::array_t<int>;         // edge type (one-hot)
    using P_EdgeIndex = py::array_t<long>;      // contatenated index_src and index_dst (int)
    using P_EdgeAttrs = py::array_t<float>;     // value + edge type

    MMAI::Schema::Action RandomValidAction(const MMAI::Schema::IState * s);

    class P_State {
    public:
        P_State(
            MMAI::Schema::V9::ISupplementaryData::Type type_,
            P_BattlefieldState state_,
            P_ActionMask actmask_,
            // P_EdgeSources edgesources_,
            // P_EdgeTargets edgetargets_,
            // P_EdgeValues edgevalues_,
            // P_EdgeTypesOH edgetypes_,
            P_EdgeIndex edgeindex_,
            P_EdgeAttrs edgeattrs_,
            const MMAI::Schema::V9::ErrorCode errcode_,
            const std::string ansiRender_
        ) : type(type_)
          , state(state_)
          , actmask(actmask_)
          // , edgesources(edgesources_)
          // , edgetargets(edgetargets_)
          // , edgevalues(edgevalues_)
          // , edgetypes(edgetypes_)
          , edgeindex(edgeindex_)
          , edgeattrs(edgeattrs_)
          , errcode(static_cast<int>(errcode_))
          , ansiRender(ansiRender_) {}

        const MMAI::Schema::V9::ISupplementaryData::Type type;
        const P_BattlefieldState state;
        const P_ActionMask actmask;
        // const P_EdgeSources edgesources;
        // const P_EdgeTargets edgetargets;
        // const P_EdgeValues edgevalues;
        // const P_EdgeTypesOH edgetypes;
        const P_EdgeIndex edgeindex;
        const P_EdgeAttrs edgeattrs;
        const int errcode;
        const std::string ansiRender;

        const P_BattlefieldState get_state() const { return state; }
        const P_ActionMask get_actmask() const { return actmask; }
        // const P_EdgeSources get_edge_sources() const { return edgesources; }
        // const P_EdgeTargets get_edge_targets() const { return edgetargets; }
        // const P_EdgeValues get_edge_values() const { return edgevalues; }
        // const P_EdgeTypesOH get_edge_types() const { return edgetypes; }
        const P_EdgeIndex get_edge_index() const { return edgeindex; }
        const P_EdgeAttrs get_edge_attrs() const { return edgeattrs; }

        const int get_errcode() const { return errcode; }
    };
}
