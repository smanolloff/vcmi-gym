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
#include "schema/base.h"
#include "schema/v15/types.h"

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


static std::string to_base36(uint64_t v) {
    if (v == 0) return "0";
    std::string s;
    while (v) {
        auto d = static_cast<unsigned>(v % 36);
        s.push_back(d < 10 ? static_cast<char>('0' + d) : static_cast<char>('a' + (d - 10)));
        v /= 36;
    }
    std::ranges::reverse(s);
    return s;
}

/*
 * native_thread_id() macro returns a thread ID that should be identical to
 * Python's threading.get_native_id().
 * This allows to track thread identities across Python and C++.
 */

#if defined(_WIN32)
  #include <windows.h>
  static uint64_t native_thread_id() { return GetCurrentThreadId(); }

#elif defined(__APPLE__)
  #include <pthread.h>
  static uint64_t native_thread_id() {
      uint64_t tid = 0;
      pthread_threadid_np(nullptr, &tid);
      return tid;
  }

#elif defined(__linux__)
  #include <sys/syscall.h>
  #include <unistd.h>
  static uint64_t native_thread_id() {
  #ifdef SYS_gettid
      return static_cast<uint64_t>(syscall(SYS_gettid));
  #else
      // Fallback: use a portable C++ id hashed (see below) if SYS_gettid is unavailable.
      return 0; // signal "no native id" and use the hash path instead
  #endif
  }

#else
  #include <pthread.h>
  // Last-resort fallback: implementation-defined; do not rely on this across systems.
  static uint64_t native_thread_id() {
      return reinterpret_cast<uint64_t>(pthread_self());
  }
#endif


namespace Connector::V15 {
    namespace py = pybind11;

    MMAI::Schema::Action RandomValidAction(const MMAI::Schema::IState * s);

    class P_State {
    public:
        P_State(
            MMAI::Schema::V15::ISupplementaryData::Type type,

            // {
            //      "HEX":  [[...], [...], ...],                             // shape (D, num_nodes)
            //      "UNIT": [[...], [...], ...],                             // shape (D, num_nodes)
            //      ...
            // }
            py::dict & nodes,


            // edges:
            //      keys = tuple(src_name, edge_name, dst_name)
            //      values = see example below
            //
            // {
            //      ("Hex", "Adjacent", "Hex"): {
            //          "index": [[src0, src1, ...], [dst0, dst1, ...]],    // shape (2, num_edges)
            //          "attrs": [[...], [...], ...],                       // shape (D, num_edges)
            //      },
            //      ...
            // }
            py::dict & edges,

            const std::vector<int64_t> & activeActionIds,
            const std::string & ansiRender
        ) : type(type)
          , nodes(nodes)
          , edges(edges)
          , activeActionIds(activeActionIds)
          , ansiRender(ansiRender) {}

        const MMAI::Schema::V15::ISupplementaryData::Type type;
        const py::dict nodes;
        const py::dict edges;
        const std::string ansiRender;
        const std::vector<int64_t> activeActionIds;

        py::dict get_nodes() const { return nodes; }
        py::dict get_edges() const { return edges; }
        std::vector<int64_t> get_active_action_ids() const { return activeActionIds; }
    };
}
