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

#include "mmai_export.h" // "vendor" header file
#include "conncommon.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

constexpr int get_n_actions() { return MMAI::Export::N_ACTIONS; }
constexpr int get_n_nonhex_actions() { return MMAI::Export::N_NONHEX_ACTIONS; }
constexpr int get_n_hex_actions() { return MMAI::Export::N_HEX_ACTIONS; }
constexpr int get_state_size_default() { return MMAI::Export::STATE_SIZE_DEFAULT; }
constexpr int get_state_size_default_one_hex() { return MMAI::Export::STATE_SIZE_DEFAULT_ONE_HEX; }
constexpr int get_state_size_float() { return MMAI::Export::STATE_SIZE_FLOAT; }
constexpr int get_state_size_float_one_hex() { return MMAI::Export::STATE_SIZE_FLOAT_ONE_HEX; }
constexpr int get_state_value_na() { return MMAI::Export::STATE_VALUE_NA; }
constexpr auto get_encoding_type_default() { return MMAI::Export::STATE_ENCODING_DEFAULT; }
constexpr auto get_encoding_type_float() { return MMAI::Export::STATE_ENCODING_FLOAT; }

static const std::map<MMAI::Export::ErrMask, std::tuple<std::string, std::string>> get_error_mapping() {
    std::map<MMAI::Export::ErrMask, std::tuple<std::string, std::string>> res;

    for (auto& [_err, tuple] : MMAI::Export::ERRORS) {
        res[std::get<0>(tuple)] = {std::get<1>(tuple), std::get<2>(tuple)};
    }

    return res;
}

// This module is designed to be importable multiple times in the same process
// (not the case with connector, which can't be imported twice in the same PID)
PYBIND11_MODULE(connexport, m) {
        m.def("get_n_actions", &get_n_actions);
        m.def("get_n_nonhex_actions", &get_n_nonhex_actions);
        m.def("get_n_hex_actions", &get_n_hex_actions);
        m.def("get_state_size_default", &get_state_size_default);
        m.def("get_state_size_default_one_hex", &get_state_size_default_one_hex);
        m.def("get_state_size_float", &get_state_size_float);
        m.def("get_state_size_float_one_hex", &get_state_size_float_one_hex);
        m.def("get_state_value_na", &get_state_value_na);
        m.def("get_encoding_type_default", &get_encoding_type_default);
        m.def("get_encoding_type_float", &get_encoding_type_float);
        m.def("get_error_mapping", &get_error_mapping, "Get available error names and flags");
}
