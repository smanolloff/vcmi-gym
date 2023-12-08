#define DLL_EXPORT __attribute__ ((visibility("default")))

#include "mmai_export.h" // "vendor" header file
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

static const int get_state_size() { return MMAI::Export::STATE_SIZE; }
static const int get_n_actions() { return MMAI::Export::N_ACTIONS; }
static const int get_n_nonhex_actions() { return MMAI::Export::N_NONHEX_ACTIONS; }
static const int get_n_hex_actions() { return MMAI::Export::N_HEX_ACTIONS; }
static const int get_n_stack_attrs() { return MMAI::Export::N_STACK_ATTRS; }
static const int get_n_hex_attrs() { return MMAI::Export::N_HEX_ATTRS; }
static const int get_nv_min() { return MMAI::Export::NV_MIN; }
static const int get_nv_max() { return MMAI::Export::NV_MAX; }

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
        m.def("get_state_size", &get_state_size);
        m.def("get_n_actions", &get_n_actions);
        m.def("get_n_nonhex_actions", &get_n_nonhex_actions);
        m.def("get_n_hex_actions", &get_n_hex_actions);
        m.def("get_n_stack_attrs", &get_n_stack_attrs);
        m.def("get_n_hex_attrs", &get_n_hex_attrs);
        m.def("get_nv_min", &get_nv_min);
        m.def("get_nv_max", &get_nv_max);
        m.def("get_error_mapping", &get_error_mapping, "Get available error names and flags");
}
