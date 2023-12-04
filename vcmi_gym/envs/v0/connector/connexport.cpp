#define DLL_EXPORT __attribute__ ((visibility("default")))

#include "mmai_export.h" // "vendor" header file
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

static const int get_state_size() { return MMAI::Export::STATE_SIZE; }
static const int get_n_actions() { return MMAI::Export::N_ACTIONS; }
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
        m.def("get_state_size", &get_state_size, "Get number of elements in state");
        m.def("get_n_actions", &get_n_actions, "Get max expected value of action");
        m.def("get_error_mapping", &get_error_mapping, "Get available error names and flags");
}
