#pragma once

#include <cstdio>
#include <memory>
#include <filesystem>
#include <pybind11/numpy.h>
#include <sstream>
#include <array>
#include <any>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#define DLL_EXPORT __attribute__ ((visibility("default")))
#include "aitypes.h" // "vendor" header file

#define VERBOSE false
#define LOG(msg) if(VERBOSE) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, msg);
#define LOGSTR(msg, a1) if (VERBOSE) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, (std::string(msg) + a1).c_str());

namespace py = pybind11;

static const int get_state_size() { return MMAI::stateSize; }
static const int get_action_max() { return MMAI::actionMax; }

using PyGymAction = int;
using PyGymState = py::array_t<float>;
struct PyGymResult {
  PyGymResult(
    PyGymState state,
    int n_errors,
    int dmg_dealt,
    int dmg_received,
    bool is_battle_over,
    bool is_victorious
  )
  : _state(state),
    _n_errors(n_errors),
    _dmg_dealt(dmg_dealt),
    _dmg_received(dmg_received),
    _is_battle_over(is_battle_over),
    _is_victorious(is_victorious) {}

  const py::array_t<float> _state;
  const int _n_errors = -1;
  const int _dmg_dealt = -1;
  const int _dmg_received = -1;
  const bool _is_battle_over = false;
  const bool _is_victorious = false;

  const py::array_t<float> &state() const { return _state; }
  const int &n_errors() const { return _n_errors; }
  const int &dmg_dealt() const { return _dmg_dealt; }
  const int &dmg_received() const { return _dmg_received; }
  const bool &is_battle_over() const { return _is_battle_over; }
  const bool &is_victorious() const { return _is_victorious; }
};

// Wrappers of functions called from/by CPP code
// See notes in aitypes.h

using WCppResetCB = const std::function<void()>;
using WPyCBResetInit = const std::function<void(WCppResetCB)>;

using WCppSysCB = const std::function<void(std::string)>;
using WPyCBSysInit = const std::function<void(WCppSysCB)>;

using WCppCB = const std::function<void(PyGymAction)>;
using WPyCBInit = const std::function<void(WCppCB)>;
using WPyCB = const std::function<void(PyGymResult)>;
