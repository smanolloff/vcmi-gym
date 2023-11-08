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

static const int get_state_size() { return MMAI::STATE_SIZE; }
static const int get_action_max() { return MMAI::N_ACTIONS; }

using P_Action = int;
using P_State = py::array_t<float>;
struct P_Result {
  P_Result(
    P_State state,
    int n_errors,
    int dmg_dealt,
    int dmg_received,
    int units_lost,
    int units_killed,
    int value_lost,
    int value_killed,
    bool is_battle_over,
    bool is_victorious
  )
  : _state(state),
    _n_errors(n_errors),
    _dmg_dealt(dmg_dealt),
    _dmg_received(dmg_received),
    _is_battle_over(is_battle_over),
    _units_lost(units_lost),
    _units_killed(units_killed),
    _value_lost(value_lost),
    _value_killed(value_killed) {}

  const py::array_t<float> _state;
  const int _n_errors = -1;
  const int _dmg_dealt = -1;
  const int _dmg_received = -1;
  const int _units_lost = -1;
  const int _units_killed = -1;
  const int _value_lost = -1;
  const int _value_killed = -1;
  const bool _is_battle_over = false;
  const bool _is_victorious = false;

  const py::array_t<float> &state() const { return _state; }
  const int &n_errors() const { return _n_errors; }
  const int &dmg_dealt() const { return _dmg_dealt; }
  const int &dmg_received() const { return _dmg_received; }
  const int &units_lost() const { return _units_lost; }
  const int &units_killed() const { return _units_killed; }
  const int &value_lost() const { return _value_lost; }
  const int &value_killed() const { return _value_killed; }
  const bool &is_battle_over() const { return _is_battle_over; }
  const bool &is_victorious() const { return _is_victorious; }
};

// Wrappers of functions called from/by CPP code
// See notes in aitypes.h

using P_RenderAnsiCB = const std::function<std::string()>;
using P_RenderAnsiCBCB = const std::function<void(P_RenderAnsiCB)>;

using P_ResetCB = const std::function<void()>;
using P_ResetCBCB = const std::function<void(P_ResetCB)>;

using P_SysCB = const std::function<void(std::string)>;
using P_SysCBCB = const std::function<void(P_SysCB)>;

using P_ActionCB = const std::function<void(P_Action)>;
using P_ActionCBCB = const std::function<void(P_ActionCB)>;
using P_ResultCB = const std::function<void(P_Result)>;
