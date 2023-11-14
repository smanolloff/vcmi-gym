#pragma once

#include <filesystem>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <condition_variable>
#include <thread>

#define DLL_EXPORT __attribute__ ((visibility("default")))
#include "aitypes.h" // "vendor" header file

#define VERBOSE true
#define LOG(msg) if(VERBOSE) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(std::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, msg);
#define LOGSTR(msg, a1) if (VERBOSE) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(std::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, (std::string(msg) + a1).c_str());

namespace py = pybind11;

using P_State = py::array_t<float>;

struct P_Result {
  P_Result(
    MMAI::ResultType type_,
    P_State state_,
    uint8_t errmask_,
    int dmg_dealt_,
    int dmg_received_,
    int units_lost_,
    int units_killed_,
    int value_lost_,
    int value_killed_,
    bool is_battle_over_,
    bool is_victorious_,
    std::string ansiRender_
  )
  : type(type_),
    errmask(errmask_),
    state(state_),
    dmg_dealt(dmg_dealt_),
    dmg_received(dmg_received_),
    is_battle_over(is_battle_over_),
    units_lost(units_lost_),
    units_killed(units_killed_),
    value_lost(value_lost_),
    value_killed(value_killed_),
    is_victorious(is_victorious_),
    ansiRender(ansiRender_) {}

  const MMAI::ResultType type;
  const py::array_t<float> state;
  const uint8_t errmask;
  const int dmg_dealt;
  const int dmg_received;
  const int units_lost;
  const int units_killed;
  const int value_lost;
  const int value_killed;
  const bool is_battle_over;
  const bool is_victorious;
  const std::string ansiRender;

  const py::array_t<float> &get_state() const { return state; }
  const uint8_t &get_errmask() const { return errmask; }
  const int &get_dmg_dealt() const { return dmg_dealt; }
  const int &get_dmg_received() const { return dmg_received; }
  const int &get_units_lost() const { return units_lost; }
  const int &get_units_killed() const { return units_killed; }
  const int &get_value_lost() const { return value_lost; }
  const int &get_value_killed() const { return value_killed; }
  const bool &get_is_battle_over() const { return is_battle_over; }
  const bool &get_is_victorious() const { return is_victorious; }
};

enum ConnectorState {
  NEW,
  AWAITING_ACTION,
  AWAITING_RESULT,
};

class CppConnector {
  std::mutex m1;
  std::mutex m2;
  std::condition_variable cond1;
  std::condition_variable cond2;

  ConnectorState state = ConnectorState::NEW;

  const std::string mapname;
  const std::string loglevel;
  std::thread vcmithread;
  MMAI::F_Sys f_sys;
  std::unique_ptr<MMAI::CBProvider> cbprovider = std::make_unique<MMAI::CBProvider>(nullptr, "from CppConnector (DEFAULT)");
  MMAI::Action action;
  MMAI::Result result;

  const P_Result convertResult(MMAI::Result);
  MMAI::Action getAction(MMAI::Result);
  const MMAI::Action getActionDummy(MMAI::Result);

public:
  CppConnector(const std::string mapname, const std::string loglevel);
  ~CppConnector();

  const P_Result start();
  const P_Result reset();
  const P_Result act(const MMAI::Action a);
  const std::string renderAnsi();
};
