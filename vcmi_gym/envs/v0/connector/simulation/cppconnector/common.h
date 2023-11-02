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

#include "callbacks.h"

#define LOG(msg) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, msg);
#define LOGSTR(msg, a1) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, (std::string(msg) + a1).c_str());

namespace py = pybind11;

py::array_t<float> stateForPython(StateF statef);
ActionF actionFromPython(py::array_t<float> pyaction);

struct Action {
    Action(std::string str_, py::array_t<float> pyaction)
    : str(str_), action(actionFromPython(pyaction)) {}

    const std::string str;
    const ActionF action;
};

struct State {
    State(std::string str_, StateF statef)
    : str(str_), state(stateForPython(statef)) {}

    const std::string str;
    const py::array_t<float> state;

    const std::string &getStr() const { return str; }
    const py::array_t<float> &getState() const { return state; }
};

using WCppCB = const std::function<void(Action)>;
using WPyCB = const std::function<void(State)>;
using WPyCBInit = const std::function<void(WCppCB)>;

// TODO:
// declare a PyCallbackProvider class/struct
// it will have a .getInitCB() and .getCB() methods
// BUT it will be passed through as std::any
// (all the way to AI, where it will be casted to a PyCallbackProvider again)
