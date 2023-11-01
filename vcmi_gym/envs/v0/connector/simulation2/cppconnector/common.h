#pragma once

#include <cstdio>
#include <memory>
#include <filesystem>
#include <pybind11/numpy.h>
#include <sstream>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#define LOG(msg) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, msg);
#define LOGSTR(msg, a1) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, (std::string(msg) + a1).c_str());

namespace py = pybind11;

using ActionF = float[3];
using StateF = float[3];

// TODO:
// Dont use setter methods, define all members as const
// This means they must be initialized in the constructor
//
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#custom-constructors

// actions are set in python and read by CPP
struct Action {
    void setA(const std::string &a_) { a = a_; }
    void setB(const py::array_t<float> &b_) { b = b_; }

    const std::string &getA() const { return a; }
    const py::array_t<float> &getB() const { return b; }

    std::string a;
    py::array_t<float> b;
};

// actions are set in CPP and read by python
struct State {
    void setA(const std::string &a_) { a = a_; }
    void setB(const py::array_t<float> &b_) { b = b_; }

    const std::string &getA() const { return a; }
    const py::array_t<float> &getB() const { return b; }

    std::string a;
    py::array_t<float> b;
};

using CppCB = const std::function<void(Action)>;
using PyCB = const std::function<void(State)>;
using PyCBInit = const std::function<void(CppCB)>;

using WCppCB = const std::function<void(const ActionF &arr)>;
using WPyCB = const std::function<void(const StateF &arr)>;
using WPyCBInit = const std::function<void(WCppCB)>;

// TODO:
// declare a PyCallbackProvider class/struct
// it will have a .getInitCB() and .getCB() methods
// BUT it will be passed through as std::any
// (all the way to AI, where it will be casted to a PyCallbackProvider again)
