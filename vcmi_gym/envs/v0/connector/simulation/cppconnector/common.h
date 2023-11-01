#pragma once

#include <cstdio>
#include <memory>
#include <filesystem>
#include <pybind11/numpy.h>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#define LOG(msg) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, msg);
#define LOGSTR(msg, a1) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, (std::string(msg) + a1).c_str());

namespace py = pybind11;

struct Action {
    void setA(const std::string &a_) { a = a_; }
    void setB(const py::array_t<float> &b_) { b = b_; }

    const std::string &getA() const { return a; }
    const py::array_t<float> &getB() const { return b; }

    std::string a;
    py::array_t<float> b;
};

struct State {
    void setA(const std::string &a_) { a = a_; }
    void setB(const py::array_t<float> &b_) { b = b_; }

    const std::string &getA() const { return a; }
    const py::array_t<float> &getB() const { return b; }

    std::string a;
    py::array_t<float> b;
};

using CppCB = std::function<void(Action)>;
using PyCB = std::function<void(State)>;
using PyCBInit = std::function<void(CppCB)>;
