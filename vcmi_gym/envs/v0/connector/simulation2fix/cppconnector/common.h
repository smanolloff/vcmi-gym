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

using CppCB = const std::function<void()>;
using PyCB = const std::function<void()>;
using PyCBInit = const std::function<void(CppCB)>;

using WCppCB = const std::function<void()>;
using WPyCB = const std::function<void()>;
using WPyCBInit = const std::function<void(WCppCB)>;

// TODO:
// declare a PyCallbackProvider class/struct
// it will have a .getInitCB() and .getCB() methods
// BUT it will be passed through as std::any
// (all the way to AI, where it will be casted to a PyCallbackProvider again)
