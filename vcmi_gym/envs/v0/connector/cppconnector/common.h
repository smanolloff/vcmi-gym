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

#define LOG(msg) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, msg);
#define LOGSTR(msg, a1) printf("<%s>[CPP][%s] (%s) %s\n", boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str(), std::filesystem::path(__FILE__).filename().string().c_str(), __FUNCTION__, (std::string(msg) + a1).c_str());

namespace py = pybind11;

using PyGymAction = int;
using PyGymState = py::array_t<float>;

using WCppCB = const std::function<void(PyGymAction)>;
using WCppSysCB = const std::function<void(std::string)>;
using WPyCB = const std::function<void(PyGymState)>;
using WPyCBInit = const std::function<void(WCppCB)>;
using WPyCBSysInit = const std::function<void(WCppSysCB)>;
