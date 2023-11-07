cmake_minimum_required(VERSION 3.16.0)
project(connector)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

# https://stackoverflow.com/a/69527824
set(Python_VIRTUALENV FIRST)
set(CMAKE_CXX_STANDARD 17)

# Seems python3.11 is not supported by pybind (as of Oct 2023):
# https://github.com/pybind/pybind11/discussions/4333
link_directories(${CMAKE_SOURCE_DIR}/lib)
find_package(Python3 3.10 REQUIRED EXACT COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Boost 1.48.0 REQUIRED COMPONENTS filesystem thread)

pybind11_add_module(cppconnector
  cppconnector/common.h
  cppconnector/cppconnector.cpp)
target_include_directories(cppconnector PUBLIC ${CMAKE_SOURCE_DIR}/../vcmi/pyclient)
target_link_libraries(cppconnector PRIVATE pyclient Boost::boost Boost::thread Boost::filesystem)
