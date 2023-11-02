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

pybind11_add_module(connsimulator
  simulation/cppconnector/common.h
  simulation/cppconnector/ai_simulator.h
  simulation/cppconnector/ai_simulator.cpp
  simulation/cppconnector/client_simulator.h
  simulation/cppconnector/client_simulator.cpp
  simulation/cppconnector/server_simulator.h
  simulation/cppconnector/server_simulator.cpp
  simulation/cppconnector/connector.h
  simulation/cppconnector/connector.cpp)
target_include_directories(connsimulator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(connsimulator PRIVATE myclient Boost::boost Boost::thread Boost::filesystem)

pybind11_add_module(conntest
  testing/cppconnector/common.h
  testing/cppconnector/cppconnector.h
  testing/cppconnector/cppconnector.cpp)
target_include_directories(conntest PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(conntest PRIVATE pyclient Boost::boost Boost::thread Boost::filesystem)

add_executable(conntestexe
  testing/cppconnector/common.h
  testing/cppconnector/cppconnector.h
  testing/cppconnector/main.h
  testing/cppconnector/main.cpp)

target_include_directories(conntestexe PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(conntestexe PRIVATE pyclient Boost::boost Boost::thread Boost::filesystem)
