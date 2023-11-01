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

# SIMULATOR
# ORIG one - works
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

# add_executable(conntest main.cpp)
# target_link_libraries(conntest PRIVATE myclient)
# target_include_directories(conntest PRIVATE ${CMAKE_SOURCE_DIR}/include)

# SIMULATOR
# Wrapper cbs one - NOT working
pybind11_add_module(connsimulator2
  simulation2/cppconnector/common.h
  simulation2/cppconnector/ai_simulator.h
  simulation2/cppconnector/ai_simulator.cpp
  simulation2/cppconnector/client_simulator.h
  simulation2/cppconnector/client_simulator.cpp
  simulation2/cppconnector/server_simulator.h
  simulation2/cppconnector/server_simulator.cpp
  simulation2/cppconnector/connector.h
  simulation2/cppconnector/connector.cpp)
target_include_directories(connsimulator2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(connsimulator2 PRIVATE myclient Boost::boost Boost::thread Boost::filesystem)

# SIMULATOR3
# Wrapper cbs one - attempts to remove python entrypoint
# and replace it with own main() for debugging
# WIP
add_executable(connsimulator3
  simulation3/cppconnector/main.cpp
  simulation3/cppconnector/common.h
  simulation3/cppconnector/ai_simulator.h
  simulation3/cppconnector/ai_simulator.cpp
  simulation3/cppconnector/client_simulator.h
  simulation3/cppconnector/client_simulator.cpp
  simulation3/cppconnector/server_simulator.h
  simulation3/cppconnector/server_simulator.cpp
  simulation3/cppconnector/connector.h
  simulation3/cppconnector/connector.cpp)

target_link_libraries(connsimulator3 PRIVATE myclient Boost::boost Boost::thread Boost::filesystem)
