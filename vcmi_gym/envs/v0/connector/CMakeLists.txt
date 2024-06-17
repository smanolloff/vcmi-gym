cmake_minimum_required(VERSION 3.16.0)
project(connector)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

# https://stackoverflow.com/a/69527824
set(Python_VIRTUALENV FIRST)
set(CMAKE_CXX_STANDARD 17)

link_directories(${CMAKE_SOURCE_DIR}/lib)
find_package(pybind11 REQUIRED)

pybind11_add_module(connector_v1 v1/conncommon.h v1/connector.h v1/connector.cpp)
pybind11_add_module(connexport_v1 v1/connexport.cpp)
target_include_directories(connector_v1 PUBLIC ${CMAKE_SOURCE_DIR}/../vcmi)
target_include_directories(connexport_v1 PUBLIC ${CMAKE_SOURCE_DIR}/../vcmi)
target_link_libraries(connector_v1 PRIVATE myclient)

# Make filenames predictable
set_property(TARGET connexport_v1 PROPERTY SUFFIX)
set_property(TARGET connector_v1 PROPERTY SUFFIX)
