cmake_minimum_required(VERSION 3.16.0)
project(connector)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

# https://stackoverflow.com/a/69527824
set(Python_VIRTUALENV FIRST)
set(CMAKE_CXX_STANDARD 17)

link_directories(${CMAKE_SOURCE_DIR}/lib)
find_package(pybind11 REQUIRED)

pybind11_add_module(connector conncommon.h connector.h connector.cpp)
pybind11_add_module(connexport connexport.cpp)
target_include_directories(connector PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(connexport PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(connector PRIVATE myclient)

# Make filenames predictable
set_property(TARGET connexport PROPERTY SUFFIX)
set_property(TARGET connector PROPERTY SUFFIX)
