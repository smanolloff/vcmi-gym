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

pybind11_add_module(connector conncommon.h connector.h connector.cpp)
pybind11_add_module(connexport connexport.cpp)
target_include_directories(connector PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(connexport PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(connector PRIVATE myclient)

add_library(loader SHARED conncommon.h loader.h loader.cpp)
target_include_directories(loader PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(loader PRIVATE pybind11::embed)

add_executable(loader_main conncommon.h loader_main.cpp)
target_include_directories(loader_main PUBLIC ${CMAKE_SOURCE_DIR}/include)
# add_dependencies(loader_main loader)
# target_link_libraries(loader_main PRIVATE loader)


# Make filenames predictable
set_property(TARGET connexport PROPERTY SUFFIX)
set_property(TARGET connector PROPERTY SUFFIX)
