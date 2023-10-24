cmake_minimum_required(VERSION 3.16.0)
project(connector)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

# https://stackoverflow.com/a/69527824
set(Python_VIRTUALENV FIRST)

message(STATUS "--------------------")
message(STATUS "CMAKE_VERSION: " ${CMAKE_VERSION})
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_BINARY_DIR: " ${CMAKE_BINARY_DIR})
message(STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})
# message(STATUS "PROJECT_BINARY_DIR: " ${PROJECT_BINARY_DIR})
# message(STATUS "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_COMMAND: " ${CMAKE_COMMAND})
message(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})
# message(STATUS "CMAKE_CURRENT_LIST_FILE and LINE: ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE}")
# message(STATUS "CMAKE_INCLUDE_PATH: " ${CMAKE_INCLUDE_PATH})
# message(STATUS "CMAKE_LIBRARY_PATH: " ${CMAKE_LIBRARY_PATH})
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
# message(STATUS "CMAKE_COMPILER_IS_GNUCC: " ${CMAKE_COMPILER_IS_GNUCC})
# message(STATUS "CMAKE_COMPILER_IS_GNUCXX: " ${CMAKE_COMPILER_IS_GNUCXX})
# message(STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS})
# message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_SYSTEM: " ${CMAKE_SYSTEM})
message(STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME})
message(STATUS "CMAKE_SYSTEM_VERSION: " ${CMAKE_SYSTEM_VERSION})
message(STATUS "CMAKE_SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "BIN_DIR: " ${BIN_DIR})
message(STATUS "--------------------")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-aligned-allocation")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmismatched-tags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")   # low chance of valid reports, a lot of emitted warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")             # large number of false-positives, disabled
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")            # large number of noise, low chance of any significant issues
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")       # low chance of any significant issues
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-varargs")            # emitted in fuzzylite headers, disabled
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fstandalone-debug")
set(SYSTEM_LIBS ${SYSTEM_LIBS} ${CMAKE_DL_LIBS})

############################################
#        Finding packages                  #
############################################

# Seems python3.11 is not supported by pybind (as of Oct 2023):
# https://github.com/pybind/pybind11/discussions/4333
find_package(Python3 3.10 REQUIRED EXACT COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_library(MYCLIENTLIB myclient PATHS ${CMAKE_SOURCE_DIR}/lib)

add_library(connector SHARED connector.cpp connector.h)

# target_link_libraries(connector PRIVATE pybind11::pybind11)
# target_link_libraries(connector PRIVATE pybind11::headers pybind11::pybind11 pybind11::embed pybind11::module pybind11::python_link_helper pybind11::windows_extras pybind11::lto pybind11::thin_lto pybind11::opt_size pybind11::python2_no_register)
target_link_libraries(connector PRIVATE pybind11::embed)

target_include_directories(connector PUBLIC ${CMAKE_SOURCE_DIR}/include)

install(TARGETS connector DESTINATION ${CMAKE_SOURCE_DIR}/build/bin)
