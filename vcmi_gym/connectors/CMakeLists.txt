cmake_minimum_required(VERSION 3.16.0)
project(connector LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

# https://stackoverflow.com/a/69527824
set(Python_VIRTUALENV FIRST)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(VCMI_DIR "${CMAKE_SOURCE_DIR}/../../vcmi")

# find_package(pybind11 REQUIRED)
add_subdirectory(pybind11)

if(APPLE)
    find_package(Boost REQUIRED COMPONENTS core format)
else()
    find_package(Boost REQUIRED system)
endif()


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_BUILD 1)
    set(VCMI_BIN_PATH "${VCMI_DIR}/build/bin")
else()
    set(VCMI_BIN_PATH "${VCMI_DIR}/rel/bin")
endif()

find_library(LIBMLCLIENT NAMES mlclient PATHS "${VCMI_BIN_PATH}" NO_DEFAULT_PATH)

if(LIBMLCLIENT)
    message(STATUS "Found libmlclient: ${LIBMLCLIENT}")
else()
    message(FATAL_ERROR "libmlclient not found (searched in \"${VCMI_BIN_PATH}\")")
endif()

include_directories(SYSTEM ${Boost_INCLUDE_DIR})

#
# V14
#
set(CONNECTOR_V14_FILES v14/common.h v14/common.cpp v14/connector.cpp v14/threadconnector.h v14/threadconnector.cpp)
set(EXPORTER_V14_FILES v14/exporter.h v14/exporter.cpp)
pybind11_add_module(connector_v14 ${CONNECTOR_V14_FILES})
pybind11_add_module(exporter_v14 ${EXPORTER_V14_FILES})
target_include_directories(connector_v14 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v14 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v14 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v14 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v14 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v14 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v14 PROPERTY SUFFIX)
set_property(TARGET connector_v14 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v14 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v14 PRIVATE DEBUG_BUILD)
endif()

#
# V15
#
set(CONNECTOR_V15_FILES v15/common.h v15/connector.cpp v15/threadconnector.h v15/threadconnector.cpp)
set(EXPORTER_V15_FILES v15/exporter.h v15/exporter.cpp v15/names.h v15/names.cpp)
pybind11_add_module(connector_v15 ${CONNECTOR_V15_FILES})
pybind11_add_module(exporter_v15 ${EXPORTER_V15_FILES})
target_include_directories(connector_v15 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v15 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v15 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v15 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v15 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v15 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v15 PROPERTY SUFFIX)
set_property(TARGET connector_v15 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v15 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v15 PRIVATE DEBUG_BUILD)
endif()
