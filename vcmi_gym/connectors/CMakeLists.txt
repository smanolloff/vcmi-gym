cmake_minimum_required(VERSION 3.16.0)
project(connector LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

# https://stackoverflow.com/a/69527824
set(Python_VIRTUALENV FIRST)
set(CMAKE_CXX_STANDARD 17)
set(VCMI_DIR "${CMAKE_SOURCE_DIR}/../../vcmi")

find_package(pybind11 REQUIRED)

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
# V3
#
set(CONNECTOR_V3_FILES v3/connector.h v3/connector.cpp v3/common.h)
set(EXPORTER_V3_FILES v3/exporter.h v3/exporter.cpp)
pybind11_add_module(connector_v3 ${CONNECTOR_V3_FILES})
pybind11_add_module(exporter_v3 ${EXPORTER_V3_FILES})
target_include_directories(connector_v3 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v3 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v3 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v3 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v3 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v3 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v3 PROPERTY SUFFIX)
set_property(TARGET connector_v3 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v3 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v3 PRIVATE DEBUG_BUILD)
endif()

#
# V7
#
set(CONNECTOR_V7_FILES v7/common.h v7/common.cpp v7/connector.cpp v7/threadconnector.h v7/threadconnector.cpp v7/procconnector.h v7/procconnector.cpp)
set(EXPORTER_V7_FILES v7/exporter.h v7/exporter.cpp)
pybind11_add_module(connector_v7 ${CONNECTOR_V7_FILES})
pybind11_add_module(exporter_v7 ${EXPORTER_V7_FILES})
target_include_directories(connector_v7 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v7 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v7 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v7 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v7 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v7 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v7 PROPERTY SUFFIX)
set_property(TARGET connector_v7 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v7 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v7 PRIVATE DEBUG_BUILD)
endif()

#
# V8
#
set(CONNECTOR_V8_FILES v8/common.h v8/common.cpp v8/connector.cpp v8/threadconnector.h v8/threadconnector.cpp v8/procconnector.h v8/procconnector.cpp)
set(EXPORTER_V8_FILES v8/exporter.h v8/exporter.cpp)
pybind11_add_module(connector_v8 ${CONNECTOR_V8_FILES})
pybind11_add_module(exporter_v8 ${EXPORTER_V8_FILES})
target_include_directories(connector_v8 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v8 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v8 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v8 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v8 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v8 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v8 PROPERTY SUFFIX)
set_property(TARGET connector_v8 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v8 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v8 PRIVATE DEBUG_BUILD)
endif()

#
# V9
#
set(CONNECTOR_V9_FILES v9/common.h v9/common.cpp v9/connector.cpp v9/threadconnector.h v9/threadconnector.cpp v9/procconnector.h v9/procconnector.cpp)
set(EXPORTER_V9_FILES v9/exporter.h v9/exporter.cpp)
pybind11_add_module(connector_v9 ${CONNECTOR_V9_FILES})
pybind11_add_module(exporter_v9 ${EXPORTER_V9_FILES})
target_include_directories(connector_v9 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v9 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v9 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v9 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v9 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v9 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v9 PROPERTY SUFFIX)
set_property(TARGET connector_v9 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v9 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v9 PRIVATE DEBUG_BUILD)
endif()
