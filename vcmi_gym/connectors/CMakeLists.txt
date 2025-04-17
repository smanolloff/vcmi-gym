cmake_minimum_required(VERSION 3.16.0)
project(connector LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

# https://stackoverflow.com/a/69527824
set(Python_VIRTUALENV FIRST)
set(CMAKE_CXX_STANDARD 17)
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

#
# V10
#
set(CONNECTOR_V10_FILES v10/common.h v10/common.cpp v10/connector.cpp v10/threadconnector.h v10/threadconnector.cpp v10/procconnector.h v10/procconnector.cpp)
set(EXPORTER_V10_FILES v10/exporter.h v10/exporter.cpp)
pybind11_add_module(connector_v10 ${CONNECTOR_V10_FILES})
pybind11_add_module(exporter_v10 ${EXPORTER_V10_FILES})
target_include_directories(connector_v10 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v10 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v10 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v10 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v10 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v10 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v10 PROPERTY SUFFIX)
set_property(TARGET connector_v10 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v10 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v10 PRIVATE DEBUG_BUILD)
endif()

#
# V11
#
set(CONNECTOR_V11_FILES v11/common.h v11/common.cpp v11/connector.cpp v11/threadconnector.h v11/threadconnector.cpp v11/procconnector.h v11/procconnector.cpp)
set(EXPORTER_V11_FILES v11/exporter.h v11/exporter.cpp)
pybind11_add_module(connector_v11 ${CONNECTOR_V11_FILES})
pybind11_add_module(exporter_v11 ${EXPORTER_V11_FILES})
target_include_directories(connector_v11 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v11 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v11 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v11 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v11 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v11 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v11 PROPERTY SUFFIX)
set_property(TARGET connector_v11 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v11 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v11 PRIVATE DEBUG_BUILD)
endif()
