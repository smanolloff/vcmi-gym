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
# V13
#
set(CONNECTOR_V13_FILES v13/common.h v13/common.cpp v13/connector.cpp v13/threadconnector.h v13/threadconnector.cpp)
set(EXPORTER_V13_FILES v13/exporter.h v13/exporter.cpp)
pybind11_add_module(connector_v13 ${CONNECTOR_V13_FILES})
pybind11_add_module(exporter_v13 ${EXPORTER_V13_FILES})
target_include_directories(connector_v13 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v13 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_include_directories(exporter_v13 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v13 PUBLIC "${VCMI_DIR}/AI/MMAI")
target_link_libraries(connector_v13 PRIVATE "${Boost_LIBRARIES}")
target_link_libraries(connector_v13 PUBLIC ${LIBMLCLIENT})
set_property(TARGET exporter_v13 PROPERTY SUFFIX)
set_property(TARGET connector_v13 PROPERTY SUFFIX)
if(DEBUG_BUILD)
    target_compile_definitions(connector_v13 PRIVATE DEBUG_BUILD)
    target_compile_definitions(exporter_v13 PRIVATE DEBUG_BUILD)
endif()

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
