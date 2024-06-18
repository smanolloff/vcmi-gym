cmake_minimum_required(VERSION 3.16.0)
project(connector LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake_modules)

# https://stackoverflow.com/a/69527824
set(Python_VIRTUALENV FIRST)
set(CMAKE_CXX_STANDARD 17)
set(VCMI_DIR "${CMAKE_SOURCE_DIR}/../../vcmi")

find_package(pybind11 REQUIRED)

if(CONFIG STREQUAL "Debug")
    set(DEBUG_BUILD 1)
    set(GYMCLIENT_PATH "${VCMI_DIR}/build/bin")
else()
    set(GYMCLIENT_PATH "${VCMI_DIR}/rel/bin")
endif()

find_library(LIBGYMCLIENT NAMES gymclient PATHS "${GYMCLIENT_PATH}" NO_DEFAULT_PATH)

if(NOT LIBGYMCLIENT)
    message(FATAL_ERROR "libgymclient not found (searched in \"${GYMCLIENT_PATH}\")")
endif()

# Using variables for the file lists make it easier to define new targets
# which reuse most of the old code and only add a few modifications
set(CONNECTOR_V1_FILES v1/connector.h v1/connector.cpp v1/common.h)
set(CONNECTOR_V2_FILES v2/connector.h v2/connector.cpp ${CONNECTOR_V1_FILES})
set(EXPORTER_V1_FILES v1/exporter.h v1/exporter.cpp)
set(EXPORTER_V2_FILES v2/exporter.h v2/exporter.cpp ${EXPORTER_V1_FILES})

pybind11_add_module(connector_v1 ${CONNECTOR_V1_FILES})
pybind11_add_module(connector_v2 ${CONNECTOR_V2_FILES})
pybind11_add_module(exporter_v1 ${EXPORTER_V1_FILES})
pybind11_add_module(exporter_v2 ${EXPORTER_V2_FILES})
target_include_directories(connector_v1 PUBLIC "${VCMI_DIR}")
target_include_directories(connector_v2 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v1 PUBLIC "${VCMI_DIR}")
target_include_directories(exporter_v2 PUBLIC "${VCMI_DIR}")
target_link_libraries(connector_v1 PRIVATE ${LIBGYMCLIENT})
target_link_libraries(connector_v2 PRIVATE ${LIBGYMCLIENT})

if(DEBUG_BUILD)
    target_compile_definitions(connector_v1 PRIVATE $<$<CONFIG:Debug>:DEBUG_BUILD>)
    target_compile_definitions(connector_v2 PRIVATE $<$<CONFIG:Debug>:DEBUG_BUILD>)
    target_compile_definitions(exporter_v1 PRIVATE $<$<CONFIG:Debug>:DEBUG_BUILD>)
    target_compile_definitions(exporter_v2 PRIVATE $<$<CONFIG:Debug>:DEBUG_BUILD>)
endif()

# Make filenames predictable
set_property(TARGET exporter_v1 PROPERTY SUFFIX)
set_property(TARGET exporter_v2 PROPERTY SUFFIX)
set_property(TARGET connector_v1 PROPERTY SUFFIX)
set_property(TARGET connector_v2 PROPERTY SUFFIX)
