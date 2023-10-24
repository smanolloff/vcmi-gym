project(proj_b)

message(STATUS ${CMAKE_SOURCE_DIR}/../proj-a)

link_directories(${CMAKE_SOURCE_DIR}/../proj-a/build)
add_executable(projb projb.cpp projb.h)

target_include_directories(projb PRIVATE ${CMAKE_SOURCE_DIR}/../proj-a)
target_link_libraries(projb PRIVATE proja)

