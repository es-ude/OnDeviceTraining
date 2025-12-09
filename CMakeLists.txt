cmake_minimum_required(VERSION 3.20...3.24)

project(OnDeviceTraining)

if(NOT ODT_PATH)
    set(ODT_PATH ${CMAKE_CURRENT_LIST_DIR})
endif()

include(cmake/external_dependencies.cmake)
add_unity()
add_ctest()


add_subdirectory(src)
add_subdirectory(test/unit)