cmake_minimum_required(VERSION 3.20...3.24)

include(cmake/external_dependencies.cmake)
include(cmake/helpers.cmake)


project(OnDeviceTraining)

odt_is_top_level_project(ODT_TOP_LEVEL_PROJECT)

add_subdirectory(src)

set(DEBUG_LEVEL "0" CACHE STRING "Debug verbosity: 0=off, 1=error, 2=warn, 3=info")

if(ODT_TOP_LEVEL_PROJECT)
    if(NOT ODT_PATH)
        set(ODT_PATH ${CMAKE_CURRENT_LIST_DIR})
    endif()

    add_unity()
    add_ctest()

    add_subdirectory(test/unit)
endif()