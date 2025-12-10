cmake_minimum_required(VERSION 3.20...3.24)

include(cmake/external_dependencies.cmake)
include(cmake/helpers.cmake)


project(OnDeviceTraining)

odt_is_top_level_project(ODT_TOP_LEVEL_PROJECT)

add_subdirectory(src)

if(ODT_TOP_LEVEL_PROJECT)
    if(NOT ODT_PATH)
        set(ODT_PATH ${CMAKE_CURRENT_LIST_DIR})
    endif()

    if(DEBUG_OUTPUT)
        add_definitions(-DDEBUG_MODE)
    endif()

    add_unity()
    add_ctest()

    add_subdirectory(test/unit)
endif()