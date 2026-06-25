cmake_minimum_required(VERSION 3.20...3.24)

include(cmake/external_dependencies.cmake)
include(cmake/helpers.cmake)


project(OnDeviceTraining)

odt_is_top_level_project(ODT_TOP_LEVEL_PROJECT)

add_subdirectory(src)

set(DEBUG_LEVEL "0" CACHE STRING "Debug verbosity: 0=off, 1=error, 2=warn, 3=info")
option(HOST "Build for host machine" ON)
option(BUILD_EXAMPLES "Build the C-side training programs under examples/" OFF)
if(NOT HOST)
    add_compile_definitions(BUILDING_FOR_REMOTE)
endif()


if(ODT_TOP_LEVEL_PROJECT)
    if(NOT ODT_PATH)
        set(ODT_PATH ${CMAKE_CURRENT_LIST_DIR})
    endif()

    # Apple's ld-prime (multi-pass) warns "ignoring duplicate libraries" on the
    # repeated static archives CMake emits so single-pass GNU ld can resolve the
    # framework's circular static-lib deps (see #234). The repeats are required
    # on the Linux CI linker and benign on macOS — silence the noise where the
    # linker supports the flag. Feature-detected, so arm-gcc / GNU ld (which do
    # not know the flag, and where the repeats are needed) are left untouched.
    include(CheckLinkerFlag)
    check_linker_flag(C "-Wl,-no_warn_duplicate_libraries" ODT_LD_HAS_NO_WARN_DUP_LIBS)
    if(ODT_LD_HAS_NO_WARN_DUP_LIBS)
        add_link_options("-Wl,-no_warn_duplicate_libraries")
    endif()

    add_unity()
    add_ctest()

    add_subdirectory(test/unit)
    if(BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif()
endif()