# Every ODT build option, declared once. Included from the root CMakeLists.txt
# and from src/CMakeLists.txt, so consumers that only add_subdirectory(src)
# see the same options and defaults. The user-facing table is in README.md
# ("Build options"); keep both in sync.
include_guard(GLOBAL)

option(HOST "Build for a host PC; OFF defines BUILDING_FOR_REMOTE (MCU target)" ON)
option(BUILD_EXAMPLES "Build the C-side training programs under examples/" OFF)
option(ODT_BUILD_TESTS "Build unit tests (Unity + ctest registration)" ON)
option(ODT_BUILD_DIFFERENTIAL
        "Build the differential numerics driver under test/differential (not a ctest)" OFF)

option(DEBUG_MODE_INFO "Enable PRINT_INFO log output (errors are always printed)" OFF)
option(DEBUG_MODE_DEBUG "Enable PRINT_DEBUG log output" OFF)

option(ODT_MEM_PROFILE "Track heap usage of reserveMemory/freeReservedMemory (peak RAM profiling)" OFF)
option(ODT_TRACK_INSTRUCTIONS "Count arithmetic instructions in the kernels (op-count profiling)" OFF)

option(ODT_REMAT_VERIFY "Poison remat wire bytes at bind and release (test builds)" OFF)
option(ODT_REMAT_CHECK "Validate every remat step before the driver runs it; OFF saves flash" ON)
option(ODT_SCALE_STRICT "Assert every stored SYM/ASYM scale is canonical dyadic" OFF)
