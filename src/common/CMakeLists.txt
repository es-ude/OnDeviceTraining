add_library(Common__hdrs INTERFACE)
target_include_directories(Common__hdrs INTERFACE
        include
)
add_library(Common INTERFACE)
target_link_libraries(Common INTERFACE Common__hdrs)

# DEBUG_MODE_ERROR is the default — every ODT consumer gets PRINT_ERROR output
# unless explicitly silenced (no opt-out preset today; revisit when MCU presets land).
target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_ERROR=true)

if(DEBUG_MODE_INFO)
    target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_INFO=true)
endif()

if(DEBUG_MODE_DEBUG)
    target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_DEBUG=true)
endif()

# ODT_SEED_GUARD: guard the SYM_INT32 seed-rescale casts (Matmul/Conv bias seed,
# LayerNorm affine seed) against the data-dependent float->int32 overflow (#189).
# Default ON (debug/CI), mirroring DEBUG_MODE_ERROR; a future MCU/release preset
# disables it with -DODT_SEED_GUARD=OFF (UBSan #204 still covers occurrences).
option(ODT_SEED_GUARD "Guard SYM_INT32 seed-rescale casts against int32 overflow" ON)
if(ODT_SEED_GUARD)
    target_compile_definitions(Common__hdrs INTERFACE ODT_SEED_GUARD=true)
endif()

# Phase hook for external profilers (#419 precondition). A compiled library,
# unlike the header-only Common target above, because the hook slot is state.
add_library(OdtHook OdtHook.c)
target_include_directories(OdtHook PUBLIC include)
target_link_libraries(OdtHook PRIVATE Common)
