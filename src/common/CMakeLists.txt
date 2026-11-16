add_library(Common__hdrs INTERFACE)
target_include_directories(Common__hdrs INTERFACE
        include
)
add_library(Common INTERFACE)
target_link_libraries(Common INTERFACE Common__hdrs)

# DEBUG_MODE_ERROR is the default — every ODT consumer gets PRINT_ERROR output
# unless explicitly silenced (no opt-out preset today; revisit when MCU presets land).
target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_ERROR=true)

# Lives here, not in the root, so consumers that only add_subdirectory(src) get it too.
if(NOT HOST)
    target_compile_definitions(Common__hdrs INTERFACE BUILDING_FOR_REMOTE)
endif()

if(DEBUG_MODE_INFO)
    target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_INFO=true)
endif()

if(DEBUG_MODE_DEBUG)
    target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE_DEBUG=true)
endif()

# Compiled separately because the process-wide hook owns mutable state.
add_library(OdtHook OdtHook.c)
target_include_directories(OdtHook PUBLIC include)
target_link_libraries(OdtHook PRIVATE Common)
