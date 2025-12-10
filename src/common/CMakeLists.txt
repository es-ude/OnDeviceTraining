## provides helper macros to improve
## - print output

add_library(Common__hdrs INTERFACE)
target_include_directories(Common__hdrs INTERFACE
        include
)
add_library(Common INTERFACE)
target_link_libraries(Common INTERFACE Common__hdrs)
if(DEBUG_MODE)
  target_compile_definitions(Common__hdrs INTERFACE DEBUG_MODE=true)
endif()
