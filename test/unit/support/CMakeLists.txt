# Header-only test-support utilities shared across unit-test modules.
#
# An INTERFACE library carries no compiled sources: linking it only propagates
# its INTERFACE_INCLUDE_DIRECTORIES to the consumer. A test target that lists
# `odt_test_death` in MORE_LIBS can therefore `#include "DeathTest.h"` without
# any per-target include plumbing.
add_library(odt_test_death INTERFACE)
target_include_directories(odt_test_death INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
