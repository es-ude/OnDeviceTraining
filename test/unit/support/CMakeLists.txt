# Header-only test-support utilities shared across unit-test modules.
#
# An INTERFACE library carries no compiled sources: linking it only propagates
# its INTERFACE_INCLUDE_DIRECTORIES to the consumer. A test target that lists
# `odt_test_death` in MORE_LIBS can therefore `#include "DeathTest.h"` without
# any per-target include plumbing.
add_library(odt_test_death INTERFACE)
target_include_directories(odt_test_death INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Borrowed-parameter layer builders (BorrowedLayer.h). Consumers must already
# link the layer/arithmetic/StorageApi targets the header's includes and the
# emitted calls need — every current consumer does, since the builders were
# previously inlined in those test files.
add_library(odt_test_layer_builders INTERFACE)
target_include_directories(odt_test_layer_builders INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Per-element size_t array assert (OdtAssert.h). Header-only, like
# odt_test_death: listing `odt_test_assert` in MORE_LIBS is all a test target
# needs to include it.
add_library(odt_test_assert INTERFACE)
target_include_directories(odt_test_assert INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
