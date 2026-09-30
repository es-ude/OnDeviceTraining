# Remat (#4): the shared, placement-free buffer table and static plan
# (spec §2.1). Links: §2.1's list plus Rounding and ArithmeticType, whose
# headers Tensor.h -> Quantization.h and LayerConfigAccess.h include.

# Test builds poison wire bytes at bind and release (spec §7.1). ON in the
# unit_test, unit_test_debug and unit_test_asan presets (unit_test_ubsan
# inherits debug); OFF in firmware and the examples.
option(ODT_REMAT_VERIFY "Poison remat wire bytes at bind and release (test builds)" OFF)

add_library(RematPlan RematWireTable.c RematPlan.c RematPlanPolicy.c)
target_include_directories(RematPlan PUBLIC include)
# PUBLIC, like ODT_MEM_PROFILE on StorageApi: tests that link RematPlan
# directly can #ifdef their poison assertions.
target_compile_definitions(RematPlan PUBLIC $<$<BOOL:${ODT_REMAT_VERIFY}>:ODT_REMAT_VERIFY>)
target_link_libraries(RematPlan PRIVATE
        Tensor
        Quantization
        TensorApi
        Layer
        LayerConfigAccess
        LossFunction
        StorageApi
        Common
        Rounding
        ArithmeticType
)

# Remat (#4): the swappable scheduler interface and its rows (spec §2.1).
# It holds the ARENA and HEAP rows. Links: §2.1's list plus Quantization and
# Rounding, whose headers Tensor.h -> Quantization.h -> Rounding.h include
# (the PR1a Assumption 27 reasoning).
add_library(RematScheduler RematScheduler.c RematArena.c RematHeap.c)
target_include_directories(RematScheduler PUBLIC include)
target_link_libraries(RematScheduler PRIVATE
        RematPlan
        StorageApi
        Layer
        LossFunction
        Tensor
        Quantization
        Rounding
        Common
)

# Remat (#4): the validating interpreter's checker (spec §2.1, §7). Rows never
# link it, and it links no RNG (D44). Links: §2.1's list plus the owners of
# the headers its includes pull in (LossFunction.h; Tensor.h -> Quantization.h
# -> Rounding.h; LayerConfigAccess.h -> ArithmeticType.h).
add_library(RematCheck RematCheck.c)
target_include_directories(RematCheck PUBLIC include)
target_link_libraries(RematCheck PRIVATE
        RematPlan
        RematScheduler
        Layer
        LayerConfigAccess
        LossFunction
        Tensor
        Quantization
        Rounding
        ArithmeticType
        Common
)
