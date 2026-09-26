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
# PR1b holds the ARENA row. Links: §2.1's list plus Quantization and
# Rounding, whose headers Tensor.h -> Quantization.h -> Rounding.h include
# (the PR1a Assumption 27 reasoning).
add_library(RematScheduler RematScheduler.c RematArena.c)
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
