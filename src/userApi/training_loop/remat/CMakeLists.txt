# Remat (#4): the shared, placement-free buffer table and static plan.
# Rounding and ArithmeticType are linked because Tensor.h -> Quantization.h
# and LayerConfigAccess.h include their headers.

# Test builds poison wire bytes at bind and release. ON in the
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

# Remat (#4): the swappable scheduler interface and its rows (ARENA and
# HEAP). Quantization and Rounding are linked because Tensor.h ->
# Quantization.h -> Rounding.h include their headers.
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

# Remat (#4): the validating interpreter's checker. Rows never link it, and
# it links no RNG. Besides its direct dependencies it links the owners of
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
