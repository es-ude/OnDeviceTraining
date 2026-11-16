# Remat (#4): the shared, placement-free buffer table and static plan.
# Rounding and ArithmeticType are linked because Tensor.h -> Quantization.h
# and LayerConfigAccess.h include their headers.

add_library(RematPlan RematWireTable.c RematDerive.c RematPlan.c RematPlanPolicy.c)
target_include_directories(RematPlan PUBLIC include)
# PUBLIC, like ODT_MEM_PROFILE on StorageApi: tests that link RematPlan
# directly can #ifdef their poison assertions.
target_compile_definitions(RematPlan PUBLIC $<$<BOOL:${ODT_REMAT_VERIFY}>:ODT_REMAT_VERIFY>)
target_link_libraries(RematPlan PRIVATE
        Model
        OpTable
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
        LayerConfigAccess
        ArithmeticType
        LossFunction
        Tensor
        Quantization
        Rounding
        Common
)

# Remat (#4): the driver's operand resolver, plus the validator when
# ODT_REMAT_CHECK is ON. Rows never link it, and it links no RNG. Besides its
# direct dependencies it links the owners of the headers its includes pull in
# (LossFunction.h; Tensor.h -> Quantization.h -> Rounding.h;
# LayerConfigAccess.h -> ArithmeticType.h).
add_library(RematCheck RematResolve.c $<$<BOOL:${ODT_REMAT_CHECK}>:RematCheck.c>)
target_include_directories(RematCheck PUBLIC include)
# PUBLIC: RematCheck.h switches on it, so every includer must see the same value.
target_compile_definitions(RematCheck PUBLIC $<$<BOOL:${ODT_REMAT_CHECK}>:ODT_REMAT_CHECK>)
target_link_libraries(RematCheck PRIVATE
        RematPlan
        RematScheduler
        Model
        OpTable
        Layer
        LayerConfigAccess
        LossFunction
        Tensor
        Quantization
        Rounding
        ArithmeticType
        Common
)
