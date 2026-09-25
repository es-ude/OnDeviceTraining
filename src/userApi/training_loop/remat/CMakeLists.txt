# Remat (#4): the shared, placement-free buffer table and static plan
# (spec §2.1). Links: §2.1's list plus Rounding and ArithmeticType, whose
# headers Tensor.h -> Quantization.h and LayerConfigAccess.h include.
add_library(RematPlan RematPlan.c)
target_include_directories(RematPlan PUBLIC include)
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
