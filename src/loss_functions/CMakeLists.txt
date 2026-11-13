add_library(CrossEntropy
        CrossEntropy.c)
target_include_directories(CrossEntropy PUBLIC include)
target_link_libraries(CrossEntropy PRIVATE
        Tensor
        Quantization
        ArithmeticType
        Rounding
        ExecuteOp
        Common
        CommonLayerLibs
        LayerExec
        RNG
)

# Header-only: the loss-kind enum, for code that must not depend on the loss implementations.
add_library(LossType INTERFACE)
target_include_directories(LossType INTERFACE include)

add_library(LossFunction
        LossFunction.c)
target_include_directories(LossFunction PUBLIC include)
target_link_libraries(LossFunction PUBLIC LossType)
target_link_libraries(LossFunction PRIVATE
        Tensor
        Rounding
        MSE
        CrossEntropy
)


add_library(MSE
        MSE.c
)
target_include_directories(MSE PUBLIC include)
target_link_libraries(MSE PRIVATE
        Tensor
        TensorConversion
        Square
        Sub
        Common
        LayerExec
        RNG
)