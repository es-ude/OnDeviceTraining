add_library(Optimizer Optimizer.c)
target_include_directories(Optimizer PUBLIC include)
target_link_libraries(Optimizer PRIVATE
        Quantization
        Rounding
        Layer
        Conv1d
        Conv1dTransposed
        Linear
        Tensor
        Common
)

add_library(Sgd Sgd.c)
target_include_directories(Sgd PUBLIC include)
target_link_libraries(Sgd PRIVATE
        DTypes
        Tensor
        Arithmetic
        ArithmeticType
        ExecuteOp
        Layer
        Common
)