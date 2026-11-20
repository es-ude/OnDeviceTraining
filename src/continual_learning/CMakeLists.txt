add_library(PpcaReplay PpcaReplay.c)
target_include_directories(PpcaReplay PUBLIC include)
target_link_libraries(PpcaReplay PRIVATE
        m
        Common
        Tensor
        DTypes
        Quantization
        Rounding
        TensorConversion
        ExecuteOp
        ArithmeticType
        Matmul
        Reduce
        Axpby
        RowBroadcast
        JacobiEig
        Sub
        Add
        Mul
        MinMax
        Distributions
        RNG
)

add_library(PpcaReplaySerialize PpcaReplaySerialize.c)
target_include_directories(PpcaReplaySerialize PUBLIC include)
target_link_libraries(PpcaReplaySerialize PRIVATE
        LossType
        Common
        PpcaReplay
        SerialWire
        Serialize
        Deserialize
        Tensor
        Quantization
        Rounding
        Layer
        RNG
        ArithmeticType
)
