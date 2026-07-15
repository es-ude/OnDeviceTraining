add_library(PpcaReplayApi PpcaReplayApi.c ReplayDataLoader.c ExemplarBuffer.c)
target_include_directories(PpcaReplayApi PUBLIC include)
target_link_libraries(PpcaReplayApi PRIVATE
        Common
        PpcaReplay
        Tensor
        TensorApi
        Quantization
        QuantizationApi
        ArithmeticType
        Rounding
        RNG
        DataLoader
        DataLoaderApi
        StorageApi
        ExecuteOp
)
