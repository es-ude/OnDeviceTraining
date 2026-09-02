add_library(DTypes
        DTypes.c
)
target_include_directories(DTypes PUBLIC include)
target_link_libraries(DTypes PRIVATE
        Rounding
)

add_library(DataStorage DataStorage.c)

target_include_directories(DataStorage PUBLIC include)
target_link_libraries(DataStorage PRIVATE
        Tensor
)

add_library(Im2Col
        Im2Col.c
)
target_include_directories(Im2Col PUBLIC include)
target_link_libraries(Im2Col PRIVATE
        Common
        Tensor
        SlidingWindow1d
)

add_library(Quantization
        Quantization.c
)
target_include_directories(Quantization PUBLIC include)
target_link_libraries(Quantization PRIVATE
        Common
        Rounding
        m
)

add_library(TensorConversion
        TensorConversion.c
)
target_include_directories(TensorConversion PUBLIC include)
target_link_libraries(TensorConversion PRIVATE
        Tensor
        Quantization
        Rounding
        Common
)

add_library(Tensor
        Tensor.c
)
target_include_directories(Tensor PUBLIC include)
target_link_libraries(Tensor PRIVATE
        DTypes
        DataStorage
        MinMax
        m
        Rounding
        Quantization
        Common
)

