add_library(Conv1d Conv1d.c)
target_include_directories(Conv1d PUBLIC include)
target_link_libraries(Conv1d PRIVATE
        DTypes
        Tensor
        Arithmetic
        Common
        Kernel
        SlidingWindow1d
        Conv1dKernel
        ConvTranspose1dKernel
)

add_library(Conv1dTransposed Conv1dTransposed.c)
target_include_directories(Conv1dTransposed PUBLIC include)
target_link_libraries(Conv1dTransposed PRIVATE
        DTypes
        Tensor
        Arithmetic
        Common
        Kernel
        SlidingWindow1d
        Conv1dKernel
        ConvTranspose1dKernel
)

add_library(MaxPool1d MaxPool1d.c)
target_include_directories(MaxPool1d PUBLIC include)
target_link_libraries(MaxPool1d PRIVATE
        DTypes
        Tensor
        Arithmetic
        Common
        Kernel
        SlidingWindow1d
)

add_library(AvgPool1d AvgPool1d.c)
target_include_directories(AvgPool1d PUBLIC include)
target_link_libraries(AvgPool1d PRIVATE
        DTypes
        Tensor
        Arithmetic
        Common
        Kernel
        SlidingWindow1d
)

add_library(AdaptiveAvgPool1d AdaptiveAvgPool1d.c)
target_include_directories(AdaptiveAvgPool1d PUBLIC include)
target_link_libraries(AdaptiveAvgPool1d PRIVATE
        DTypes
        Tensor
        Common
        AdaptiveWindow1d
)

add_library(Dropout Dropout.c)
target_include_directories(Dropout PUBLIC include)
target_link_libraries(Dropout PRIVATE
        DTypes
        Tensor
        Layer
        Common
        Bernoulli
        Rounding
)

add_library(LayerNorm LayerNorm.c)
target_include_directories(LayerNorm PUBLIC include)
target_link_libraries(LayerNorm PRIVATE
        DTypes
        Tensor
        TensorConversion
        Quantization
        Add
        Arithmetic
        Rounding
        Layer
        Common
)

add_library(Layer Layer.c)
target_include_directories(Layer PUBLIC include)
target_link_libraries(Layer PRIVATE
        DTypes
        Tensor
        Arithmetic
        Linear
        Relu
        Conv1d
        Conv1dTransposed
        MaxPool1d
        AvgPool1d
        AdaptiveAvgPool1d
        Softmax
        Flatten
        QuantizationLayer
        Dropout
        LayerNorm
)

add_library(Linear Linear.c)
target_include_directories(Linear PUBLIC include)
target_link_libraries(Linear PRIVATE
        DTypes
        Tensor
        Arithmetic
        Quantization
        TensorConversion
        Layer
        Common
)

add_library(QuantizationLayer QuantizationLayer.c)
target_include_directories(QuantizationLayer PUBLIC include)
target_link_libraries(QuantizationLayer PRIVATE
        Common
        Quantization
        Rounding
        Tensor
        TensorConversion
)

add_library(Relu Relu.c)
target_include_directories(Relu PUBLIC include)
target_link_libraries(Relu PRIVATE
        DTypes
        Tensor
        Arithmetic
        Comparison
        Layer
        Common
)

add_library(Softmax Softmax.c)
target_include_directories(Softmax PUBLIC include)
target_link_libraries(Softmax PRIVATE
        DTypes
        Tensor
        Arithmetic
        Layer
        Common
)

add_library(Flatten Flatten.c)
target_include_directories(Flatten PUBLIC include)
target_link_libraries(Flatten PRIVATE
        DTypes
        Tensor
        Arithmetic
        Layer
        Common
)

add_library(CommonLayerLibs INTERFACE)
target_link_libraries(CommonLayerLibs INTERFACE
        Layer
        Linear
        Relu
        Conv1d
        Conv1dTransposed
        MaxPool1d
        AvgPool1d
        AdaptiveAvgPool1d
        Softmax
        Flatten
        QuantizationLayer
        Dropout
        LayerNorm
        Rounding
        Tensor
        Sgd
)

