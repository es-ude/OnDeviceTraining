# Shared memory instrumentation consumed by BOTH HAR trainers. HAR-local (keeps
# the topology-specific activation/IO model out of examples_shared) and DRY (one
# translation unit, both binaries link it). Public link deps propagate the header
# include paths (Optimizer/TrainingLoopApi) to consumers.
add_library(har_mem_instrument STATIC mem_instrument.c)
target_include_directories(har_mem_instrument PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(har_mem_instrument PUBLIC
        StorageApi
        MemProfile
        Tensor
        Optimizer
        CalculateGradsSequential
        TrainingLoopApi
        Layer
        MaxPool1d
        LossFunction
        DataLoader
        InferenceApi
        Rounding
        Common
)

add_executable(train_c_har_classifier train_c.c)

target_link_libraries(train_c_har_classifier PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        Conv1dApi
        Conv1d

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

        Pool1dApi
        MaxPool1d
        AvgPool1d

        QuantizationApi
        Quantization

        TensorApi
        Tensor
        Rounding

        TrainingLoopApi
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        Optimizer

        LossFunction
        CrossEntropy

        SoftmaxApi
        Softmax

        Sgd
        SgdApi

        InferenceApi

        StateDictApi
        LayerWeightsApi
        LayerQuant
        LayerCommon
        Distributions

        Common
        StorageApi
        RNG

        examples_shared
        har_mem_instrument
)

add_executable(train_c_har_classifier_adamw train_c_adamw.c)

target_link_libraries(train_c_har_classifier_adamw PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        Conv1dApi
        Conv1d

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

        Pool1dApi
        MaxPool1d
        AvgPool1d

        QuantizationApi
        Quantization

        TensorApi
        Tensor
        Rounding

        TrainingLoopApi
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        Optimizer

        LossFunction
        CrossEntropy

        SoftmaxApi
        Softmax

        Sgd
        AdamW
        AdamWApi

        InferenceApi

        StateDictApi
        LayerWeightsApi
        LayerQuant
        LayerCommon
        Distributions

        Common
        StorageApi
        RNG

        examples_shared
        har_mem_instrument
)

add_executable(train_c_continual train_c_continual.c)

target_link_libraries(train_c_continual PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        Conv1dApi
        Conv1d

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

        Pool1dApi
        MaxPool1d
        AvgPool1d

        QuantizationApi
        Quantization

        TensorApi
        Tensor
        Rounding

        TrainingLoopApi
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        Optimizer

        LossFunction
        CrossEntropy

        SoftmaxApi
        Softmax

        Sgd
        SgdApi

        InferenceApi

        StateDictApi
        LayerWeightsApi
        LayerQuant
        LayerCommon
        Distributions

        Common
        StorageApi
        RNG

        PpcaReplay
        PpcaReplayApi
        PpcaReplaySerialize
        Axpby
        RowBroadcast
        JacobiEig
        ExecuteOp
        TensorConversion
        ArithmeticType

        examples_shared
        har_mem_instrument
)

add_executable(train_c_har_classifier_sym train_c_sym.c)

target_link_libraries(train_c_har_classifier_sym PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        Conv1dApi
        Conv1d

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

        Pool1dApi
        MaxPool1d
        AvgPool1d

        ArithmeticType

        QuantizationApi
        Quantization

        TensorApi
        Tensor
        TensorConversion
        Rounding

        TrainingLoopApi
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        Optimizer
        LrScheduler

        LossFunction
        CrossEntropy

        SoftmaxApi
        Softmax

        Sgd
        SgdApi

        InferenceApi

        StateDictApi
        LayerWeightsApi
        LayerQuant
        LayerCommon
        Distributions

        Common
        StorageApi
        RNG

        examples_shared
        har_mem_instrument
)
