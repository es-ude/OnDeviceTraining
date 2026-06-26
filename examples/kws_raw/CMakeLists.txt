add_executable(train_c_kws_raw train_c.c)

target_link_libraries(train_c_kws_raw PRIVATE
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

        AdaptivePool1dApi
        AdaptiveAvgPool1d

        LayerNormApi
        LayerNorm

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
)
