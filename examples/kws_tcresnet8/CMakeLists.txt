add_executable(train_c_kws_tcresnet8 train_c.c)

target_link_libraries(train_c_kws_tcresnet8 PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer
        Model
        OpTable
        ModelGraphApi

        AddApi
        AddLayer

        Conv1dApi
        Conv1d

        BatchNorm1dApi
        BatchNorm1d

        DropoutApi
        Dropout

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

        AdaptivePool1dApi
        AdaptiveAvgPool1d

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
        OptimizerApi
        ParameterWalk
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
)
