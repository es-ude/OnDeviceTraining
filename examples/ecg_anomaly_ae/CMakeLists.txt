add_executable(train_c_ecg_anomaly_ae train_c.c)

target_link_libraries(train_c_ecg_anomaly_ae PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        Conv1dApi
        Conv1d

        Conv1dTransposedApi
        Conv1dTransposed

        ReluApi
        Relu

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
        MSE

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
