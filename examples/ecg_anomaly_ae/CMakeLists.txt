add_executable(train_c_ecg_anomaly_ae train_c.c)

target_link_libraries(train_c_ecg_anomaly_ae PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        Conv1dApi
        Conv1d
        Conv1dTransposed

        ReluApi
        Relu

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

        Common
        StorageApi
        RNG

        examples_shared
)
