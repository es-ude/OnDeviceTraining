add_executable(MnistExperiment MnistExperiment.c)
target_link_libraries(MnistExperiment PRIVATE
        DataLoaderApi
        DataLoader
        NPYLoaderApi
        NPYLoader

        Layer

        LinearApi
        Linear

        ReluApi
        Relu

        FlattenApi
        Flatten

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

        CSVHelper

        Common
        StorageApi

)
