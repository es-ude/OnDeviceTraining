add_library(TrainingEpochDefault TrainingEpochDefault.c)
target_include_directories(TrainingEpochDefault PUBLIC include)
target_link_libraries(TrainingEpochDefault PRIVATE
        TrainingLoopApi
        TrainingBatchDefault
        TensorApi
        Tensor
        Rounding
        Layer
        Optimizer
        OptimizerApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        Common
)
