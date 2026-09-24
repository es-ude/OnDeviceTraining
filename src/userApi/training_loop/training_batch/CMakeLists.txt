add_library(TrainingBatchDefault TrainingBatchDefault.c)
target_include_directories(TrainingBatchDefault PUBLIC include)
target_link_libraries(TrainingBatchDefault PRIVATE
        TrainingLoopApi
        TensorApi
        Tensor
        Rounding
        Layer
        Optimizer
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        Common
        StorageApi
        # #152 PR3b: the m > 1 FLOAT32 gate (layerIsFloat32Only).
        LayerConfigAccess
        ArithmeticType
)
