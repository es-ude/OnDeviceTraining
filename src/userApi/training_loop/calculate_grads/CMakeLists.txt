add_library(CalculateGradsSequential CalculateGradsSequential.c)
target_include_directories(CalculateGradsSequential PUBLIC include)
target_link_libraries(CalculateGradsSequential PRIVATE
        RematScheduler
        RematCheck
        AddLayer
        TrainingLoopApi
        LayerExec
        RNG
        TensorApi
        Tensor
        Layer
        Dropout
        BatchNorm1d
        LayerConfigAccess
        Optimizer
        LossFunction
        InferenceApi
        DataLoader
        Common
        StorageApi
        OdtHook
        OpTable
)
