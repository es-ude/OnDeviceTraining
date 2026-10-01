add_subdirectory(stack_gather)
add_subdirectory(calculate_grads)
add_subdirectory(training_batch)
add_subdirectory(training_epoch)
add_subdirectory(remat)

add_library(TrainingLoopApi TrainingLoopApi.c)
target_include_directories(TrainingLoopApi PUBLIC include)
target_link_libraries(TrainingLoopApi PRIVATE
        TrainingEpochDefault
        StackGather
        LayerConfigAccess
        ArithmeticType
        TensorApi
        Tensor
        TensorConversion
        Rounding
        Layer
        BatchNorm1d
        LossFunction
        Optimizer
        LrScheduler
        BsScheduler
        Common
        DataLoader
        DataLoaderApi
        InferenceApi
        StorageApi
)
