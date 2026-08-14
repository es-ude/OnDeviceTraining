add_subdirectory(calculate_grads)
add_subdirectory(training_batch)
add_subdirectory(training_epoch)

add_library(TrainingLoopApi TrainingLoopApi.c)
target_include_directories(TrainingLoopApi PUBLIC include)
target_link_libraries(TrainingLoopApi PRIVATE
        TrainingEpochDefault
        TensorApi
        Tensor
        TensorConversion
        Rounding
        Layer
        LossFunction
        Optimizer
        LrScheduler
        Common
        DataLoader
        DataLoaderApi
        InferenceApi
        StorageApi
)
