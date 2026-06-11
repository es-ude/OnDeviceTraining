add_library(CalculateGradsSequential CalculateGradsSequential.c)
target_include_directories(CalculateGradsSequential PUBLIC include)
target_include_directories(CalculateGradsSequential PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(CalculateGradsSequential PRIVATE
        TrainingLoopApi
        TensorApi
        Tensor
        Rounding
        Layer
        Conv1d
        Conv1dTransposed
        MaxPool1d
        AvgPool1d
        AdaptiveAvgPool1d
        LayerNorm
        QuantizationLayer
        Optimizer
        LossFunction
        InferenceApi
        DataLoader
        Common
        StorageApi
)
