add_library(CalculateGradsSequential CalculateGradsSequential.c)
target_include_directories(CalculateGradsSequential PUBLIC include)
# Strict W_dead (a dead backward input gets a header without bytes on every
# plan): test builds only, like the poison in
# RematPlan. The presets set the cache entry before remat/ declares the option.
target_compile_definitions(CalculateGradsSequential
        PRIVATE $<$<BOOL:${ODT_REMAT_VERIFY}>:ODT_REMAT_VERIFY>)
target_link_libraries(CalculateGradsSequential PRIVATE
        RematScheduler
        RematCheck
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
        LayerConfigAccess
        Optimizer
        LossFunction
        InferenceApi
        DataLoader
        Common
        StorageApi
        OdtHook
)
