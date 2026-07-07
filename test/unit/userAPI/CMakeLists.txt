add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        StorageApi
)


add_elastic_ai_unit_test(LIB_UNDER_TEST MemProfile MORE_LIBS StorageApi)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        LayerQuant
        MORE_LIBS
        ArithmeticType
        Common
        QuantizationApi
        Quantization
        Rounding
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        LayerWeightsApi
        MORE_LIBS
        Common
        LinearApi
        Conv1dApi
        Conv1dTransposedApi
        LayerNormApi
        QuantizationApi
        LayerQuant
        LayerCommon
        Quantization
        Rounding
        Linear
        Conv1d
        Conv1dTransposed
        LayerNorm
        Kernel
        Tensor
        TensorApi
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        StateDictApi
        MORE_LIBS
        Common
        LinearApi
        ReluApi
        LayerNormApi
        LayerQuant
        QuantizationApi
        LayerWeightsApi
        LayerCommon
        Quantization
        Rounding
        Linear
        Relu
        LayerNorm
        Tensor
        TensorApi
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        InferenceApi
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        LinearApi
        LayerWeightsApi
        ReluApi
        LayerNormApi
        LayerNorm
        LayerQuant
        ArithmeticType
        Quantization
        QuantizationApi
        LossFunction
        DataLoader
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        TrainingLoopApi
        MORE_LIBS
        TrainingEpochDefault
        TrainingBatchDefault
        CalculateGradsSequential
        CommonLayerLibs
        TensorApi
        LinearApi
        ArithmeticType
        ReluApi
        SgdApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        StorageApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1dApi
        MORE_LIBS
        ArithmeticType
        Common
        LayerQuant
        LayerCommon
        QuantizationApi
        Quantization
        Rounding
        Conv1d
        Distributions
        RNG
        Kernel
        Tensor
        TensorApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1dTransposedApi
        MORE_LIBS
        ArithmeticType
        Common
        LayerQuant
        LayerCommon
        QuantizationApi
        Quantization
        Rounding
        Conv1dTransposed
        Distributions
        RNG
        Kernel
        Tensor
        TensorApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Pool1dApi
        MORE_LIBS
        ArithmeticType
        Common
        LayerQuant
        QuantizationApi
        Quantization
        Rounding
        MaxPool1d
        AvgPool1d
        Kernel
        Tensor
        TensorApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        AdaptivePool1dApi
        MORE_LIBS
        ArithmeticType
        Common
        LayerQuant
        QuantizationApi
        Quantization
        Rounding
        AdaptiveAvgPool1d
        TensorApi
)

add_executable(UnitTestDropoutIntegration UnitTestDropoutIntegration.c)
target_link_libraries(UnitTestDropoutIntegration PRIVATE
        unity
        TrainingLoopApi
        CalculateGradsSequential
        CommonLayerLibs
        DropoutApi
        Dropout
        Bernoulli
        RNG
        SgdApi
        Sgd
        Optimizer
        StateDictApi
        LayerQuant
        LayerCommon
        LinearApi
        TensorApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        StorageApi
)
__register_target_as_unit_test(UnitTestDropoutIntegration)

add_executable(UnitTestLayerNormIntegration UnitTestLayerNormIntegration.c)
target_link_libraries(UnitTestLayerNormIntegration PRIVATE
        unity
        TrainingLoopApi
        CalculateGradsSequential
        CommonLayerLibs
        LayerNormApi
        LayerNorm
        SgdApi
        Sgd
        Optimizer
        LayerQuant
        LayerCommon
        LinearApi
        TensorApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        StorageApi
)
__register_target_as_unit_test(UnitTestLayerNormIntegration)

add_executable(UnitTestGroupNormIntegration UnitTestGroupNormIntegration.c)
target_link_libraries(UnitTestGroupNormIntegration PRIVATE
        unity
        TrainingLoopApi
        TrainingEpochDefault
        TrainingBatchDefault
        CalculateGradsSequential
        CommonLayerLibs
        GroupNormApi
        GroupNorm
        Reduce
        Conv1dApi
        Conv1d
        AdaptivePool1dApi
        FlattenApi
        SoftmaxApi
        ReluApi
        LinearApi
        SgdApi
        Sgd
        Optimizer
        StateDictApi
        LayerQuant
        LayerCommon
        TensorApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        StorageApi
)
__register_target_as_unit_test(UnitTestGroupNormIntegration)

add_executable(UnitTestAdaptiveAvgPool1dIntegration UnitTestAdaptiveAvgPool1dIntegration.c)
target_link_libraries(UnitTestAdaptiveAvgPool1dIntegration PRIVATE
        unity
        TrainingLoopApi
        CalculateGradsSequential
        CommonLayerLibs
        AdaptivePool1dApi
        SgdApi
        Sgd
        Optimizer
        StateDictApi
        LayerQuant
        TensorApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        StorageApi
)
__register_target_as_unit_test(UnitTestAdaptiveAvgPool1dIntegration)

add_executable(UnitTestMultiLayerTraining UnitTestMultiLayerTraining.c)
target_link_libraries(UnitTestMultiLayerTraining PRIVATE
        unity
        TrainingLoopApi
        TrainingEpochDefault
        TrainingBatchDefault
        CalculateGradsSequential
        CommonLayerLibs
        TensorApi
        LinearApi
        LayerQuant
        ArithmeticType
        ReluApi
        SoftmaxApi
        SgdApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        StorageApi
)
__register_target_as_unit_test(UnitTestMultiLayerTraining)

add_executable(UnitTestMnistSmoke UnitTestMnistSmoke.c)
target_link_libraries(UnitTestMnistSmoke PRIVATE
        unity
        TrainingLoopApi
        TrainingEpochDefault
        TrainingBatchDefault
        CalculateGradsSequential
        CommonLayerLibs
        TensorApi
        LinearApi
        LayerQuant
        ReluApi
        SoftmaxApi
        SgdApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        StorageApi
)
__register_target_as_unit_test(UnitTestMnistSmoke)

add_executable(UnitTestFlattenIntegration UnitTestFlattenIntegration.c)
target_link_libraries(UnitTestFlattenIntegration PRIVATE
        unity
        TrainingLoopApi
        CalculateGradsSequential
        CommonLayerLibs
        FlattenApi
        LinearApi
        SoftmaxApi
        LayerQuant
        SgdApi
        TensorApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        StorageApi
)
__register_target_as_unit_test(UnitTestFlattenIntegration)

add_executable(UnitTestBiaslessConvOptim UnitTestBiaslessConvOptim.c)
target_link_libraries(UnitTestBiaslessConvOptim PRIVATE
        unity
        Conv1dApi
        Conv1d
        Conv1dTransposedApi
        Conv1dTransposed
        ConvTranspose1dKernel
        Conv1dKernel
        SlidingWindow1d
        LinearApi
        Linear
        SgdApi
        Sgd
        Optimizer
        ArithmeticType
        Common
        LayerQuant
        LayerCommon
        QuantizationApi
        Quantization
        Rounding
        Distributions
        RNG
        Kernel
        Tensor
        TensorApi
        StorageApi
)
__register_target_as_unit_test(UnitTestBiaslessConvOptim)

add_executable(UnitTestOptimizerScaling UnitTestOptimizerScaling.c)
target_link_libraries(UnitTestOptimizerScaling PRIVATE
        unity
        OptimizerApi
        SgdApi
        Sgd
        Optimizer
        LinearApi
        ArithmeticType
        CommonLayerLibs
        TensorApi
        QuantizationApi
        StorageApi
)
__register_target_as_unit_test(UnitTestOptimizerScaling)

add_executable(UnitTestQuantizationApiBitsControl UnitTestQuantizationApiBitsControl.c)
target_link_libraries(UnitTestQuantizationApiBitsControl PRIVATE
        unity
        QuantizationApi
        Quantization
        Rounding
)
__register_target_as_unit_test(UnitTestQuantizationApiBitsControl)
add_executable(UnitTestQuantLayerIntegration UnitTestQuantLayerIntegration.c)
target_link_libraries(UnitTestQuantLayerIntegration PRIVATE
        unity
        TrainingLoopApi
        CalculateGradsSequential
        CommonLayerLibs
        LayerNormApi
        LayerNorm
        QuantizationLayer
        SgdApi
        Sgd
        Optimizer
        LayerQuant
        ArithmeticType
        LayerCommon
        LinearApi
        QuantLayerApi
        ModelValidationApi
        TensorApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        StorageApi
        Conv1dTransposed
        Conv1dKernel
        ConvTranspose1dKernel
        Kernel
)
__register_target_as_unit_test(UnitTestQuantLayerIntegration)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        ModelValidationApi
        MORE_LIBS
        ArithmeticType
        CommonLayerLibs
        LinearApi
        LayerCommon
        LayerQuant
        QuantizationApi
        Quantization
        Rounding
        TensorApi
        StorageApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        LayerConfigAccess
        MORE_LIBS
        ArithmeticType
        CommonLayerLibs
        LinearApi
        ReluApi
        SoftmaxApi
        Conv1dApi
        Conv1dTransposedApi
        Pool1dApi
        AdaptivePool1dApi
        DropoutApi
        LayerNormApi
        QuantLayerApi
        FlattenApi
        LayerCommon
        LayerQuant
        QuantizationApi
        Quantization
        Rounding
        Tensor
        TensorApi
        StorageApi
)
