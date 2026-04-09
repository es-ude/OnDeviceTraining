add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        InferenceApi
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        LinearApi
        ReluApi
        QuantizationApi
        LossFunction
        DataLoader
)


add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        TrainingLoopApi
        MORE_LIBS
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        CommonLayerLibs
        TensorApi
        LinearApi
        ReluApi
        SgdApi
        QuantizationApi
        LossFunction
        InferenceApi
        DataLoader
        DataLoaderApi
        StorageApi
)

add_executable(UnitTestMultiLayerTraining UnitTestMultiLayerTraining.c)
target_link_libraries(UnitTestMultiLayerTraining PRIVATE
        unity
        CalculateGradsSequential
        TrainingBatchDefault
        TrainingEpochDefault
        TrainingLoopApi
        CommonLayerLibs
        TensorApi
        LinearApi
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