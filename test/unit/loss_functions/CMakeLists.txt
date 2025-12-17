add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        CrossEntropy
        MORE_LIBS
        CommonLayerLibs
        Log
        SoftmaxAPI
        QuantizationAPI
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        MSE
        MORE_LIBS
        CommonLayerLibs
        TensorAPI
)