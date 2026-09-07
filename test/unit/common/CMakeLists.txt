add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Common
)

# Phase hook (#419 precondition): the library is two functions, but the test
# drives a real Linear+Softmax model through calculateGradsSequential /
# tracedGrads / trainingEpochDefault + an SGD step, hence the long link list
# (mirrors test/unit/training_loop + the TrainingLoopApi test).
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        OdtHook
        MORE_LIBS
        CalculateGradsSequential
        TrainingEpochDefault
        TrainingBatchDefault
        CommonLayerLibs
        LinearApi
        ReluApi
        SoftmaxApi
        LayerQuant
        ArithmeticType
        LayerCommon
        LayerWeightsApi
        StateDictApi
        QuantizationApi
        Quantization
        Rounding
        TensorApi
        Tensor
        StorageApi
        Common
        Optimizer
        OptimizerApi
        SgdApi
        Sgd
        TrainingLoopApi
        InferenceApi
        DataLoader
        DataLoaderApi
        LossFunction
        CrossEntropy
        Distributions
        RNG
        odt_test_layer_builders
)
