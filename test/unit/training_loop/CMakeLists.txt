add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        CalculateGradsSequential
        MORE_LIBS
        CommonLayerLibs
        LinearApi
        ReluApi
        SoftmaxApi
        Pool1dApi
        Kernel
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

# Remat PR1a (#4): the RematPlan library (buffer table + static plan). Fixture
# models are built with the real layer factories.
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        RematPlan
        MORE_LIBS
        RematScheduler
        CommonLayerLibs
        Conv1dApi
        LinearApi
        ReluApi
        SoftmaxApi
        FlattenApi
        Pool1dApi
        LayerNormApi
        QuantLayerApi
        Kernel
        LayerQuant
        LayerCommon
        LayerConfigAccess
        ArithmeticType
        QuantizationApi
        Quantization
        Rounding
        TensorApi
        Tensor
        StorageApi
        Common
        LossFunction
        Distributions
        RNG
        Serialize
        Deserialize
        odt_test_death
)

# Remat (#4): the RematScheduler library, both rows driven through the const
# vtable and the rematBegin/Next/Done/End dispatch. The placement tests reach
# the ARENA-private RematPlace.h through the source directory; RematPlan is
# listed for its PUBLIC ODT_REMAT_VERIFY.
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        RematScheduler
        MORE_LIBS
        RematPlan
        CommonLayerLibs
        Conv1dApi
        LinearApi
        ReluApi
        SoftmaxApi
        FlattenApi
        Pool1dApi
        QuantLayerApi
        LayerNormApi
        Kernel
        LayerQuant
        LayerCommon
        LayerConfigAccess
        ArithmeticType
        QuantizationApi
        Quantization
        Rounding
        TensorApi
        Tensor
        StorageApi
        Common
        LossFunction
        Distributions
        RNG
        odt_test_death
)
target_include_directories(UnitTestRematScheduler PRIVATE
        ${PROJECT_SOURCE_DIR}/src/userApi/training_loop/remat)

# Remat (#4): the RematCheck library, driven over real schedulers on
# both rows. RematScheduler and RematPlan are listed because the tests build
# schedulers and read the table (RematPlan.h).
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        RematCheck
        MORE_LIBS
        RematScheduler
        RematPlan
        CommonLayerLibs
        Conv1dApi
        LinearApi
        ReluApi
        SoftmaxApi
        FlattenApi
        Pool1dApi
        QuantLayerApi
        LayerNormApi
        Kernel
        LayerQuant
        LayerCommon
        LayerConfigAccess
        ArithmeticType
        QuantizationApi
        Quantization
        Rounding
        TensorApi
        Tensor
        StorageApi
        Common
        LossFunction
        Distributions
        RNG
        odt_test_death
)

# Remat PR2 (#4): the driver against the Legacy oracle (remat D42) over a
# fixture zoo. Registered by hand, like UnitTestDeathTest:
# add_elastic_ai_unit_test names the binary after its LIB_UNDER_TEST, and this
# binary tests the driver against a second library rather than one library on
# its own.
add_executable(UnitTestCalculateGradsConformance UnitTestCalculateGradsConformance.c)
target_link_libraries(UnitTestCalculateGradsConformance PRIVATE
        CalculateGradsSequential
        odt_test_legacy_grads
        unity
        RematScheduler
        RematPlan
        CommonLayerLibs
        Conv1dApi
        LinearApi
        ReluApi
        SoftmaxApi
        FlattenApi
        Pool1dApi
        QuantLayerApi
        LayerNormApi
        GroupNormApi
        DropoutApi
        BatchNorm1dApi
        Kernel
        LayerQuant
        LayerCommon
        LayerConfigAccess
        ArithmeticType
        QuantizationApi
        Quantization
        Rounding
        TensorApi
        Tensor
        StorageApi
        Common
        OdtHook
        TrainingLoopApi
        InferenceApi
        DataLoader
        LossFunction
        Distributions
        RNG
        odt_test_death
)
__register_target_as_unit_test(UnitTestCalculateGradsConformance)

# Remat PR2 (#4): the driver-level deaths (tampered plans, decorator rows,
# schedule-key mismatches), each asserted on both rows. Registered by hand
# like the conformance binary; a binary of its own because the deaths need
# small models and decorator rows, not the zoo or Legacy. RematPlan is listed
# because the tamper tests edit the plan (RematPlan.h).
add_executable(UnitTestCalculateGradsDeaths UnitTestCalculateGradsDeaths.c)
target_link_libraries(UnitTestCalculateGradsDeaths PRIVATE
        CalculateGradsSequential
        unity
        RematScheduler
        RematPlan
        CommonLayerLibs
        Conv1dApi
        LinearApi
        ReluApi
        SoftmaxApi
        FlattenApi
        Pool1dApi
        QuantLayerApi
        LayerNormApi
        Kernel
        LayerQuant
        LayerCommon
        LayerConfigAccess
        ArithmeticType
        QuantizationApi
        Quantization
        Rounding
        TensorApi
        Tensor
        StorageApi
        Common
        OdtHook
        TrainingLoopApi
        InferenceApi
        DataLoader
        LossFunction
        Distributions
        RNG
        odt_test_death
)
__register_target_as_unit_test(UnitTestCalculateGradsDeaths)
