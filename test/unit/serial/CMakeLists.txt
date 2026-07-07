add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Deserialize
        MORE_LIBS
        TensorApi
        Tensor
        Rounding
        Quantization
        Serialize

        QuantizationApi
        Layer
        Flatten
        FlattenApi

        StorageApi
        odt_test_death
)

# Absolute scratch-file path so the test runs from any working directory
# (host build dir vs. Docker /work, see Phase 2 LSan migration).
target_compile_definitions(UnitTestDeserialize PRIVATE
        SERIALIZE_TEST_FILE_PATH="${CMAKE_CURRENT_BINARY_DIR}/SerializeTestFile.bin"
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Serialize
        MORE_LIBS
        Common
        Deserialize
        TensorApi
        Tensor
        Rounding
        Quantization

        QuantizationApi
        Layer
        Linear
        LinearApi
        Relu
        ReluApi
        Softmax
        SoftmaxApi
        Flatten
        FlattenApi
        LayerQuant
        ArithmeticType

        Kernel
        Conv1d
        Conv1dApi
        Conv1dTransposed
        Conv1dTransposedApi
        MaxPool1d
        AvgPool1d
        Pool1dApi
        AdaptiveAvgPool1d
        AdaptivePool1dApi
        Dropout
        DropoutApi
        LayerNorm
        LayerNormApi
        GroupNorm
        GroupNormApi
        Reduce
        QuantizationLayer
        QuantLayerApi

        StorageApi
)

target_compile_definitions(UnitTestSerialize PRIVATE
        SERIALIZE_TEST_FILE_PATH="${CMAKE_CURRENT_BINARY_DIR}/SerializeRoundTripTestFile.bin"
)
