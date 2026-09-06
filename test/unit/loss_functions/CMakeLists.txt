set(GEN_CROSS_ENTROPY_SYM_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_cross_entropy_sym.h)
add_custom_command(
        OUTPUT ${GEN_CROSS_ENTROPY_SYM_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_cross_entropy_sym.py
                --out ${GEN_CROSS_ENTROPY_SYM_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_cross_entropy_sym.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_cross_entropy_sym.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_cross_entropy_sym
        DEPENDS ${GEN_CROSS_ENTROPY_SYM_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        CrossEntropy
        MORE_LIBS
        CommonLayerLibs
        Log
        SoftmaxApi
        QuantizationApi
        LayerQuant
        TensorApi
        TensorConversion
        Quantization
        Rounding
        StorageApi
)
add_dependencies(UnitTestCrossEntropy generate_expected_cross_entropy_sym)
target_include_directories(UnitTestCrossEntropy PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        MSE
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        TensorConversion
        Quantization
        StorageApi
        QuantizationApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        LossFunction
        MORE_LIBS
        CommonLayerLibs
        MSE
        CrossEntropy
        TensorApi
        QuantizationApi
        StorageApi
)