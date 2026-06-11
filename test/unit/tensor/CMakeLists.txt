add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Quantization
        MORE_LIBS
        Tensor
        Rounding
        Quantization
)
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Tensor
        MORE_LIBS
        Rounding
        Quantization
        StorageApi
)
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        TensorConversion
        MORE_LIBS
        Tensor
        Rounding
        Quantization
        TensorApi
        StorageApi
)
set(GEN_REQUANT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_requant.h)
add_custom_command(
        OUTPUT ${GEN_REQUANT_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_requant.py
                --out ${GEN_REQUANT_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_requant.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_requant.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_requant
        DEPENDS ${GEN_REQUANT_HEADER})
add_dependencies(UnitTestTensorConversion generate_expected_requant)
target_include_directories(UnitTestTensorConversion PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        TensorApi
        MORE_LIBS
        Tensor
        Rounding
        Quantization
        QuantizationApi
        StorageApi
)
add_executable(UnitTestTensorBool UnitTestTensorBool.c)
target_link_libraries(UnitTestTensorBool PRIVATE
        unity
        Tensor
        Rounding
        Quantization
        StorageApi
)
__register_target_as_unit_test(UnitTestTensorBool)