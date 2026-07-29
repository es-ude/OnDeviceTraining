add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Quantization
        MORE_LIBS
        Tensor
        Rounding
        Quantization
        StorageApi
        odt_test_death
)
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Tensor
        MORE_LIBS
        Rounding
        Quantization
        StorageApi
        odt_test_death
)
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        TensorConversion
        MORE_LIBS
        Tensor
        Rounding
        Quantization
        TensorApi
        QuantizationApi
        StorageApi
        RNG
        odt_test_death
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
set(GEN_GROUP_QUANT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_group_quant.h)
add_custom_command(
        OUTPUT ${GEN_GROUP_QUANT_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_group_quant.py
                --out ${GEN_GROUP_QUANT_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_group_quant.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_group_quant.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_group_quant
        DEPENDS ${GEN_GROUP_QUANT_HEADER})
add_dependencies(UnitTestTensorConversion generate_expected_group_quant)
set(GEN_ASYM_NUDGED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_asym_nudged.h)
add_custom_command(
        OUTPUT ${GEN_ASYM_NUDGED_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_asym_nudged.py
                --out ${GEN_ASYM_NUDGED_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_asym_nudged.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_asym_nudged.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_asym_nudged
        DEPENDS ${GEN_ASYM_NUDGED_HEADER})
add_dependencies(UnitTestTensorConversion generate_expected_asym_nudged)
add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        TensorApi
        MORE_LIBS
        Tensor
        Rounding
        Quantization
        QuantizationApi
        StorageApi
        odt_test_death
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