find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GEN_CONV1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv1d.h)
add_custom_command(
        OUTPUT ${GEN_CONV1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d.py
                --out ${GEN_CONV1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_conv1d.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv1d
        DEPENDS ${GEN_CONV1D_HEADER})

set(GEN_CONV1D_GROUPED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv1d_grouped.h)
add_custom_command(
        OUTPUT ${GEN_CONV1D_GROUPED_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_grouped.py
                --out ${GEN_CONV1D_GROUPED_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_grouped.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_conv1d_grouped.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv1d_grouped
        DEPENDS ${GEN_CONV1D_GROUPED_HEADER})

set(GEN_CONV1D_TRANSPOSED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv1d_transposed.h)
add_custom_command(
        OUTPUT ${GEN_CONV1D_TRANSPOSED_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_transposed.py
                --out ${GEN_CONV1D_TRANSPOSED_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_transposed.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_conv1d_transposed.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv1d_transposed
        DEPENDS ${GEN_CONV1D_TRANSPOSED_HEADER})

set(GEN_CONVT1D_GROUPED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_convT1d_grouped.h)
add_custom_command(
        OUTPUT ${GEN_CONVT1D_GROUPED_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_convT1d_grouped.py
                --out ${GEN_CONVT1D_GROUPED_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_convT1d_grouped.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_convT1d_grouped.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_convT1d_grouped
        DEPENDS ${GEN_CONVT1D_GROUPED_HEADER})

set(GEN_LINEAR_GROUPED_ASYM_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_linear_grouped_asym.h)
add_custom_command(
        OUTPUT ${GEN_LINEAR_GROUPED_ASYM_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_linear_grouped_asym.py
                --out ${GEN_LINEAR_GROUPED_ASYM_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_linear_grouped_asym.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_linear_grouped_asym.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_linear_grouped_asym
        DEPENDS ${GEN_LINEAR_GROUPED_ASYM_HEADER})

# BFP epic PR2 (Task 7): ONE fixture header for the three GEMM-family layer
# forward tests (Linear/Conv1d/ConvT1d share the binary dir; each test target
# below adds its own dependency on this one generate target).
set(GEN_BFP_LAYER_FORWARD_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_bfp_layer_forward.h)
add_custom_command(
        OUTPUT ${GEN_BFP_LAYER_FORWARD_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_bfp_layer_forward.py
                --out ${GEN_BFP_LAYER_FORWARD_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_bfp_layer_forward.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_bfp_layer_forward.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_bfp_layer_forward
        DEPENDS ${GEN_BFP_LAYER_FORWARD_HEADER})

set(GEN_MAX_POOL_1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_max_pool_1d.h)
add_custom_command(
        OUTPUT ${GEN_MAX_POOL_1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_max_pool_1d.py
                --out ${GEN_MAX_POOL_1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_max_pool_1d.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_max_pool_1d.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_max_pool_1d
        DEPENDS ${GEN_MAX_POOL_1D_HEADER})

set(GEN_AVG_POOL_1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_avg_pool_1d.h)
add_custom_command(
        OUTPUT ${GEN_AVG_POOL_1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_avg_pool_1d.py
                --out ${GEN_AVG_POOL_1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_avg_pool_1d.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_avg_pool_1d.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_avg_pool_1d
        DEPENDS ${GEN_AVG_POOL_1D_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1d
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        Conv1dApi
        LayerQuant
        Conv1dKernel
        ConvTranspose1dKernel
        Kernel
        Quantization
        StorageApi
        odt_test_death
        odt_test_layer_builders
)
add_dependencies(UnitTestConv1d generate_expected_conv1d)
add_dependencies(UnitTestConv1d generate_expected_conv1d_grouped)
add_dependencies(UnitTestConv1d generate_expected_bfp_layer_forward)
# #416 (BFP PR3): the layer-level conv-groups=2 dx test consumes the SAME
# operand fixture as the kernel-level parity test, single-sourced in the
# arithmetic generator's header (two generators re-embedding the values
# would silently drift apart) -- dependency and include dir stay PAIRED or
# parallel builds race a missing header.
add_dependencies(UnitTestConv1d generate_expected_bfp_conv1d)
target_include_directories(UnitTestConv1d PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/../arithmetic)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Linear
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        LinearApi
        LayerQuant
        LayerWeightsApi
        Quantization
        Rounding
        StorageApi
        Add
        Distributions
        RNG
        OptimizerApi
        SgdApi
        Sgd
        Optimizer
        odt_test_death
        odt_test_layer_builders
)
add_dependencies(UnitTestLinear generate_expected_linear_grouped_asym)
add_dependencies(UnitTestLinear generate_expected_bfp_layer_forward)
target_include_directories(UnitTestLinear PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Relu
        MORE_LIBS
        ArithmeticType
        CommonLayerLibs
        ReluApi
        TensorApi
        QuantizationApi
        LayerQuant
        Quantization
        Rounding
        StorageApi
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Softmax
        MORE_LIBS
        ArithmeticType
        CommonLayerLibs
        TensorApi
        SoftmaxApi
        QuantizationApi
        LayerQuant
        Quantization
        Rounding
        StorageApi
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Flatten
        MORE_LIBS
        CommonLayerLibs
        FlattenApi
        TensorApi
        QuantizationApi
        StorageApi
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1dTransposed
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        Conv1dTransposedApi
        LayerQuant
        Conv1dKernel
        ConvTranspose1dKernel
        Kernel
        StorageApi
        odt_test_death
)
add_dependencies(UnitTestConv1dTransposed generate_expected_conv1d_transposed)
add_dependencies(UnitTestConv1dTransposed generate_expected_convT1d_grouped)
add_dependencies(UnitTestConv1dTransposed generate_expected_bfp_layer_forward)
target_include_directories(UnitTestConv1dTransposed PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        MaxPool1d
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        StorageApi
        odt_test_death
)
add_dependencies(UnitTestMaxPool1d generate_expected_max_pool_1d)
target_include_directories(UnitTestMaxPool1d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        AvgPool1d
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        StorageApi
        odt_test_death
)
add_dependencies(UnitTestAvgPool1d generate_expected_avg_pool_1d)
target_include_directories(UnitTestAvgPool1d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

set(GEN_ADAPTIVE_AVG_POOL_1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_adaptive_avg_pool_1d.h)
add_custom_command(
        OUTPUT ${GEN_ADAPTIVE_AVG_POOL_1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_adaptive_avg_pool_1d.py
                --out ${GEN_ADAPTIVE_AVG_POOL_1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_adaptive_avg_pool_1d.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_adaptive_avg_pool_1d.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_adaptive_avg_pool_1d
        DEPENDS ${GEN_ADAPTIVE_AVG_POOL_1D_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        AdaptiveAvgPool1d
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        StorageApi
        odt_test_death
)
add_dependencies(UnitTestAdaptiveAvgPool1d generate_expected_adaptive_avg_pool_1d)
target_include_directories(UnitTestAdaptiveAvgPool1d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Dropout
        MORE_LIBS
        CommonLayerLibs
        Bernoulli
        RNG
        TensorApi
        QuantizationApi
        LayerQuant
        Quantization
        Rounding
        StorageApi
        DropoutApi
        odt_test_death
)

set(GEN_LAYERNORM_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_layernorm.h)
add_custom_command(
        OUTPUT ${GEN_LAYERNORM_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_layernorm.py
                --out ${GEN_LAYERNORM_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_layernorm.py
        COMMENT "Generating expected_layernorm.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_layernorm
        DEPENDS ${GEN_LAYERNORM_HEADER})

set(GEN_LAYERNORM_SYM_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_layernorm_sym.h)
add_custom_command(
        OUTPUT ${GEN_LAYERNORM_SYM_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_layernorm_sym.py
                --out ${GEN_LAYERNORM_SYM_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_layernorm_sym.py
        COMMENT "Generating expected_layernorm_sym.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_layernorm_sym
        DEPENDS ${GEN_LAYERNORM_SYM_HEADER})

set(GEN_LAYERNORM_SYM_BWD_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_layernorm_sym_bwd.h)
add_custom_command(
        OUTPUT ${GEN_LAYERNORM_SYM_BWD_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_layernorm_sym_bwd.py
                --out ${GEN_LAYERNORM_SYM_BWD_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_layernorm_sym_bwd.py
        COMMENT "Generating expected_layernorm_sym_bwd.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_layernorm_sym_bwd
        DEPENDS ${GEN_LAYERNORM_SYM_BWD_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        LayerNorm
        MORE_LIBS
        ArithmeticType
        CommonLayerLibs
        TensorApi
        QuantizationApi
        LayerNormApi
        LayerQuant
        Quantization
        Rounding
        Arithmetic
        StorageApi
        odt_test_death
)
add_dependencies(UnitTestLayerNorm generate_expected_layernorm)
add_dependencies(UnitTestLayerNorm generate_expected_layernorm_sym)
add_dependencies(UnitTestLayerNorm generate_expected_layernorm_sym_bwd)
target_include_directories(UnitTestLayerNorm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

set(GEN_GROUPNORM_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_groupnorm.h)
add_custom_command(
        OUTPUT ${GEN_GROUPNORM_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_groupnorm.py
                --out ${GEN_GROUPNORM_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_groupnorm.py
        COMMENT "Generating expected_groupnorm.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_groupnorm
        DEPENDS ${GEN_GROUPNORM_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        GroupNorm
        MORE_LIBS
        ArithmeticType
        CommonLayerLibs
        TensorApi
        QuantizationApi
        GroupNormApi
        LayerQuant
        Quantization
        Rounding
        Arithmetic
        StorageApi
        Reduce
        odt_test_death
)
add_dependencies(UnitTestGroupNorm generate_expected_groupnorm)
target_include_directories(UnitTestGroupNorm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        QuantizationLayer
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        TensorConversion
        QuantizationApi
        LayerQuant
        QuantLayerApi
        Quantization
        Rounding
        StorageApi
        odt_test_death
)
