find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GEN_CONV1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv1d.h)
add_custom_command(
        OUTPUT ${GEN_CONV1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d.py
                --out ${GEN_CONV1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d.py
        COMMENT "Generating expected_conv1d.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv1d
        DEPENDS ${GEN_CONV1D_HEADER})

set(GEN_CONV1D_TRANSPOSED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv1d_transposed.h)
add_custom_command(
        OUTPUT ${GEN_CONV1D_TRANSPOSED_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_transposed.py
                --out ${GEN_CONV1D_TRANSPOSED_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_transposed.py
        COMMENT "Generating expected_conv1d_transposed.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv1d_transposed
        DEPENDS ${GEN_CONV1D_TRANSPOSED_HEADER})

set(GEN_MAX_POOL_1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_max_pool_1d.h)
add_custom_command(
        OUTPUT ${GEN_MAX_POOL_1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_max_pool_1d.py
                --out ${GEN_MAX_POOL_1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_max_pool_1d.py
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
        Kernel
        QuantizationApi
        StorageApi
)
add_dependencies(UnitTestConv1d generate_expected_conv1d)
target_include_directories(UnitTestConv1d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Linear
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        LinearApi
        LayerQuant
        Quantization
        Rounding
        StorageApi
        Add
        SgdApi
        Sgd
        Optimizer
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Relu
        MORE_LIBS
        CommonLayerLibs
        ReluApi
        TensorApi
        QuantizationApi
        LayerQuant
        Quantization
        Rounding
        StorageApi
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Softmax
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        SoftmaxApi
        QuantizationApi
        LayerQuant
        Quantization
        Rounding
        StorageApi
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
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1dTransposed
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        Kernel
        StorageApi
)
add_dependencies(UnitTestConv1dTransposed generate_expected_conv1d_transposed)
target_include_directories(UnitTestConv1dTransposed PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        MaxPool1d
        MORE_LIBS
        CommonLayerLibs
        TensorApi
        QuantizationApi
        StorageApi
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
)
add_dependencies(UnitTestAvgPool1d generate_expected_avg_pool_1d)
target_include_directories(UnitTestAvgPool1d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

set(GEN_ADAPTIVE_AVG_POOL_1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_adaptive_avg_pool_1d.h)
add_custom_command(
        OUTPUT ${GEN_ADAPTIVE_AVG_POOL_1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_adaptive_avg_pool_1d.py
                --out ${GEN_ADAPTIVE_AVG_POOL_1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_adaptive_avg_pool_1d.py
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
        CommonLayerLibs
        TensorApi
        QuantizationApi
        LayerNormApi
        LayerQuant
        Quantization
        Rounding
        Arithmetic
        StorageApi
)
add_dependencies(UnitTestLayerNorm generate_expected_layernorm)
add_dependencies(UnitTestLayerNorm generate_expected_layernorm_sym)
add_dependencies(UnitTestLayerNorm generate_expected_layernorm_sym_bwd)
target_include_directories(UnitTestLayerNorm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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
)
