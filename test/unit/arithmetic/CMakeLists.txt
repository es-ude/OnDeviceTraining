find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GEN_CONV1D_KERNEL_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv1d_kernel.h)
add_custom_command(
        OUTPUT ${GEN_CONV1D_KERNEL_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_kernel.py
                --out ${GEN_CONV1D_KERNEL_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv1d_kernel.py
        COMMENT "Generating expected_conv1d_kernel.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv1d_kernel
        DEPENDS ${GEN_CONV1D_KERNEL_HEADER})

set(GEN_CONVT_1D_KERNEL_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_conv_transpose_1d_kernel.h)
add_custom_command(
        OUTPUT ${GEN_CONVT_1D_KERNEL_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv_transpose_1d_kernel.py
                --out ${GEN_CONVT_1D_KERNEL_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_conv_transpose_1d_kernel.py
        COMMENT "Generating expected_conv_transpose_1d_kernel.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_conv_transpose_1d_kernel
        DEPENDS ${GEN_CONVT_1D_KERNEL_HEADER})

set(GEN_GROUP_MATMUL_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_group_matmul.h)
add_custom_command(
        OUTPUT ${GEN_GROUP_MATMUL_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_group_matmul.py
                --out ${GEN_GROUP_MATMUL_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_group_matmul.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_group_matmul.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_group_matmul
        DEPENDS ${GEN_GROUP_MATMUL_HEADER})

set(GEN_BFP_MATMUL_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_bfp_matmul.h)
add_custom_command(
        OUTPUT ${GEN_BFP_MATMUL_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_bfp_matmul.py
                --out ${GEN_BFP_MATMUL_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_bfp_matmul.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_bfp_matmul.h via NumPy emulation"
        VERBATIM
)
add_custom_target(generate_expected_bfp_matmul
        DEPENDS ${GEN_BFP_MATMUL_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Add
        MORE_LIBS
        Tensor
        TensorApi
        QuantizationApi
        Arithmetic
        StorageApi
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Arithmetic
        MORE_LIBS
        Tensor
        Quantization
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        ArithmeticType
        MORE_LIBS
        Quantization
        Rounding
        Common
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Axpby
        MORE_LIBS
        Tensor
        TensorApi
        QuantizationApi
        StorageApi
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Comparison
        MORE_LIBS
        Arithmetic
        Tensor
        m
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Distributions
        MORE_LIBS
        MinMax
        RNG
        Common
        TensorApi
        QuantizationApi
        StorageApi
        Tensor
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Div
        MORE_LIBS
        Arithmetic
        Tensor
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        JacobiEig
        MORE_LIBS
        m
        Tensor
        TensorApi
        QuantizationApi
        StorageApi
        Reduce
        MinMax
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Log
        MORE_LIBS
        Arithmetic
        Tensor
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Matmul
        MORE_LIBS
        Arithmetic
        Tensor
        Quantization
        RNG
        Common
        odt_test_death
)
add_dependencies(UnitTestMatmul generate_expected_group_matmul generate_expected_bfp_matmul)
target_include_directories(UnitTestMatmul PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Mul
        MORE_LIBS
        Arithmetic
        Tensor
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Reduce
        MORE_LIBS
        m
        Tensor
        TensorApi
        QuantizationApi
        Arithmetic
        StorageApi
        MinMax
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        RowBroadcast
        MORE_LIBS
        Tensor
        TensorApi
        QuantizationApi
        StorageApi
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Rounding
        MORE_LIBS
        Tensor
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Square
        MORE_LIBS
        Arithmetic
        Tensor
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Sub
        MORE_LIBS
        Arithmetic
        Tensor
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        SlidingWindow1d
        MORE_LIBS
        Kernel
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        AdaptiveWindow1d
)

set(GEN_BFP_CONV1D_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_bfp_conv1d.h)
add_custom_command(
        OUTPUT ${GEN_BFP_CONV1D_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_bfp_conv1d.py
                --out ${GEN_BFP_CONV1D_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_bfp_conv1d.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../goldgen/sym_gold.py
        COMMENT "Generating expected_bfp_conv1d.h via NumPy emulation"
        VERBATIM
)
add_custom_target(generate_expected_bfp_conv1d
        DEPENDS ${GEN_BFP_CONV1D_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1dKernel
        MORE_LIBS
        SlidingWindow1d
        Kernel
        Tensor
        TensorApi
        Quantization
        QuantizationApi
        StorageApi
        odt_test_death
)
add_dependencies(UnitTestConv1dKernel generate_expected_conv1d_kernel
        generate_expected_bfp_conv1d)
target_include_directories(UnitTestConv1dKernel PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        ConvTranspose1dKernel
        MORE_LIBS
        Kernel
        Tensor
        TensorApi
        QuantizationApi
        StorageApi
)
add_dependencies(UnitTestConvTranspose1dKernel generate_expected_conv_transpose_1d_kernel)
target_include_directories(UnitTestConvTranspose1dKernel PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        ExecuteOp
        MORE_LIBS
        Tensor
        TensorApi
        TensorConversion
        Quantization
        QuantizationApi
        Add
        Arithmetic
        ArithmeticType
        Rounding
        RNG
        StorageApi
        Common
        odt_test_death
)

set(GEN_POINTWISE_FUSED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_pointwise_fused.h)
add_custom_command(
        OUTPUT ${GEN_POINTWISE_FUSED_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_pointwise_fused.py
                --out ${GEN_POINTWISE_FUSED_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_pointwise_fused.py
        COMMENT "Generating expected_pointwise_fused.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_pointwise_fused
        DEPENDS ${GEN_POINTWISE_FUSED_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        PointwiseFused
        MORE_LIBS
        Tensor
        Common
        odt_test_death
)
add_dependencies(UnitTestPointwiseFused generate_expected_pointwise_fused)
target_include_directories(UnitTestPointwiseFused PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
