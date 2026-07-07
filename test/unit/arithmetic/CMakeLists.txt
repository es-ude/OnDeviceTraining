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

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Add
        MORE_LIBS
        Tensor
        TensorApi
        QuantizationApi
        Arithmetic
        StorageApi
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
        odt_test_death
)

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

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Conv1dKernel
        MORE_LIBS
        SlidingWindow1d
        Kernel
        Tensor
        TensorApi
        QuantizationApi
        StorageApi
)
add_dependencies(UnitTestConv1dKernel generate_expected_conv1d_kernel)
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
