add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Add
        MORE_LIBS
        Tensor
        Arithmetic
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
        Comparison
        MORE_LIBS
        Arithmetic
        Tensor
        m
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