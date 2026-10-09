add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        RNG
        MORE_LIBS
        odt_test_death
)

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Bernoulli
        MORE_LIBS
        RNG
        Tensor
        Rounding
        TensorApi
        QuantizationApi
        StorageApi
)
