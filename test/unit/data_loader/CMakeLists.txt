add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        DataLoader
        MORE_LIBS

        TensorApi
        Tensor
        Quantization
        Rounding

        QuantizationApi

        StorageApi

        DataLoaderApi

        NPYLoaderApi
        NPYLoader

        RNG

        odt_test_death
        odt_test_assert
)

target_compile_definitions(UnitTestDataLoader PRIVATE
        PROXY_X_PATH="${CMAKE_CURRENT_SOURCE_DIR}/proxy_x.npy"
        PROXY_Y_PATH="${CMAKE_CURRENT_SOURCE_DIR}/proxy_y.npy"
        PROXY_W_PATH="${CMAKE_CURRENT_SOURCE_DIR}/proxy_w.npy"
)