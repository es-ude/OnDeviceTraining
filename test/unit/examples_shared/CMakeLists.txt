# examples/_shared is example tooling, not framework code, but its sink and
# gate are the sweep's observability layer -- same unit-test discipline.
# Explicit add_executable (not add_elastic_ai_unit_test): the library name
# examples_shared does not follow the UnitTest<Lib> naming rule.
add_executable(UnitTestNpyDumpSink UnitTestNpyDumpSink.c)
target_link_libraries(UnitTestNpyDumpSink PRIVATE
        unity
        examples_shared
        Layer
        NPYLoaderApi
        NPYLoader
        DataLoader
        TensorApi
        QuantizationApi
        TensorConversion
        Tensor
        Quantization
        Rounding
        Common
        StorageApi
        odt_test_death
)
target_compile_definitions(UnitTestNpyDumpSink PRIVATE
        NPY_DUMP_SINK_TEST_DIR="${CMAKE_CURRENT_BINARY_DIR}")
__register_target_as_unit_test(UnitTestNpyDumpSink)
