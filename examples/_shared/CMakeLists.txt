add_library(examples_shared STATIC npy_writer.c npy_dump_sink.c param_gate.c)
target_include_directories(examples_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Include propagation is PRIVATE repo-wide: list every library whose header a
# source here includes (Layer.h, Tensor.h, TensorConversion.h, TensorApi.h,
# Quantization.h, QuantizationApi.h, Rounding.h, Common.h).
target_link_libraries(examples_shared PRIVATE
        Layer
        Tensor
        TensorConversion
        TensorApi
        Quantization
        QuantizationApi
        Rounding
        Common
)
