add_library(DTypes
        DTypes.c
)
target_include_directories(DTypes PUBLIC include)

add_library(DataStorage DataStorage.c)

target_include_directories(DataStorage PUBLIC include)
target_link_libraries(DataStorage PRIVATE
        Tensor
)

add_library(Quantization
        Quantization.c
)
target_include_directories(Quantization PUBLIC include)
target_link_libraries(Quantization PRIVATE
        DyadicScale
        Common
        Rounding
        StorageApi
        m
)

add_library(TensorConversion
        TensorConversion.c
)
target_include_directories(TensorConversion PUBLIC include)
target_link_libraries(TensorConversion PRIVATE
        DyadicScale
        Tensor
        Quantization
        Rounding
        RNG
        Common
)

add_library(Tensor
        Tensor.c
)
target_include_directories(Tensor PUBLIC include)
target_link_libraries(Tensor PRIVATE
        DyadicScale
        DTypes
        DataStorage
        MinMax
        m
        Rounding
        Quantization
        Common
)

add_library(DyadicScale
        DyadicScale.c
)
target_include_directories(DyadicScale PUBLIC include)
target_link_libraries(DyadicScale PRIVATE
        Common
)
# ODT_SCALE_STRICT (cmake/OdtOptions.cmake): every canonical store site of a
# SYM / SYM_INT32 / ASYM scale asserts the stored float is an exact 15-bit
# dyadic value (odtScaleAssertCanonical, DyadicScale.h; fail fast otherwise).
# OFF, the assert compiles to nothing. The unit_test_scale_strict preset turns
# it on. PUBLIC on DyadicScale, the one target that exports DyadicScale.h, so
# every TU that sees the header sees the same declaration.
if(ODT_SCALE_STRICT)
    target_compile_definitions(DyadicScale PUBLIC ODT_SCALE_STRICT=1)
endif()
