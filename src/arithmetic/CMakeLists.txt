add_library(Add Add.c)
target_include_directories(Add PUBLIC include)
target_link_libraries(Add PRIVATE
        Tensor
        DTypes
        Arithmetic
        TensorConversion
        Common
)

add_library(Arithmetic Arithmetic.c)
target_include_directories(Arithmetic PUBLIC include)
target_link_libraries(Arithmetic PRIVATE
        Tensor
        DTypes
        Add
        Div
        Matmul
        Mul
        Square
        Sub
        Common
)

add_library(ArithmeticType ArithmeticType.c)
target_include_directories(ArithmeticType PUBLIC include)
target_link_libraries(ArithmeticType PRIVATE
        Quantization
        Rounding
)

add_library(Axpby Axpby.c)
target_include_directories(Axpby PUBLIC include)
target_link_libraries(Axpby PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(Comparison Comparison.c)
target_include_directories(Comparison PUBLIC include)
target_link_libraries(Comparison PRIVATE
        Tensor
        DTypes
        Common
)

add_library(Distributions Distributions.c)
target_include_directories(Distributions PUBLIC include)
target_link_libraries(Distributions PUBLIC RNG PRIVATE
        Common
        m
        Tensor
        DTypes
        Quantization
        Rounding
)

add_library(Div Div.c)
target_include_directories(Div PUBLIC include)
target_link_libraries(Div PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(JacobiEig JacobiEig.c)
target_include_directories(JacobiEig PUBLIC include)
target_link_libraries(JacobiEig PRIVATE
        m
        Common
        Tensor
        DTypes
        Quantization
        Reduce
        MinMax
)

add_library(Kernel Kernel.c)
target_include_directories(Kernel PUBLIC include)
target_link_libraries(Kernel PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
        Layer
)

add_library(Log Log.c)
target_include_directories(Log PUBLIC include)
target_link_libraries(Log PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(Matmul Matmul.c)
target_include_directories(Matmul PUBLIC include)
target_link_libraries(Matmul PRIVATE
        Common
        Tensor
        DTypes
        Mul
        Rounding
        Quantization
        m
)

add_library(Mul Mul.c)
target_include_directories(Mul PUBLIC include)
target_link_libraries(Mul PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(MinMax MinMax.c)
target_include_directories(MinMax PUBLIC include)
target_link_libraries(MinMax PRIVATE
        Tensor
        DTypes
        Common
)

add_library(Reduce Reduce.c)
target_include_directories(Reduce PUBLIC include)
target_link_libraries(Reduce PRIVATE m Common Div Sub Add Mul Square Arithmetic Tensor DTypes Quantization Rounding MinMax)

add_library(RowBroadcast RowBroadcast.c)
target_include_directories(RowBroadcast PUBLIC include)
target_link_libraries(RowBroadcast PRIVATE
        Common
        Tensor
        DTypes
        Quantization
)

add_library(Rounding Rounding.c)
target_include_directories(Rounding PUBLIC include)
target_link_libraries(Rounding PRIVATE
        m
        RNG
        Common
)


add_library(Square Square.c)
target_include_directories(Square PUBLIC include)
target_link_libraries(Square PRIVATE
        Tensor
        DTypes
        Common
)

add_library(Sub Sub.c)
target_include_directories(Sub PUBLIC include)
target_link_libraries(Sub PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(Sum Sum.c)
target_include_directories(Sum PUBLIC include)
target_link_libraries(Sum PRIVATE
        Add
        Tensor
        Rounding
        Common
)

add_library(SlidingWindow1d SlidingWindow1d.c)
target_include_directories(SlidingWindow1d PUBLIC include)
target_link_libraries(SlidingWindow1d PRIVATE
        Kernel
        Common
)

add_library(AdaptiveWindow1d AdaptiveWindow1d.c)
target_include_directories(AdaptiveWindow1d PUBLIC include)
target_link_libraries(AdaptiveWindow1d PRIVATE
        Common
)

add_library(Conv1dKernel Conv1dKernel.c)
target_include_directories(Conv1dKernel PUBLIC include)
target_link_libraries(Conv1dKernel PRIVATE
        Tensor
        Kernel
        SlidingWindow1d
        Mul
        Rounding
        Quantization
        Common
        m
)

add_library(ConvTranspose1dKernel ConvTranspose1dKernel.c)
target_include_directories(ConvTranspose1dKernel PUBLIC include)
target_link_libraries(ConvTranspose1dKernel PRIVATE
        Tensor
        Kernel
        SlidingWindow1d
        Mul
        Rounding
        Common
)

add_library(ExecuteOp ExecuteOp.c)
target_include_directories(ExecuteOp PUBLIC include)
target_link_libraries(ExecuteOp PRIVATE
        Tensor
        Quantization
        ArithmeticType
        TensorConversion
        Add
        Rounding
        Common
)

add_library(PointwiseFused PointwiseFused.c)
target_include_directories(PointwiseFused PUBLIC include)
target_link_libraries(PointwiseFused PRIVATE
        Tensor
        Common
        m
)
# Rounding order is the torch-parity contract. The in-source pragma is NOT
# enough: GCC ignores `#pragma STDC FP_CONTRACT` and its default
# (-ffp-contract=fast) fuses a+t*c into vfma on ARM (verified on
# arm-none-eabi-gcc 10.3 / Cortex-M4F) — invisible on x86-64 CI, which
# cannot contract without -mfma. The flag binds on both GCC and Clang.
target_compile_options(PointwiseFused PRIVATE -ffp-contract=off)
# Latent instruction counters (ODT_MEM_PROFILE pattern, src/userApi/CMakeLists.txt:43):
# the define must sit on the LIBRARY (a test-target define is inert); PUBLIC so
# guarded test TUs inherit it through the link edge.
target_compile_definitions(PointwiseFused PUBLIC
        $<$<BOOL:${ODT_TRACK_INSTRUCTIONS}>:TRACK_INSTRUCTIONS>)
