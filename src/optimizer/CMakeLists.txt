add_library(Optimizer Optimizer.c)
target_include_directories(Optimizer PUBLIC include)
target_link_libraries(Optimizer PRIVATE
        Quantization
        Rounding
        Layer
        Conv1d
        Conv1dTransposed
        Linear
        Tensor
        Common
        # The [ADAM_W] vtable row references adamWStep/adamWGetLr/adamWSetLr.
        # Linked HERE (unlike Sgd, historically consumer-listed) so the new
        # row does not force every existing Optimizer consumer to grow its
        # link list; CMake propagates static-lib PRIVATE deps to final links.
        AdamW
)

add_library(Sgd Sgd.c)
target_include_directories(Sgd PUBLIC include)
target_link_libraries(Sgd PRIVATE
        DTypes
        Tensor
        Arithmetic
        ArithmeticType
        ExecuteOp
        Layer
        Common
)

add_library(AdamW AdamW.c)
target_include_directories(AdamW PUBLIC include)
target_link_libraries(AdamW PRIVATE
        DTypes
        Tensor
        Mul
        PointwiseFused
        ArithmeticType
        ExecuteOp
        Layer
        Common
        m
)
# The double scalar composition (1-beta, 1-lr*wd, -(lr/bc1)) is the
# torch-parity contract. GCC ignores `#pragma STDC FP_CONTRACT` and its
# default (-ffp-contract=fast) would contract 1.0 - lr*wd into fma on ARM
# (PointwiseFused precedent, verified arm-none-eabi-gcc 10.3 / Cortex-M4F).
target_compile_options(AdamW PRIVATE -ffp-contract=off)

add_library(LrScheduler LrScheduler.c)
target_include_directories(LrScheduler PUBLIC include)
target_link_libraries(LrScheduler PRIVATE
        Optimizer
        # #327: PRIVATE-everywhere convention (docs/conventions) - Optimizer.h
        # transitively includes Layer.h/Tensor.h/Quantization.h/Rounding.h;
        # Optimizer's own links to those are PRIVATE, so consumers must list
        # the header-owning targets too (same precedent as Sgd's own deps).
        Layer
        Tensor
        Arithmetic
        Common
        m
)
