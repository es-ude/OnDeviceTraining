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
        OdtHook
        # The vtable rows reference sgdStepM/sgdGetLr/sgdSetLr and
        # adamWStep/adamWGetLr/adamWSetLr. Both libraries are linked HERE so no
        # Optimizer consumer has to list them after Optimizer itself: CMake
        # propagates static-lib PRIVATE deps to final links, after the
        # library, which is the order GNU ld needs.
        Sgd
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

add_library(BsScheduler BsScheduler.c)
target_include_directories(BsScheduler PUBLIC include)
target_link_libraries(BsScheduler PRIVATE
        Optimizer
        DataLoader
        # #327 PRIVATE-everywhere convention (docs/conventions): Optimizer.h and
        # DataLoader.h transitively include Layer.h/Tensor.h/Dataset.h/
        # Quantization.h/Rounding.h; their owners' links are PRIVATE, so this
        # consumer lists the header-owning targets itself (LrScheduler precedent).
        Layer
        Tensor
        Arithmetic
        Common
        m
)
