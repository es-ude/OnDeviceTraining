add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        Sgd
        MORE_LIBS
        CommonLayerLibs
        OptimizerApi
        SgdApi
        TensorApi
        QuantizationApi
        StorageApi
        LinearApi
        ReluApi
        LayerQuant
        ArithmeticType
        RNG
        odt_test_death
)

set(GEN_LR_SCHEDULER_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_lr_scheduler.h)
add_custom_command(
        OUTPUT ${GEN_LR_SCHEDULER_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_lr_scheduler.py
                --out ${GEN_LR_SCHEDULER_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_lr_scheduler.py
        COMMENT "Generating expected_lr_scheduler.h via PyTorch"
        VERBATIM
)
add_custom_target(generate_expected_lr_scheduler
        DEPENDS ${GEN_LR_SCHEDULER_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        LrScheduler
        MORE_LIBS
        Optimizer
        Sgd
        ArithmeticType
        # #327: PRIVATE-everywhere convention - Optimizer.h/ArithmeticType.h
        # transitively need Layer.h/Tensor.h (Quantization.h/Rounding.h ride
        # along via their shared include dirs), so the test binary must list
        # the header-owning targets explicitly too.
        Layer
        Tensor
        Common
        odt_test_death
)
add_dependencies(UnitTestLrScheduler generate_expected_lr_scheduler)
target_include_directories(UnitTestLrScheduler PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

set(GEN_ADAMW_HEADER ${CMAKE_CURRENT_BINARY_DIR}/expected_adamw.h)
add_custom_command(
        OUTPUT ${GEN_ADAMW_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_adamw.py
                --out ${GEN_ADAMW_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_adamw.py
        COMMENT "Generating expected_adamw.h (documented-order replication + torch sanity)"
        VERBATIM
)
add_custom_target(generate_expected_adamw DEPENDS ${GEN_ADAMW_HEADER})

add_elastic_ai_unit_test(
        LIB_UNDER_TEST
        AdamW
        MORE_LIBS
        Optimizer
        Sgd
        LrScheduler
        ArithmeticType
        TensorApi
        QuantizationApi
        StorageApi
        Layer
        Tensor
        Common
        m
        odt_test_death
        AdamWApi
        OptimizerApi
        ExecuteOp
)
add_dependencies(UnitTestAdamW generate_expected_adamw)
target_include_directories(UnitTestAdamW PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
