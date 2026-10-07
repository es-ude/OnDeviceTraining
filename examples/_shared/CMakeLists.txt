add_library(examples_shared STATIC npy_writer.c npy_dump_sink.c param_gate.c remat_select.c)
target_include_directories(examples_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Include propagation is PRIVATE repo-wide: list every library whose header a
# source here includes (Layer.h, Tensor.h, TensorConversion.h, TensorApi.h,
# Quantization.h, QuantizationApi.h, Rounding.h, Common.h, BfpKernelSupport.h;
# remat_select.c: RematScheduler.h, CalculateGradsSequential.h and what it
# pulls in: TrainingLoopApi.h, InferenceApi.h, DataLoader.h, LossFunction.h,
# Optimizer.h).
target_link_libraries(examples_shared PRIVATE
        RematScheduler
        CalculateGradsSequential
        TrainingLoopApi
        InferenceApi
        DataLoader
        LossFunction
        Optimizer
        Layer
        Tensor
        TensorConversion
        TensorApi
        Quantization
        QuantizationApi
        Rounding
        Common
        ArithmeticType
)
