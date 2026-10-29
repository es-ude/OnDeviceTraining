# Differential numerics harness (see README.md): one C driver that runs a
# single op through the real libraries and dumps integer codes + scales for
# the Python references in ref/. Not a ctest target; built only when the
# root option ODT_BUILD_DIFFERENTIAL is ON (preset `differential`).
#
# Include propagation is PRIVATE repo-wide: every header odt_diffdrive.c
# includes is owned by a target listed here (layer configs, kernels, funnel,
# tensor/quantization core, npy I/O, the examples/_shared npy writer).
add_executable(odt_diffdrive odt_diffdrive.c)
target_link_libraries(odt_diffdrive PRIVATE
        CommonLayerLibs
        Conv1d
        Conv1dKernel
        ConvTranspose1dKernel
        Kernel
        SlidingWindow1d
        ExecuteOp
        ArithmeticType
        LayerExec
        RNG
        Tensor
        TensorApi
        Quantization
        QuantizationApi
        TensorConversion
        Rounding
        StorageApi
        Common
        DTypes
        NPYLoaderApi
        NPYLoader
        DataLoader
        examples_shared
        m
)
