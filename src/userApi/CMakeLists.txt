add_subdirectory(continual_learning)
add_subdirectory(data_loader)
add_subdirectory(layer)
add_subdirectory(optimizer)
add_subdirectory(tensor)
add_subdirectory(training_loop)

add_library(MemProfile MemProfile.c)
target_include_directories(MemProfile PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(MemProfile PRIVATE Threads::Threads)

add_library(LayerCommon LayerCommon.c)
target_include_directories(LayerCommon PUBLIC include)
target_link_libraries(LayerCommon PRIVATE
        Common
        Distributions
        Tensor
        TensorApi
)

add_library(InferenceApi InferenceApi.c)
target_include_directories(InferenceApi PUBLIC include)
target_link_libraries(InferenceApi PRIVATE
        TensorApi
        Tensor
        Rounding
        Layer
        Conv1d
        Conv1dTransposed
        MaxPool1d
        AvgPool1d
        AdaptiveAvgPool1d
        LayerNorm
        QuantizationLayer
        LayerConfigAccess
        LossFunction
        Common
        DataLoader
)

add_library(StorageApi StorageApi.c)
target_include_directories(StorageApi PUBLIC include)
target_compile_definitions(StorageApi PUBLIC $<$<BOOL:${ODT_MEM_PROFILE}>:ODT_MEM_PROFILE>)
target_link_libraries(StorageApi PRIVATE
        Tensor
        Rounding
        Common
)

add_library(LayerQuant LayerQuant.c)
target_include_directories(LayerQuant PUBLIC include)
target_link_libraries(LayerQuant PRIVATE
        ArithmeticType
        Common
        ExecuteOp
        Quantization
        Rounding
        StorageApi
)

add_library(LayerWeightsApi LayerWeightsApi.c)
target_include_directories(LayerWeightsApi PUBLIC include)
target_link_libraries(LayerWeightsApi PRIVATE
        Common
        Conv1d
        Conv1dTransposed
        Layer
        LayerNorm
        Linear
        Rounding
        Tensor
        TensorApi
)

add_library(StateDictApi StateDictApi.c)
target_include_directories(StateDictApi PUBLIC include)
target_link_libraries(StateDictApi PRIVATE
        Common
        Layer
        LayerWeightsApi
        StorageApi
        Tensor
        Rounding
        m
)


add_library(LayerConfigAccess LayerConfigAccess.c)
target_include_directories(LayerConfigAccess PUBLIC include)
target_link_libraries(LayerConfigAccess PRIVATE
        ArithmeticType
        Common
        CommonLayerLibs
)

add_library(ModelValidationApi ModelValidationApi.c)
target_include_directories(ModelValidationApi PUBLIC include)
target_link_libraries(ModelValidationApi PRIVATE
        Common
        Layer
        Quantization
        Rounding
        Tensor
)
