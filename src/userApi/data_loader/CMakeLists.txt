add_library(DataLoaderApi DataLoaderApi.c)
target_include_directories(DataLoaderApi PUBLIC include)
target_link_libraries(DataLoaderApi PRIVATE
        DataLoader
        Common
        StorageApi
        Tensor
        Rounding
)

add_library(NPYLoaderApi NPYLoaderApi.c)
target_include_directories(NPYLoaderApi PUBLIC include)
target_link_libraries(NPYLoaderApi PRIVATE
        NPYLoader
        DataLoader
        TensorApi
        Tensor
        Rounding
        Common
        StorageApi
        QuantizationApi
        Quantization
)