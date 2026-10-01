add_library(StackGather StackGather.c)
target_include_directories(StackGather PUBLIC include)
target_link_libraries(StackGather PRIVATE
        Layer
        Tensor
        TensorApi
        StorageApi
        Common
)
