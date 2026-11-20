add_library(StackGather StackGather.c)
target_include_directories(StackGather PUBLIC include)
target_link_libraries(StackGather PRIVATE
        LossType
        Layer
        Tensor
        TensorApi
        StorageApi
        Common
)
