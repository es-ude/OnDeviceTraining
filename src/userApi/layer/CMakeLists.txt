add_library(LinearApi LinearApi.c)
target_include_directories(LinearApi PUBLIC include)
target_link_libraries(LinearApi PRIVATE
        Tensor
        TensorApi
        Rounding
        Layer
        Linear
        Common
        StorageApi
)

add_library(ReluApi ReluApi.c)
target_include_directories(ReluApi PUBLIC include)
target_link_libraries(ReluApi PRIVATE
        Tensor
        Rounding
        Layer
        Relu
        Common
        StorageApi
)

add_library(SoftmaxApi SoftmaxApi.c)
target_include_directories(SoftmaxApi PUBLIC include)
target_link_libraries(SoftmaxApi PRIVATE
        Layer
        Tensor
        Rounding
        Common
        StorageApi
)