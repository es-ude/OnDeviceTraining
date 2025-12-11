add_library(Sgd
        Sgd.c
)
target_include_directories(Sgd PUBLIC include)
target_link_libraries(Sgd PRIVATE
        DTypes
        Tensor
        Arithmetic
        Layer
)