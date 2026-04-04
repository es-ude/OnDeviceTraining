add_library(Add Add.c)
target_include_directories(Add PUBLIC include)
target_link_libraries(Add PRIVATE
        Tensor
        DTypes
        Arithmetic
        TensorConversion
        Common
)

add_library(Arithmetic Arithmetic.c)
target_include_directories(Arithmetic PUBLIC include)
target_link_libraries(Arithmetic PRIVATE
        Tensor
        DTypes
        Add
        Div
        Matmul
        Mul
        Square
        Sub
        Common
)

add_library(Comparison Comparison.c)
target_include_directories(Comparison PUBLIC include)
target_link_libraries(Comparison PRIVATE
        Tensor
        DTypes
        Common
)

add_library(Distributions Distributions.c)
target_include_directories(Distributions PUBLIC include)
target_link_libraries(Distributions PRIVATE
        Common
        RNG
        m
)

add_library(Div Div.c)
target_include_directories(Div PUBLIC include)
target_link_libraries(Div PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(Log Log.c)
target_include_directories(Log PUBLIC include)
target_link_libraries(Log PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(Matmul Matmul.c)
target_include_directories(Matmul PUBLIC include)
target_link_libraries(Matmul PRIVATE
        Common
        Tensor
        DTypes
        Mul
        Common
)

add_library(Mul Mul.c)
target_include_directories(Mul PUBLIC include)
target_link_libraries(Mul PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(MinMax MinMax.c)
target_include_directories(MinMax PUBLIC include)
target_link_libraries(MinMax PRIVATE
        Tensor
        DTypes
        Common
)

add_library(Rounding Rounding.c)
target_include_directories(Rounding PUBLIC include)
target_link_libraries(Rounding PRIVATE
        m
        RNG
        Common
)


add_library(Square Square.c)
target_include_directories(Square PUBLIC include)
target_link_libraries(Square PRIVATE
        Tensor
        DTypes
        Common
)

add_library(Sub Sub.c)
target_include_directories(Sub PUBLIC include)
target_link_libraries(Sub PRIVATE
        Tensor
        DTypes
        Arithmetic
        Common
)

add_library(Sum Sum.c)
target_include_directories(Sum PUBLIC include)
target_link_libraries(Sum PRIVATE
        Add
        Tensor
        Rounding
        Common
)



