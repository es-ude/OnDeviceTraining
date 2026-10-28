add_library(DataLoader DataLoader.c)
target_include_directories(DataLoader PUBLIC include)
target_link_libraries(DataLoader PRIVATE
        Tensor
        Rounding
        Common
        RNG
)

add_library(NPYLoader NPYLoader.c)
target_include_directories(NPYLoader PUBLIC include)
target_link_libraries(NPYLoader PRIVATE
        Tensor
        Rounding
        StorageApi
        Common
)