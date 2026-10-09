add_library(RNG RNG.c)
target_include_directories(RNG PUBLIC include)
target_link_libraries(RNG PRIVATE Common)

add_library(Bernoulli Bernoulli.c)
target_include_directories(Bernoulli PUBLIC include)
target_link_libraries(Bernoulli PRIVATE
        RNG
        Tensor
        Rounding
)
