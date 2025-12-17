add_library(InferenceAPI InferenceAPI.c)
target_include_directories(InferenceAPI PUBLIC include)
target_link_libraries(InferenceAPI PRIVATE
        Tensor
        Rounding
        Layer
        LossFunction
)

add_library(LinearAPI LinearAPI.c)
target_include_directories(LinearAPI PUBLIC include)
target_link_libraries(LinearAPI PRIVATE
        Tensor
        Rounding
        Layer
        Linear
)

add_library(OptimizerAPI OptimizerAPI.c)
target_include_directories(OptimizerAPI PUBLIC include)
target_link_libraries(OptimizerAPI PRIVATE
        Sgd
        Optimizer
        Tensor
        Rounding
        Layer
)

add_library(QuantizationAPI QuantizationAPI.c)
target_include_directories(QuantizationAPI PUBLIC include)
target_link_libraries(QuantizationAPI PRIVATE
        Quantization
        Rounding
)

add_library(ReluAPI ReluAPI.c)
target_include_directories(ReluAPI PUBLIC include)
target_link_libraries(ReluAPI PRIVATE
        Tensor
        Rounding
        Layer
        Relu
)

add_library(SgdAPI SgdAPI.c)
target_include_directories(SgdAPI PUBLIC include)
target_link_libraries(SgdAPI PRIVATE
        Tensor
        Rounding
        Layer
        Sgd
        Optimizer
)

add_library(SoftmaxAPI SoftmaxAPI.c)
target_include_directories(SoftmaxAPI PUBLIC include)
target_link_libraries(SoftmaxAPI PRIVATE
        Layer
        Tensor
        Rounding
        StorageAPI
)

add_library(StorageAPI StorageAPI.c)
target_include_directories(StorageAPI PUBLIC include)
target_link_libraries(StorageAPI PRIVATE
        Tensor
        Rounding
)

add_library(TensorAPI TensorAPI.c)
target_include_directories(TensorAPI PUBLIC include)
target_link_libraries(TensorAPI PRIVATE
        Tensor
        Rounding
        StorageAPI
)

add_library(TrainingAPI TrainingAPI.c)
target_include_directories(TrainingAPI PUBLIC include)
target_link_libraries(TrainingAPI PRIVATE
        Layer
        Tensor
        Rounding
        MSE
        Optimizer
)



