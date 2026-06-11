#include "Arithmetic.h"
#include "Matmul.h"
#include "Tensor.h"
#include "unity.h"

#include <DTypes.h>
#include <TensorConversion.h>

void testMatmulInt32() {
    size_t numberOfElements = 6;

    /*
    1, 2, 3,
    4, 5, 6
    */
    int32_t aData[] = {1, 2, 3, 4, 5, 6};

    size_t aNumberOfDims = 2;
    size_t aDims[] = {2, 3};
    size_t aOrderOfDims[] = {0, 1};
    shape_t aShape = {.dimensions = aDims,
                      .orderOfDimensions = aOrderOfDims,
                      .numberOfDimensions = aNumberOfDims};

    quantization_t aQ = {.type = INT32};

    tensor_t aTensor = {
        .data = (uint8_t *)aData,
        .shape = &aShape,
        .quantization = &aQ,
        .sparsity = NULL,
    };

    /*
    1, 4,
    2, 5,
    3, 6
    */
    int32_t bData[] = {1, 4, 2, 5, 3, 6};
    size_t bNumberOfDims = 2;
    size_t bDims[] = {3, 2};
    size_t bOrderOfDims[] = {0, 1};
    shape_t bShape = {.dimensions = bDims,
                      .orderOfDimensions = bOrderOfDims,
                      .numberOfDimensions = bNumberOfDims};

    quantization_t bQ = {.type = INT32};

    tensor_t bTensor = {
        .data = (uint8_t *)bData,
        .shape = &bShape,
        .quantization = &bQ,
        .sparsity = NULL,
    };

    int32_t outputData[] = {0, 0, 0, 0};
    size_t outputNumberOfDims = 2;
    size_t outputDims[] = {2, 2};
    size_t outputOrderOfDims[] = {0, 1};
    shape_t outputShape = {.dimensions = outputDims,
                           .orderOfDimensions = outputOrderOfDims,
                           .numberOfDimensions = outputNumberOfDims};

    quantization_t outputQ = {.type = INT32};

    tensor_t outputTensor = {
        .data = (uint8_t *)outputData,
        .shape = &outputShape,
        .quantization = &outputQ,
        .sparsity = NULL,
    };

    matmulInt32Tensors(&aTensor, &bTensor, &outputTensor);

    int32_t expected[] = {14, 32, 32, 77};

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, outputTensor.data, 4);
}

void testMatmulInt32WithVector() {
    /*
    1, 2, 3,
    4, 5, 6
    */
    int32_t aData[] = {1, 2, 3, 4, 5, 6};
    size_t aNumberOfDims = 2;
    size_t aDims[] = {2, 3};
    size_t aOrderOfDims[] = {0, 1};
    shape_t aShape = {.dimensions = aDims,
                      .orderOfDimensions = aOrderOfDims,
                      .numberOfDimensions = aNumberOfDims};

    quantization_t aQ = {.type = INT32};

    tensor_t aTensor = {
        .data = (uint8_t *)aData,
        .shape = &aShape,
        .quantization = &aQ,
        .sparsity = NULL,
    };

    /*
    1,
    2,
    3
    */
    int32_t bData[] = {1, 2, 3};
    size_t bNumberOfDims = 1;
    size_t bDims[] = {3};
    size_t bOrderOfDims[] = {0};
    shape_t bShape = {.dimensions = bDims,
                      .orderOfDimensions = bOrderOfDims,
                      .numberOfDimensions = bNumberOfDims};

    quantization_t bQ = {.type = INT32};

    tensor_t bTensor = {
        .data = (uint8_t *)bData,
        .shape = &bShape,
        .quantization = &bQ,
        .sparsity = NULL,
    };

    int32_t outputData[] = {0, 0};
    size_t outputNumberOfDims = 1;
    size_t outputDims[] = {2};
    size_t outputOrderOfDims[] = {0};
    shape_t outputShape = {.dimensions = outputDims,
                           .orderOfDimensions = outputOrderOfDims,
                           .numberOfDimensions = outputNumberOfDims};

    quantization_t outputQ = {.type = INT32};

    tensor_t outputTensor = {
        .data = (uint8_t *)outputData,
        .shape = &outputShape,
        .quantization = &outputQ,
        .sparsity = NULL,
    };

    matmulInt32Tensors(&aTensor, &bTensor, &outputTensor);

    int32_t expected[] = {14, 32};

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, outputTensor.data, 2);
}

void testMatmulFloatVectors() {
    /*
    1.1, 2.4, 3.9,
    */
    float aData[] = {1.1f, 2.4f, 3.9f};
    size_t aNumberOfDims = 1;
    size_t aDims[] = {3};
    size_t aOrderOfDims[] = {0};
    shape_t aShape = {.dimensions = aDims,
                      .orderOfDimensions = aOrderOfDims,
                      .numberOfDimensions = aNumberOfDims};

    quantization_t aQ = {.type = FLOAT32};

    tensor_t aTensor = {
        .data = (uint8_t *)aData,
        .shape = &aShape,
        .quantization = &aQ,
        .sparsity = NULL,
    };

    /*
    1.5,
    2.9,
    3.3
    */
    float bData[] = {1.5f, 2.9f, 3.3f};
    size_t bNumberOfDims = 1;
    size_t bDims[] = {3};
    size_t bOrderOfDims[] = {0};
    shape_t bShape = {.dimensions = bDims,
                      .orderOfDimensions = bOrderOfDims,
                      .numberOfDimensions = bNumberOfDims};

    quantization_t bQ = {.type = FLOAT32};

    tensor_t bTensor = {
        .data = (uint8_t *)bData,
        .shape = &bShape,
        .quantization = &bQ,
        .sparsity = NULL,
    };

    float outputData[] = {0};
    size_t outputNumberOfDims = 1;
    size_t outputDims[] = {1};
    size_t outputOrderOfDims[] = {0};
    shape_t outputShape = {.dimensions = outputDims,
                           .orderOfDimensions = outputOrderOfDims,
                           .numberOfDimensions = outputNumberOfDims};

    quantization_t outputQ = {.type = FLOAT32};

    tensor_t outputTensor = {
        .data = (uint8_t *)outputData,
        .shape = &outputShape,
        .quantization = &outputQ,
        .sparsity = NULL,
    };

    matmulFloat32Tensors(&aTensor, &bTensor, &outputTensor);

    float expected[] = {21.48f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, outputTensor.data, 1);
}

void testMatmulSymInt32Tensors() {

    tensor_t aTensor;
    int32_t aData[] = {1, 2, 3, 4, 5, 6};
    size_t aNumberOfDims = 2;
    size_t aDims[] = {2, 3};
    size_t aOrderOfDims[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, aNumberOfDims, aOrderOfDims);
    symInt32QConfig_t aSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &aSymInt32QC);
    aSymInt32QC.scale = 2.f;
    quantization_t aQ;
    initSymInt32Quantization(&aSymInt32QC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)aData, &aShape, &aQ, NULL);

    tensor_t bTensor;
    int32_t bData[] = {1, 4, 2, 5, 3, 6};
    size_t bNumberOfDims = 2;
    size_t bDims[] = {3, 2};
    size_t bOrderOfDims[] = {0, 1};
    shape_t bShape;
    setShape(&bShape, bDims, bNumberOfDims, bOrderOfDims);
    symInt32QConfig_t bSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &bSymInt32QC);
    quantization_t bQ;
    initSymInt32Quantization(&bSymInt32QC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)bData, &bShape, &bQ, NULL);

    tensor_t outputTensor;
    int32_t outputData[4];
    size_t outputNumberOfDims = 2;
    size_t outputDims[] = {2, 2};
    size_t outputOrderOfDims[] = {0, 1};
    shape_t outputShape;
    setShape(&outputShape, outputDims, outputNumberOfDims, outputOrderOfDims);
    symInt32QConfig_t outputSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &outputSymInt32QC);
    quantization_t outputQ;
    initSymInt32Quantization(&outputSymInt32QC, &outputQ);
    setTensorValues(&outputTensor, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    matmulSymInt32Tensors(&aTensor, &bTensor, &outputTensor);

    float expected[] = {28.f, 64.f, 64.f, 154.f};

    float actualData[4];
    quantization_t actualQ;
    initFloat32Quantization(&actualQ);

    tensor_t actual;
    setTensorValues(&actual, (uint8_t *)actualData, &outputShape, &actualQ, NULL);
    convertTensor(&outputTensor, &actual);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual.data, 4);
}

void testMatmulFloat32TensorsWithBiasBroadcastsOverRows() {
    /* a = [[1,2,3],[4,5,6]] (2x3), b = [[1,4],[2,5],[3,6]] (3x2). */
    float aData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    size_t aDims[] = {2, 3};
    size_t aOrder[] = {0, 1};
    shape_t aShape = {.dimensions = aDims, .orderOfDimensions = aOrder, .numberOfDimensions = 2};
    quantization_t aQ = {.type = FLOAT32};
    tensor_t aTensor = {
        .data = (uint8_t *)aData, .shape = &aShape, .quantization = &aQ, .sparsity = NULL};

    float bData[] = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    size_t bDims[] = {3, 2};
    size_t bOrder[] = {0, 1};
    shape_t bShape = {.dimensions = bDims, .orderOfDimensions = bOrder, .numberOfDimensions = 2};
    quantization_t bQ = {.type = FLOAT32};
    tensor_t bTensor = {
        .data = (uint8_t *)bData, .shape = &bShape, .quantization = &bQ, .sparsity = NULL};

    /* bias = [10, 20] (rank-1, length == output columns == 2). */
    float biasData[] = {10.f, 20.f};
    size_t biasDims[] = {2};
    size_t biasOrder[] = {0};
    shape_t biasShape = {
        .dimensions = biasDims, .orderOfDimensions = biasOrder, .numberOfDimensions = 1};
    quantization_t biasQ = {.type = FLOAT32};
    tensor_t biasTensor = {
        .data = (uint8_t *)biasData, .shape = &biasShape, .quantization = &biasQ, .sparsity = NULL};

    float outputData[] = {0, 0, 0, 0};
    size_t outputDims[] = {2, 2};
    size_t outputOrder[] = {0, 1};
    shape_t outputShape = {
        .dimensions = outputDims, .orderOfDimensions = outputOrder, .numberOfDimensions = 2};
    quantization_t outputQ = {.type = FLOAT32};
    tensor_t outputTensor = {.data = (uint8_t *)outputData,
                             .shape = &outputShape,
                             .quantization = &outputQ,
                             .sparsity = NULL};

    matmulFloat32TensorsWithBias(&aTensor, &bTensor, &outputTensor, &biasTensor);

    /* plain matmul = [[14,32],[32,77]]; + bias [10,20] per row = [[24,52],[42,97]]. */
    float expected[] = {24.f, 52.f, 42.f, 97.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, outputTensor.data, 4);
}

void testMatmulFloat32TensorsWithBiasNullEqualsPlain() {
    float aData[] = {1.1f, 2.4f, 3.9f};
    size_t aDims[] = {3};
    size_t aOrder[] = {0};
    shape_t aShape = {.dimensions = aDims, .orderOfDimensions = aOrder, .numberOfDimensions = 1};
    quantization_t aQ = {.type = FLOAT32};
    tensor_t aTensor = {
        .data = (uint8_t *)aData, .shape = &aShape, .quantization = &aQ, .sparsity = NULL};

    float bData[] = {1.5f, 2.9f, 3.3f};
    size_t bDims[] = {3};
    size_t bOrder[] = {0};
    shape_t bShape = {.dimensions = bDims, .orderOfDimensions = bOrder, .numberOfDimensions = 1};
    quantization_t bQ = {.type = FLOAT32};
    tensor_t bTensor = {
        .data = (uint8_t *)bData, .shape = &bShape, .quantization = &bQ, .sparsity = NULL};

    float outputData[] = {0};
    size_t outputDims[] = {1};
    size_t outputOrder[] = {0};
    shape_t outputShape = {
        .dimensions = outputDims, .orderOfDimensions = outputOrder, .numberOfDimensions = 1};
    quantization_t outputQ = {.type = FLOAT32};
    tensor_t outputTensor = {.data = (uint8_t *)outputData,
                             .shape = &outputShape,
                             .quantization = &outputQ,
                             .sparsity = NULL};

    matmulFloat32TensorsWithBias(&aTensor, &bTensor, &outputTensor, NULL);

    float expected[] = {21.48f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, outputTensor.data, 1);
}

void testMatmulSymInt32TensorsWithBiasRescalesBias() {
    tensor_t aTensor;
    int32_t aData[] = {1, 2, 3, 4, 5, 6};
    size_t aDims[] = {2, 3};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = 2.f;
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)aData, &aShape, &aQ, NULL);

    tensor_t bTensor;
    int32_t bData[] = {1, 4, 2, 5, 3, 6};
    size_t bDims[] = {3, 2};
    size_t bOrder[] = {0, 1};
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    symInt32QConfig_t bQC;
    initSymInt32QConfig(HALF_AWAY, &bQC);
    bQC.scale = 1.f;
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)bData, &bShape, &bQ, NULL);

    tensor_t biasTensor;
    int32_t biasData[] = {1, 2};
    size_t biasDims[] = {2};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    symInt32QConfig_t biasQC;
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    biasQC.scale = 4.f;
    quantization_t biasQ;
    initSymInt32Quantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)biasData, &biasShape, &biasQ, NULL);

    tensor_t outputTensor;
    int32_t outputData[4];
    size_t outputDims[] = {2, 2};
    size_t outputOrder[] = {0, 1};
    shape_t outputShape;
    setShape(&outputShape, outputDims, 2, outputOrder);
    symInt32QConfig_t outputQC;
    initSymInt32QConfig(HALF_AWAY, &outputQC);
    quantization_t outputQ;
    initSymInt32Quantization(&outputQC, &outputQ);
    setTensorValues(&outputTensor, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    matmulSymInt32TensorsWithBias(&aTensor, &bTensor, &outputTensor, &biasTensor);

    /* a_real (scale 2) @ b = [[28,64],[64,154]]; + bias_real [4,8] = [[32,72],[68,162]]. */
    float expected[] = {32.f, 72.f, 68.f, 162.f};
    float actualData[4];
    quantization_t actualQ;
    initFloat32Quantization(&actualQ);
    tensor_t actual;
    setTensorValues(&actual, (uint8_t *)actualData, &outputShape, &actualQ, NULL);
    convertTensor(&outputTensor, &actual);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual.data, 4);
}

void setUp() {}
void tearDown() {}

int main(void) {

    UNITY_BEGIN();
    RUN_TEST(testMatmulInt32);
    RUN_TEST(testMatmulInt32WithVector);
    RUN_TEST(testMatmulFloatVectors);
    RUN_TEST(testMatmulSymInt32Tensors);
    RUN_TEST(testMatmulFloat32TensorsWithBiasBroadcastsOverRows);
    RUN_TEST(testMatmulFloat32TensorsWithBiasNullEqualsPlain);
    RUN_TEST(testMatmulSymInt32TensorsWithBiasRescalesBias);

    return UNITY_END();
}
