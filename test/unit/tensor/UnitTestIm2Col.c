#include "Im2Col.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#include <string.h>

void setUp(){}
void tearDown(){}


void testSimpleIm2Col1dWithStrideAndDilation() {
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 7;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *inputTensor = initTensor(inputShape, quantizationInitFloat(), NULL);
    float inputData[] = {
        1, 2, 3, 4, 5, 6, 7
    };

    memcpy(inputTensor->data, inputData, sizeof(float)*7);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 6;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *outputTensor = initTensor(outputShape, quantizationInitFloat(), NULL);

    windowGeometry1d_t geometry = {
        .dilation = 2,
        .stride = 2,
        .inputLength = 7,
        .kernelSize = 2,
        .outputLength = 3,
        .padLeft = 0,
        .padRight = 0
    };

    im2col1dFloat(inputTensor, &geometry, outputTensor);

    float *actual = (float*)outputTensor->data;
    float expected[] = {
        1, 3, 5,
        3, 5, 7
    };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, 6);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSimpleIm2Col1dWithStrideAndDilation);
    UNITY_END();
}