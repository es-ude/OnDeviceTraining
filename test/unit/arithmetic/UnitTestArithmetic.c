#include "Add.h"
#include "Arithmetic.h"
#include "DTypes.h"
#include "Mul.h"
#include "OdtAssert.h"
#include "Quantization.h"
#include "Tensor.h"
#include "unity.h"

#include <stdlib.h>
#include <string.h>

void testOrderDims() {
    size_t dimensions[] = {2, 3, 4};
    size_t orderOfDimensions[] = {1, 0, 2};
    size_t numberOfDims = 3;

    shape_t shape = {.dimensions = dimensions,
                     .orderOfDimensions = orderOfDimensions,
                     .numberOfDimensions = numberOfDims};

    tensor_t tensor = {.shape = &shape};

    size_t expected[] = {3, 2, 4};

    size_t actual[numberOfDims];
    orderDims(&tensor, actual);

    ODT_ASSERT_EQUAL_size_t_ARRAY(expected, actual, numberOfDims);
}

/*23 = [2, 1, 3]

reorderedElementIndex = [1, 2, 3]
dims = 2, 3, 4

size_t index = indices[numDims - 1];               // 3
size_t offset = dims[numDims - 1];                 // 4
for (i = numDims - 2; i >= 0; i--){
    index += indices[i] * offset                   3 + 2 * 4 = 11 | 11 + 1 * 12 = 23
    offset *= dims[i]                              4 * 3 = 12     | 12 * 2 = 24*/
void testCalcTensorIndex() {
    size_t numberOfDimensions = 3;
    size_t dimensions[] = {2, 3, 4};
    size_t indices[] = {1, 2, 3};

    size_t actual = calcTensorIndexByIndices(numberOfDimensions, dimensions, indices);
    size_t expected = 23;
    TEST_ASSERT_EQUAL_size_t(expected, actual);
}

void testCalcIndexByRawIndex() {
    size_t numberOfDimensions = 3;
    size_t dimensions[] = {2, 3, 4};
    size_t expected[] = {1, 2, 3};

    size_t actual[3];
    calcIndicesByRawIndex(numberOfDimensions, dimensions, 23, actual);

    ODT_ASSERT_EQUAL_size_t_ARRAY(expected, actual, numberOfDimensions);
}

void testInt32PointWiseArithmetic() {
    size_t bytesPerElement = sizeof(int32_t);
    size_t numberOfElements = 8;

    /*
    [ [ [-1, 2, 3, 4], [5, 6, -7, 8] ] ]


// order: 4, 2, 1

[ [ [-1], [5] ],
  [ [2],  [6] ],
  [ [3], [-7] ],
  [ [4],  [8] ] ]
    */
    int32_t aData[] = {-1, 2, 3, 4, 5, 6, -7, 8};

    size_t aDims[] = {1, 2, 4};
    size_t aOrderDims[] = {0, 1, 2};
    size_t aNumberOfDims = 3;
    shape_t aShape = {
        .dimensions = aDims, .orderOfDimensions = aOrderDims, .numberOfDimensions = aNumberOfDims};

    quantization_t aQuantization;
    initInt32Quantization(&aQuantization);

    tensor_t aTensor;
    setTensorValues(&aTensor, (uint8_t *)aData, &aShape, &aQuantization, NULL);

    int32_t bData[] = {-1, 2, 3, 4, 5, 6, -7, 8};

    size_t bNumberOfDims = 3;
    size_t bDims[] = {2, 1, 4};
    size_t bOrderDims[] = {1, 0, 2};
    shape_t bShape = {
        .dimensions = bDims, .orderOfDimensions = bOrderDims, .numberOfDimensions = bNumberOfDims};

    quantization_t bQuantization;
    initInt32Quantization(&bQuantization);

    tensor_t bTensor;
    setTensorValues(&bTensor, (uint8_t *)bData, &bShape, &bQuantization, NULL);

    uint32_t outputData[numberOfElements];

    size_t outputNumberOfDims = 3;
    size_t outputDims[] = {4, 2, 1};
    size_t outputOrderDims[] = {2, 1, 0};
    shape_t outputShape = {.dimensions = outputDims,
                           .orderOfDimensions = outputOrderDims,
                           .numberOfDimensions = outputNumberOfDims};

    quantization_t outputQuantization;
    initInt32Quantization(&outputQuantization);

    tensor_t outputTensor;
    setTensorValues(&outputTensor, (uint8_t *)outputData, &outputShape, &outputQuantization, NULL);

    int32_t expectedValues[] = {-2, 4, 6, 8, 10, 12, -14, 16};

    int32PointWiseArithmetic(&aTensor, &bTensor, addInt32s, &outputTensor);

    int32_t actual[numberOfElements];
    readBytesAsInt32Array(numberOfElements, outputTensor.data, actual);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedValues, actual, numberOfElements);
}

void testFloat32ElementWithTensorArithmetic() {
    float x = 2.f;

    quantization_t aQ;
    initFloat32Quantization(&aQ);
    float aData[] = {1.f, 2.f, 3.f, 4.f};
    size_t aDims[] = {4};
    size_t aNumberOfDims = 1;
    size_t aOrderOfDims[] = {0};
    shape_t aShape = {.dimensions = aDims,
                      .orderOfDimensions = aOrderOfDims,
                      .numberOfDimensions = aNumberOfDims};

    tensor_t aTensor;
    setTensorValues(&aTensor, (uint8_t *)aData, &aShape, &aQ, NULL);

    floatElementWithTensorArithmeticInplace(&aTensor, x, mulFloat32s);

    float expected[] = {2.f, 4.f, 6.f, 8.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, aData, 4);
}

void testDoDimensionsMatch_SameShape_ReturnsTrue() {
    size_t aDims[] = {2, 3};
    size_t aOrder[] = {0, 1};
    shape_t aShape = {.dimensions = aDims, .orderOfDimensions = aOrder, .numberOfDimensions = 2};
    tensor_t a = {.shape = &aShape};

    size_t bDims[] = {2, 3};
    size_t bOrder[] = {0, 1};
    shape_t bShape = {.dimensions = bDims, .orderOfDimensions = bOrder, .numberOfDimensions = 2};
    tensor_t b = {.shape = &bShape};

    TEST_ASSERT_TRUE(doDimensionsMatch(&a, &b));
}

void testDoDimensionsMatch_DifferentDims_ReturnsFalse() {
    size_t aDims[] = {2, 3};
    size_t aOrder[] = {0, 1};
    shape_t aShape = {.dimensions = aDims, .orderOfDimensions = aOrder, .numberOfDimensions = 2};
    tensor_t a = {.shape = &aShape};

    size_t bDims[] = {2, 4};
    size_t bOrder[] = {0, 1};
    shape_t bShape = {.dimensions = bDims, .orderOfDimensions = bOrder, .numberOfDimensions = 2};
    tensor_t b = {.shape = &bShape};

    TEST_ASSERT_FALSE(doDimensionsMatch(&a, &b));
}

// NOTE: doDimensionsMatch now calls exit(1) on rank mismatch — cannot test with Unity.
// The fix is verified by: different-rank inputs no longer silently read out of bounds.

void setUp() {}
void tearDown() {}

/* #344 defect 2: calcElementIndexByIndices must INVERT orderOfDimensions, not
 * apply it forward — the two agree only for involutions (identity / single
 * swaps), which is every permutation the live Matmul/Reduce/LayerNorm callers
 * see today. For the 3-cycle order [1,2,0] over physical dims [2,3,4], logical
 * index [1,1,2] maps to storage offset 21 (physical multi-index P[p]=L[order[p]]
 * = [1,2,1] -> 1*12 + 2*4 + 1). Pre-fix it computed 29 — out of bounds in the
 * 24-element buffer. */
void testCalcElementIndexByIndicesThreeCycle() {
    size_t dims[] = {2, 3, 4};
    size_t order[] = {1, 2, 0};
    size_t indices[] = {1, 1, 2};
    TEST_ASSERT_EQUAL_size_t(21, calcElementIndexByIndices(3, dims, indices, order));
}

/* #344 defect 1 + #339: an in-place pointwise op on a transposed (non-identity
 * order) FIRST operand must decode the raw index against the ORDERED (logical)
 * dims and write the result to the physical slot it read (aElementIndex) — not
 * decode against physical dims and write flat. Here a is a genuine [3,2]->[2,3]
 * transpose (both swapped axes > 1, so the size-1 masking in the existing test
 * cannot hide the defect); b is identity-order with the same logical shape. */
void testFloat32PointWiseArithmeticInplaceTransposed() {
    float aData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    size_t aDims[] = {3, 2};
    size_t aOrder[] = {0, 1};
    shape_t aShape = {.dimensions = aDims, .orderOfDimensions = aOrder, .numberOfDimensions = 2};
    quantization_t aQ;
    initFloat32Quantization(&aQ);
    tensor_t aTensor;
    setTensorValues(&aTensor, (uint8_t *)aData, &aShape, &aQ, NULL);
    transposeTensor(&aTensor, 0, 1); /* physical [3,2] -> logical [2,3], order [1,0] */

    float bData[] = {10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
    size_t bDims[] = {2, 3};
    size_t bOrder[] = {0, 1};
    shape_t bShape = {.dimensions = bDims, .orderOfDimensions = bOrder, .numberOfDimensions = 2};
    quantization_t bQ;
    initFloat32Quantization(&bQ);
    tensor_t bTensor;
    setTensorValues(&bTensor, (uint8_t *)bData, &bShape, &bQ, NULL);

    addFloat32TensorsInplace(&aTensor, &bTensor);

    /* a logical [[1,3,5],[2,4,6]] + b logical [[10,20,30],[40,50,60]] =
     * [[11,23,35],[42,54,66]] written back to a's physical slots (offset l1*2+l0). */
    float expected[] = {11.f, 42.f, 23.f, 54.f, 35.f, 66.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, aData, 6);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testOrderDims);
    RUN_TEST(testCalcTensorIndex);
    RUN_TEST(testCalcIndexByRawIndex);
    RUN_TEST(testInt32PointWiseArithmetic);
    RUN_TEST(testCalcElementIndexByIndicesThreeCycle);
    RUN_TEST(testFloat32PointWiseArithmeticInplaceTransposed);
    RUN_TEST(testFloat32ElementWithTensorArithmetic);
    RUN_TEST(testDoDimensionsMatch_SameShape_ReturnsTrue);
    RUN_TEST(testDoDimensionsMatch_DifferentDims_ReturnsFalse);
    // testDoDimensionsMatch_DifferentRank — now exit(1)s, verified by code review

    return UNITY_END();
}
