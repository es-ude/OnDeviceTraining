#include "Add.h"
#include "Arithmetic.h"
#include "OdtAssert.h"
#include "Quantization.h"
#include "Tensor.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

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

void testCalcIndexByRawIndex() {
    size_t numberOfDimensions = 3;
    size_t dimensions[] = {2, 3, 4};
    size_t expected[] = {1, 2, 3};

    size_t actual[3];
    calcIndicesByRawIndex(numberOfDimensions, dimensions, 23, actual);

    ODT_ASSERT_EQUAL_size_t_ARRAY(expected, actual, numberOfDimensions);
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

void testDoDimensionsMatchDifferentRankReturnsFalse() {
    size_t aDims[] = {2, 3};
    size_t aOrder[] = {0, 1};
    shape_t aShape = {.dimensions = aDims, .orderOfDimensions = aOrder, .numberOfDimensions = 2};
    tensor_t a = {.shape = &aShape};

    size_t bDims[] = {6};
    size_t bOrder[] = {0};
    shape_t bShape = {.dimensions = bDims, .orderOfDimensions = bOrder, .numberOfDimensions = 1};
    tensor_t b = {.shape = &bShape};

    TEST_ASSERT_FALSE(doDimensionsMatch(&a, &b));
}

/* Only a non-involution order tells the inverse permutation from the forward one:
 * for order [1,2,0] over physical dims [2,3,4], logical [1,1,2] is physical [1,2,1]
 * = offset 21 (the forward form gives 29, out of bounds). */
void testCalcElementIndexByIndicesThreeCycle() {
    size_t dims[] = {2, 3, 4};
    size_t order[] = {1, 2, 0};
    size_t indices[] = {1, 1, 2};
    TEST_ASSERT_EQUAL_size_t(21, calcElementIndexByIndices(3, dims, indices, order));
}

/* a is a real [3,2]->[2,3] transpose, b identity-order with the same logical shape;
 * the result must land in a's physical slots. */
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
    RUN_TEST(testCalcIndexByRawIndex);
    RUN_TEST(testDoDimensionsMatch_SameShape_ReturnsTrue);
    RUN_TEST(testDoDimensionsMatch_DifferentDims_ReturnsFalse);
    RUN_TEST(testDoDimensionsMatchDifferentRankReturnsFalse);
    RUN_TEST(testCalcElementIndexByIndicesThreeCycle);
    RUN_TEST(testFloat32PointWiseArithmeticInplaceTransposed);
    return UNITY_END();
}
