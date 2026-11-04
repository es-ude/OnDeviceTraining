#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "Arithmetic.h"
#include "BfpKernelSupport.h"
#include "BfpRawFixture.h"
#include "DeathTest.h"
#include "Matmul.h"
#include "RNG.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"
#include "unity.h"

#include <DTypes.h>
#include <TensorConversion.h>

#include "expected_bfp_matmul.h"
#include "expected_group_matmul.h"

/* Grouped-weight ctx fixtures: the stored dtype's code domain fixes the
 * weight's operand range (SymGroupedWeight.h). */
static symGroupedWeight_t symGroupedWeightSym(const symQConfig_t *groups) {
    return (symGroupedWeight_t){.groups = groups, .domain = SYM_CODES_TWOS_COMPLEMENT};
}
static symGroupedWeight_t symGroupedWeightAsymImage(const symQConfig_t *groups) {
    return (symGroupedWeight_t){.groups = groups, .domain = SYM_CODES_ASYM_IMAGE};
}

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
    convertTensor(&outputTensor, &actual, NULL);

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

    matmulSymInt32TensorsWithBias(&aTensor, &bTensor, &outputTensor, &biasTensor, NULL);

    /* a_real (scale 2) @ b = [[28,64],[64,154]]; + bias_real [4,8] = [[32,72],[68,162]]. */
    float expected[] = {32.f, 72.f, 68.f, 162.f};
    float actualData[4];
    quantization_t actualQ;
    initFloat32Quantization(&actualQ);
    tensor_t actual;
    setTensorValues(&actual, (uint8_t *)actualData, &outputShape, &actualQ, NULL);
    convertTensor(&outputTensor, &actual, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual.data, 4);
}

/* The int12 operand contract is no longer a rejection: the accumulator guard
 * (SymAccumGuard.h) is exact for any declared width. The wide operand's
 * wrapping and fitting cases are pinned with the guard tests below. */
void testMatmulSymInt32WiderOperandRunsExact() {
    tensor_t a, b, out;
    int32_t aData[] = {1, 2, 3, 4, 5, 6};
    int32_t bData[] = {1, 4, 2, 5, 3, 6};
    int32_t outData[4];
    size_t aDims[] = {2, 3}, bDims[] = {3, 2}, oDims[] = {2, 2}, order[] = {0, 1};
    shape_t aS, bS, oS;
    setShape(&aS, aDims, 2, order);
    setShape(&bS, bDims, 2, order);
    setShape(&oS, oDims, 2, order);

    symInt32QConfig_t aQC, bQC, oQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &aQC, 13); /* wider than the int12 default */
    initSymInt32QConfig(HALF_AWAY, &bQC);
    initSymInt32QConfig(HALF_AWAY, &oQC);
    quantization_t aQ, bQ, oQ;
    initSymInt32Quantization(&aQC, &aQ);
    initSymInt32Quantization(&bQC, &bQ);
    initSymInt32Quantization(&oQC, &oQ);
    setTensorValues(&a, (uint8_t *)aData, &aS, &aQ, NULL);
    setTensorValues(&b, (uint8_t *)bData, &bS, &bQ, NULL);
    setTensorValues(&out, (uint8_t *)outData, &oS, &oQ, NULL);

    matmulSymInt32Tensors(&a, &b, &out);

    int32_t expected[] = {14, 32, 32, 77};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, outData, 4);
}

void testMatmulSymInt32TensorsWithBiasRejectsNonSymInt32Bias() {
    /* aTensor/bTensor are valid SYM_INT32 operands, but the bias is FLOAT32.
     * Without a type guard the bias branch reinterprets the float bytes as
     * int32 and dereferences a FLOAT32 qConfig as symInt32QConfig_t (#247). */
    tensor_t a, b, out;
    int32_t aData[] = {1, 2, 3, 4, 5, 6};
    int32_t bData[] = {1, 4, 2, 5, 3, 6};
    int32_t outData[4];
    size_t aDims[] = {2, 3}, bDims[] = {3, 2}, oDims[] = {2, 2}, order[] = {0, 1};
    shape_t aS, bS, oS;
    setShape(&aS, aDims, 2, order);
    setShape(&bS, bDims, 2, order);
    setShape(&oS, oDims, 2, order);

    symInt32QConfig_t aQC, bQC, oQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    initSymInt32QConfig(HALF_AWAY, &bQC);
    initSymInt32QConfig(HALF_AWAY, &oQC);
    quantization_t aQ, bQ, oQ;
    initSymInt32Quantization(&aQC, &aQ);
    initSymInt32Quantization(&bQC, &bQ);
    initSymInt32Quantization(&oQC, &oQ);
    setTensorValues(&a, (uint8_t *)aData, &aS, &aQ, NULL);
    setTensorValues(&b, (uint8_t *)bData, &bS, &bQ, NULL);
    setTensorValues(&out, (uint8_t *)outData, &oS, &oQ, NULL);

    /* bias: FLOAT32, element count == output columns (2) so it clears the
     * count check and reaches the type-confusing read. */
    tensor_t bias;
    float biasData[] = {1.f, 2.f};
    size_t biasDims[] = {2}, biasOrder[] = {0};
    shape_t biasS;
    setShape(&biasS, biasDims, 1, biasOrder);
    quantization_t biasQ;
    initFloat32Quantization(&biasQ);
    setTensorValues(&bias, (uint8_t *)biasData, &biasS, &biasQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(matmulSymInt32TensorsWithBias(&a, &b, &out, &bias, NULL));
}

/* ---- Group-quant PR2 (Task 3): matmulSymInt32TensorsGroupedWeight -------
 *
 * `b` in every test below is the shape the executeOp prologue's grouped-
 * operand branch ALWAYS produces when it reaches this kernel: raw (unpacked)
 * int32 mantissas, SYM_INT32 dtype, scale poisoned to 1.0f (never read here
 * — the real per-group scales live in the separate `weightGroups`
 * argument). Both gold fixtures share the same `a`/`b` mantissas and bias
 * (generate_expected_group_matmul.py); only the group SHAPE differs. */

void testMatmulGroupedWeightPerChannelMatchesGold(void) {
    tensor_t aTensor;
    size_t aDims[] = {(size_t)kPerChannelOutRows, (size_t)kPerChannelReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = kPerChannelAScale;
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kPerChannelAMantissas, &aShape, &aQ, NULL);

    tensor_t bTensor;
    size_t bDims[] = {(size_t)kPerChannelOutCols, (size_t)kPerChannelReduceLen};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    symInt32QConfig_t bQC;
    initSymInt32QConfig(HALF_AWAY, &bQC);
    bQC.scale = 1.0f; /* poison, per the funnel's grouped-scratch contract */
    bQC.qMaxBits = 8;
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)kPerChannelWMantissas, &bShape, &bQ, NULL);

    float wScales[3];
    memcpy(wScales, kPerChannelWScales, sizeof(wScales));
    symQConfig_t weightGroups = {.scales = wScales,
                                 .numGroups = (size_t)kPerChannelNumGroups,
                                 .groupSize = (size_t)kPerChannelGroupSize,
                                 .qBits = 8,
                                 .roundingMode = HALF_AWAY};

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kPerChannelOutCols};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    symInt32QConfig_t biasQC;
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    biasQC.scale = kPerChannelBiasScale;
    quantization_t biasQ;
    initSymInt32Quantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kPerChannelBiasMantissas, &biasShape, &biasQ, NULL);

    tensor_t outputTensor;
    int32_t outputData[6];
    size_t outputDims[] = {(size_t)kPerChannelOutRows, (size_t)kPerChannelOutCols};
    size_t outputOrder[] = {0, 1};
    shape_t outputShape;
    setShape(&outputShape, outputDims, 2, outputOrder);
    symInt32QConfig_t outputQC;
    initSymInt32QConfig(HALF_AWAY, &outputQC);
    quantization_t outputQ;
    initSymInt32Quantization(&outputQC, &outputQ);
    setTensorValues(&outputTensor, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, &biasTensor, &outputTensor,
                                       &groupedWeight, NULL);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kPerChannelOutMantissas, outputData, 6);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, kPerChannelOutScale, outputQC.scale);
}

void testMatmulGroupedWeightGeneralGroupsMatchesGold(void) {
    tensor_t aTensor;
    size_t aDims[] = {(size_t)kGeneralOutRows, (size_t)kGeneralReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = kGeneralAScale;
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kGeneralAMantissas, &aShape, &aQ, NULL);

    tensor_t bTensor;
    size_t bDims[] = {(size_t)kGeneralOutCols, (size_t)kGeneralReduceLen};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    symInt32QConfig_t bQC;
    initSymInt32QConfig(HALF_AWAY, &bQC);
    bQC.scale = 1.0f;
    bQC.qMaxBits = 8;
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)kGeneralWMantissas, &bShape, &bQ, NULL);

    float wScales[6];
    memcpy(wScales, kGeneralWScales, sizeof(wScales));
    symQConfig_t weightGroups = {.scales = wScales,
                                 .numGroups = (size_t)kGeneralNumGroups,
                                 .groupSize = (size_t)kGeneralGroupSize,
                                 .qBits = 8,
                                 .roundingMode = HALF_AWAY};

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kGeneralOutCols};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    symInt32QConfig_t biasQC;
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    biasQC.scale = kGeneralBiasScale;
    quantization_t biasQ;
    initSymInt32Quantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kGeneralBiasMantissas, &biasShape, &biasQ, NULL);

    tensor_t outputTensor;
    int32_t outputData[6];
    size_t outputDims[] = {(size_t)kGeneralOutRows, (size_t)kGeneralOutCols};
    size_t outputOrder[] = {0, 1};
    shape_t outputShape;
    setShape(&outputShape, outputDims, 2, outputOrder);
    symInt32QConfig_t outputQC;
    initSymInt32QConfig(HALF_AWAY, &outputQC);
    quantization_t outputQ;
    initSymInt32Quantization(&outputQC, &outputQ);
    setTensorValues(&outputTensor, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, &biasTensor, &outputTensor,
                                       &groupedWeight, NULL);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kGeneralOutMantissas, outputData, 6);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, kGeneralOutScale, outputQC.scale);
}

/* Equal-scales grouped twin: every group's scale is the SAME power-of-two
 * value (0.25f), and aScale (0.5f) is also a power of two, so s_acc =
 * aScale*maxScale and every combine's paramScale (aScale*scales[g]) are BIT-
 * IDENTICAL float32 values (same formula, same operands, IEEE754 is
 * deterministic). Dividing a float by the SAME power-of-two value it was
 * just multiplied by is an EXACT round trip (pure exponent shifts, no
 * mantissa rounding, no overflow at these magnitudes) — so
 * round_half_away(partial*paramScale/s_acc) reproduces `partial` exactly.
 * A non-power-of-two scale (e.g. 0.02, the other fixtures) would NOT
 * guarantee this: `partial*paramScale` itself already carries a rounding
 * error before the divide-back, and double-rounding is not generally
 * invertible. The grouped kernel's output must therefore be BIT-IDENTICAL to
 * the scalar (non-grouped) matmulSymInt32TensorsWithBias run on the same
 * mantissas with b's scale == the common group scale. */
void testMatmulGroupedEqualScalesBitIdenticalToScalar(void) {
    const float commonScale = 0.25f;
    const float aScaleVal = 0.5f;

    tensor_t aTensor;
    size_t aDims[] = {2, 6};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = aScaleVal;
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kPerChannelAMantissas, &aShape, &aQ, NULL);

    size_t bDims[] = {3, 6};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);

    tensor_t bTensorGrouped;
    symInt32QConfig_t bGroupedQC;
    initSymInt32QConfig(HALF_AWAY, &bGroupedQC);
    bGroupedQC.scale = 1.0f;
    quantization_t bGroupedQ;
    initSymInt32Quantization(&bGroupedQC, &bGroupedQ);
    setTensorValues(&bTensorGrouped, (uint8_t *)kPerChannelWMantissas, &bShape, &bGroupedQ, NULL);

    float scales[3] = {commonScale, commonScale, commonScale};
    symQConfig_t weightGroups = {
        .scales = scales, .numGroups = 3, .groupSize = 6, .qBits = 8, .roundingMode = HALF_AWAY};

    size_t outDims[] = {2, 3};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);

    tensor_t outGrouped;
    int32_t outGroupedData[6];
    symInt32QConfig_t outGroupedQC;
    initSymInt32QConfig(HALF_AWAY, &outGroupedQC);
    quantization_t outGroupedQ;
    initSymInt32Quantization(&outGroupedQC, &outGroupedQ);
    setTensorValues(&outGrouped, (uint8_t *)outGroupedData, &outShape, &outGroupedQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensorGrouped, NULL, &outGrouped, &groupedWeight,
                                       NULL);

    tensor_t bTensorScalar;
    symInt32QConfig_t bScalarQC;
    initSymInt32QConfig(HALF_AWAY, &bScalarQC);
    bScalarQC.scale = commonScale;
    quantization_t bScalarQ;
    initSymInt32Quantization(&bScalarQC, &bScalarQ);
    setTensorValues(&bTensorScalar, (uint8_t *)kPerChannelWMantissas, &bShape, &bScalarQ, NULL);

    tensor_t outScalar;
    int32_t outScalarData[6];
    symInt32QConfig_t outScalarQC;
    initSymInt32QConfig(HALF_AWAY, &outScalarQC);
    quantization_t outScalarQ;
    initSymInt32Quantization(&outScalarQC, &outScalarQ);
    setTensorValues(&outScalar, (uint8_t *)outScalarData, &outShape, &outScalarQ, NULL);

    matmulSymInt32TensorsWithBias(&aTensor, &bTensorScalar, &outScalar, NULL, NULL);

    TEST_ASSERT_EQUAL_INT32_ARRAY(outScalarData, outGroupedData, 6);
    TEST_ASSERT_EQUAL_FLOAT(outScalarQC.scale, outGroupedQC.scale);
}

/* Rounding-mode-honored check (substitutes for a Python-emulated SR-mode
 * gold fixture, which would need the C-side keyed stream — no existing
 * goldgen script emulates SR_HALF_AWAY, see generate_expected_group_matmul.py's
 * docstring). Reuses the general-groups fixture (known non-exact combine
 * quotients) with SR_HALF_AWAY as the op's rounding mode (carried via b's own
 * qConfig — the same field the executeOp prologue would set from
 * arithmetic.roundingMode). If the combine hardcoded HALF_AWAY instead of
 * honoring roundingMode, two runs on streams of different keys would be BIT-
 * IDENTICAL (SR jitter never consumed); a correct implementation must differ
 * on at least one output element. */
void testMatmulGroupedHonorsOpRoundingMode(void) {
    tensor_t aTensor;
    size_t aDims[] = {(size_t)kGeneralOutRows, (size_t)kGeneralReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = kGeneralAScale;
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kGeneralAMantissas, &aShape, &aQ, NULL);

    tensor_t bTensor;
    size_t bDims[] = {(size_t)kGeneralOutCols, (size_t)kGeneralReduceLen};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    symInt32QConfig_t bQC;
    initSymInt32QConfig(SR_HALF_AWAY, &bQC); /* the op's rounding mode */
    bQC.scale = 1.0f;
    bQC.qMaxBits = 8;
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)kGeneralWMantissas, &bShape, &bQ, NULL);

    float wScales[6];
    memcpy(wScales, kGeneralWScales, sizeof(wScales));
    symQConfig_t weightGroups = {.scales = wScales,
                                 .numGroups = (size_t)kGeneralNumGroups,
                                 .groupSize = (size_t)kGeneralGroupSize,
                                 .qBits = 8,
                                 .roundingMode = HALF_AWAY};

    size_t outDims[] = {(size_t)kGeneralOutRows, (size_t)kGeneralOutCols};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);

    tensor_t out1;
    int32_t out1Data[6];
    symInt32QConfig_t out1QC;
    initSymInt32QConfig(HALF_AWAY, &out1QC);
    quantization_t out1Q;
    initSymInt32Quantization(&out1QC, &out1Q);
    setTensorValues(&out1, (uint8_t *)out1Data, &outShape, &out1Q, NULL);

    tensor_t out2;
    int32_t out2Data[6];
    symInt32QConfig_t out2QC;
    initSymInt32QConfig(HALF_AWAY, &out2QC);
    quantization_t out2Q;
    initSymInt32Quantization(&out2QC, &out2Q);
    setTensorValues(&out2, (uint8_t *)out2Data, &outShape, &out2Q, NULL);

    rngStream_t rs1 = rngOpen(rngKeyFromSeed(1u));
    rngStream_t rs2 = rngOpen(rngKeyFromSeed(2u));
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &out1, &groupedWeight, &rs1);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &out2, &groupedWeight, &rs2);

    bool anyDiffer = false;
    for (size_t i = 0; i < 6; i++) {
        if (out1Data[i] != out2Data[i]) {
            anyDiffer = true;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(anyDiffer,
                             "grouped combine did not honor SR_HALF_AWAY (hardcoded rounding?)");
}

/* ---- Group-quant PR3 (Task 1): dx (Linear propLoss) orientation ---------
 *
 * The weight `b` is passed in its RAW [outFeatures, inFeatures] storage
 * order (order {0,1}, NO transpose — exactly what linearBackward's propLoss
 * op hands the kernel): logical dim 0 (the reduction axis, outFeatures)
 * strides weight STORAGE by inFeatures. Groups still bind to flat storage,
 * so consecutive reduction steps hop groups — the unified per-element group
 * lookup must fold the running partial on every visited-group change (for
 * per-channel weights that is EVERY step) plus the tail. Fixtures:
 * generate_expected_group_matmul.py's Dx* cases (same weight mantissas as
 * the forward fixtures, seeded row-distinct pseudo-random loss). */

void testMatmulGroupedWeightStridedReductionMatchesGold(void) {
    tensor_t aTensor;
    size_t aDims[] = {(size_t)kDxPerChannelOutRows, (size_t)kDxPerChannelReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = kDxPerChannelLossScale;
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kDxPerChannelLossMantissas, &aShape, &aQ, NULL);

    tensor_t bTensor;
    size_t bDims[] = {(size_t)kDxPerChannelReduceLen, (size_t)kDxPerChannelOutCols};
    size_t bOrder[] = {0, 1}; /* RAW storage view: reduction axis strided by outCols */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    symInt32QConfig_t bQC;
    initSymInt32QConfig(HALF_AWAY, &bQC);
    bQC.scale = 1.0f; /* poison, per the funnel's grouped-scratch contract */
    bQC.qMaxBits = 8;
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)kDxPerChannelWMantissas, &bShape, &bQ, NULL);

    float wScales[3];
    memcpy(wScales, kDxPerChannelWScales, sizeof(wScales));
    symQConfig_t weightGroups = {.scales = wScales,
                                 .numGroups = (size_t)kDxPerChannelNumGroups,
                                 .groupSize = (size_t)kDxPerChannelGroupSize,
                                 .qBits = 8,
                                 .roundingMode = HALF_AWAY};

    tensor_t outputTensor;
    int32_t outputData[12];
    size_t outputDims[] = {(size_t)kDxPerChannelOutRows, (size_t)kDxPerChannelOutCols};
    size_t outputOrder[] = {0, 1};
    shape_t outputShape;
    setShape(&outputShape, outputDims, 2, outputOrder);
    symInt32QConfig_t outputQC;
    initSymInt32QConfig(HALF_AWAY, &outputQC);
    quantization_t outputQ;
    initSymInt32Quantization(&outputQC, &outputQ);
    setTensorValues(&outputTensor, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &outputTensor, &groupedWeight,
                                       NULL);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kDxPerChannelOutMantissas, outputData, 12);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, kDxPerChannelOutScale, outputQC.scale);
}

void testMatmulGroupedWeightStridedGeneralGroupsMatchesGold(void) {
    tensor_t aTensor;
    size_t aDims[] = {(size_t)kDxGeneralOutRows, (size_t)kDxGeneralReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = kDxGeneralLossScale;
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kDxGeneralLossMantissas, &aShape, &aQ, NULL);

    tensor_t bTensor;
    size_t bDims[] = {(size_t)kDxGeneralReduceLen, (size_t)kDxGeneralOutCols};
    size_t bOrder[] = {0, 1}; /* RAW storage view: reduction axis strided by outCols */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    symInt32QConfig_t bQC;
    initSymInt32QConfig(HALF_AWAY, &bQC);
    bQC.scale = 1.0f;
    bQC.qMaxBits = 8;
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)kDxGeneralWMantissas, &bShape, &bQ, NULL);

    float wScales[6];
    memcpy(wScales, kDxGeneralWScales, sizeof(wScales));
    symQConfig_t weightGroups = {.scales = wScales,
                                 .numGroups = (size_t)kDxGeneralNumGroups,
                                 .groupSize = (size_t)kDxGeneralGroupSize,
                                 .qBits = 8,
                                 .roundingMode = HALF_AWAY};

    tensor_t outputTensor;
    int32_t outputData[12];
    size_t outputDims[] = {(size_t)kDxGeneralOutRows, (size_t)kDxGeneralOutCols};
    size_t outputOrder[] = {0, 1};
    shape_t outputShape;
    setShape(&outputShape, outputDims, 2, outputOrder);
    symInt32QConfig_t outputQC;
    initSymInt32QConfig(HALF_AWAY, &outputQC);
    quantization_t outputQ;
    initSymInt32Quantization(&outputQC, &outputQ);
    setTensorValues(&outputTensor, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &outputTensor, &groupedWeight,
                                       NULL);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kDxGeneralOutMantissas, outputData, 12);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, kDxGeneralOutScale, outputQC.scale);
}

/* dx-orientation twin of testMatmulGroupedEqualScalesBitIdenticalToScalar
 * (same power-of-two exactness argument, see that test's comment): every
 * group's scale is 0.25f and aScale is 0.5f, so every combine's rescale is
 * an exact identity and the grouped kernel's strided-reduction output must
 * be BIT-IDENTICAL to the scalar matmulSymInt32Tensors run on the same RAW
 * weight view with b's scale == the common group scale. Pins the per-element
 * walk's bookkeeping (every element folded exactly once) independently of
 * any rescale rounding. */
void testMatmulGroupedDxEqualScalesBitIdenticalToScalar(void) {
    const float commonScale = 0.25f;
    const float aScaleVal = 0.5f;

    tensor_t aTensor;
    size_t aDims[] = {(size_t)kDxPerChannelOutRows, (size_t)kDxPerChannelReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = aScaleVal;
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kDxPerChannelLossMantissas, &aShape, &aQ, NULL);

    size_t bDims[] = {(size_t)kDxPerChannelReduceLen, (size_t)kDxPerChannelOutCols};
    size_t bOrder[] = {0, 1}; /* RAW storage view: reduction axis strided by outCols */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);

    tensor_t bTensorGrouped;
    symInt32QConfig_t bGroupedQC;
    initSymInt32QConfig(HALF_AWAY, &bGroupedQC);
    bGroupedQC.scale = 1.0f;
    quantization_t bGroupedQ;
    initSymInt32Quantization(&bGroupedQC, &bGroupedQ);
    setTensorValues(&bTensorGrouped, (uint8_t *)kDxPerChannelWMantissas, &bShape, &bGroupedQ, NULL);

    float scales[3] = {commonScale, commonScale, commonScale};
    symQConfig_t weightGroups = {
        .scales = scales, .numGroups = 3, .groupSize = 6, .qBits = 8, .roundingMode = HALF_AWAY};

    size_t outDims[] = {(size_t)kDxPerChannelOutRows, (size_t)kDxPerChannelOutCols};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);

    tensor_t outGrouped;
    int32_t outGroupedData[12];
    symInt32QConfig_t outGroupedQC;
    initSymInt32QConfig(HALF_AWAY, &outGroupedQC);
    quantization_t outGroupedQ;
    initSymInt32Quantization(&outGroupedQC, &outGroupedQ);
    setTensorValues(&outGrouped, (uint8_t *)outGroupedData, &outShape, &outGroupedQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensorGrouped, NULL, &outGrouped, &groupedWeight,
                                       NULL);

    tensor_t bTensorScalar;
    symInt32QConfig_t bScalarQC;
    initSymInt32QConfig(HALF_AWAY, &bScalarQC);
    bScalarQC.scale = commonScale;
    quantization_t bScalarQ;
    initSymInt32Quantization(&bScalarQC, &bScalarQ);
    setTensorValues(&bTensorScalar, (uint8_t *)kDxPerChannelWMantissas, &bShape, &bScalarQ, NULL);

    tensor_t outScalar;
    int32_t outScalarData[12];
    symInt32QConfig_t outScalarQC;
    initSymInt32QConfig(HALF_AWAY, &outScalarQC);
    quantization_t outScalarQ;
    initSymInt32Quantization(&outScalarQC, &outScalarQ);
    setTensorValues(&outScalar, (uint8_t *)outScalarData, &outShape, &outScalarQ, NULL);

    matmulSymInt32Tensors(&aTensor, &bTensorScalar, &outScalar);

    TEST_ASSERT_EQUAL_INT32_ARRAY(outScalarData, outGroupedData, 12);
    TEST_ASSERT_EQUAL_FLOAT(outScalarQC.scale, outGroupedQC.scale);
}

/* The public grouped entry must reject the per-tensor sentinel {numGroups=1,
 * groupSize=0} (see symQConfig_t's field comments, Quantization.h): without
 * this guard, matmulIntCoreGrouped's `wStorageIdx / weightGroups->groupSize`
 * divides by the sentinel's groupSize=0. Per-tensor weights have their own
 * scalar entries (matmulSymInt32Tensors[WithBias]); this entry is grouped-
 * only. */
void testMatmulGroupedWeightRejectsPerTensorSentinel(void) {
    tensor_t aTensor;
    int32_t aData[] = {1, 1};
    size_t aDims[] = {1, 2};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)aData, &aShape, &aQ, NULL);

    tensor_t bTensor;
    int32_t bData[] = {1, 1};
    size_t bDims[] = {1, 2};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    symInt32QConfig_t bQC;
    initSymInt32QConfig(HALF_AWAY, &bQC);
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)bData, &bShape, &bQ, NULL);

    float sentinelScale = 1.0f;
    symQConfig_t weightGroups = {.scales = &sentinelScale,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 8,
                                 .roundingMode = HALF_AWAY};

    tensor_t outputTensor;
    int32_t outputData[1];
    size_t outDims[] = {1, 1};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);
    symInt32QConfig_t outputQC;
    initSymInt32QConfig(HALF_AWAY, &outputQC);
    quantization_t outputQ;
    initSymInt32Quantization(&outputQC, &outputQ);
    setTensorValues(&outputTensor, (uint8_t *)outputData, &outShape, &outputQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    ASSERT_EXITS_WITH_FAILURE(matmulSymInt32TensorsGroupedWeight(
        &aTensor, &bTensor, NULL, &outputTensor, &groupedWeight, NULL));
}

/* ---- BFP epic PR2 (Task 3): matmulBfpTensors ----------------------------
 *
 * Operands arrive in the funnel's UNPACKED-BFP scratch form: ->data holds
 * int32 sign-extended mantissa codes, ->quantization is BFP with a live
 * bfpQConfig_t (stack-fixture idiom, Quantization.h). Output is RAW FLOAT32
 * -- the kernel never rounds and never width-restores (both are the funnel's
 * job, not the kernel's). The gold fixture lives in the exact float regime
 * (generate_expected_bfp_matmul.py asserts it), so expectations are BIT-
 * pinned via TEST_ASSERT_EQUAL_MEMORY, not a tolerance. */

void testMatmulBfpMatchesGold(void) {
    tensor_t aTensor;
    size_t aDims[] = {(size_t)kBfpOutRows, (size_t)kBfpReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    /* sizeof(fixture) sizing: a regenerated gold with a different group count
     * fails loudly at the numGroups check instead of silently short-copying */
    uint8_t aExponents[sizeof(kBfpAExponents)];
    memcpy(aExponents, kBfpAExponents, sizeof(aExponents));
    bfpQConfig_t aQC = {.exponents = aExponents,
                        .numGroups = (size_t)kBfpANumGroups,
                        .groupSize = (size_t)kBfpAGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpAMantissaBits,
                        .exponentBits = (uint8_t)kBfpAExponentBits};
    quantization_t aQ;
    initBfpQuantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kBfpACodes, &aShape, &aQ, NULL);

    tensor_t bTensor;
    size_t bDims[] = {(size_t)kBfpOutCols, (size_t)kBfpReduceLen};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    uint8_t bExponents[sizeof(kBfpBExponents)];
    memcpy(bExponents, kBfpBExponents, sizeof(bExponents));
    bfpQConfig_t bQC = {.exponents = bExponents,
                        .numGroups = (size_t)kBfpBNumGroups,
                        .groupSize = (size_t)kBfpBGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpBMantissaBits,
                        .exponentBits = (uint8_t)kBfpBExponentBits};
    quantization_t bQ;
    initBfpQuantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)kBfpBCodes, &bShape, &bQ, NULL);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpOutCols};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpBiasExponents)];
    memcpy(biasExponents, kBfpBiasExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpBiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpBiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpBiasCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[6];
    size_t outDims[] = {(size_t)kBfpOutRows, (size_t)kBfpOutCols};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    matmulBfpTensors(&aTensor, &bTensor, &biasTensor, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpMatmulExpected, outTensor.data,
                             kBfpMatmulExpected_len * sizeof(float));
}

/* BFP epic PR2: the bias seed must dequantize through ITS OWN
 * group's exponent (bfpGroupScale(biasQC, bfpGroupOf(biasQC, col))). Same
 * bias VALUES as the gold, stored grouped {numGroups=3, groupSize=1} with
 * NON-UNIFORM exponents (goldgen asserts both, plus that a group-0 collapse
 * differs) -- so the expected output is bit-identical to the per-tensor gold,
 * and a kernel reading every seed through group 0 is off by a power of two
 * on the other columns. */
void testMatmulBfpGroupedBiasBindsPerGroupExponent(void) {
    tensor_t aTensor;
    size_t aDims[] = {(size_t)kBfpOutRows, (size_t)kBfpReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    uint8_t aExponents[sizeof(kBfpAExponents)];
    memcpy(aExponents, kBfpAExponents, sizeof(aExponents));
    bfpQConfig_t aQC = {.exponents = aExponents,
                        .numGroups = (size_t)kBfpANumGroups,
                        .groupSize = (size_t)kBfpAGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpAMantissaBits,
                        .exponentBits = (uint8_t)kBfpAExponentBits};
    quantization_t aQ;
    initBfpQuantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kBfpACodes, &aShape, &aQ, NULL);

    tensor_t bTensor;
    size_t bDims[] = {(size_t)kBfpOutCols, (size_t)kBfpReduceLen};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    uint8_t bExponents[sizeof(kBfpBExponents)];
    memcpy(bExponents, kBfpBExponents, sizeof(bExponents));
    bfpQConfig_t bQC = {.exponents = bExponents,
                        .numGroups = (size_t)kBfpBNumGroups,
                        .groupSize = (size_t)kBfpBGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpBMantissaBits,
                        .exponentBits = (uint8_t)kBfpBExponentBits};
    quantization_t bQ;
    initBfpQuantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)kBfpBCodes, &bShape, &bQ, NULL);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpOutCols};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpBiasGroupedExponents)];
    memcpy(biasExponents, kBfpBiasGroupedExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = (size_t)kBfpBiasGroupedNumGroups,
                           .groupSize = 1,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpBiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpBiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpBiasGroupedCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[6];
    size_t outDims[] = {(size_t)kBfpOutRows, (size_t)kBfpOutCols};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    matmulBfpTensors(&aTensor, &bTensor, &biasTensor, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpMatmulExpected, outTensor.data,
                             kBfpMatmulExpected_len * sizeof(float));
}

void testMatmulBfpNoBiasZeroSeeds(void) {
    tensor_t aTensor;
    size_t aDims[] = {(size_t)kBfpOutRows, (size_t)kBfpReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    uint8_t aExponents[sizeof(kBfpAExponents)];
    memcpy(aExponents, kBfpAExponents, sizeof(aExponents));
    bfpQConfig_t aQC = {.exponents = aExponents,
                        .numGroups = (size_t)kBfpANumGroups,
                        .groupSize = (size_t)kBfpAGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpAMantissaBits,
                        .exponentBits = (uint8_t)kBfpAExponentBits};
    quantization_t aQ;
    initBfpQuantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)kBfpACodes, &aShape, &aQ, NULL);

    tensor_t bTensor;
    size_t bDims[] = {(size_t)kBfpOutCols, (size_t)kBfpReduceLen};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    uint8_t bExponents[sizeof(kBfpBExponents)];
    memcpy(bExponents, kBfpBExponents, sizeof(bExponents));
    bfpQConfig_t bQC = {.exponents = bExponents,
                        .numGroups = (size_t)kBfpBNumGroups,
                        .groupSize = (size_t)kBfpBGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpBMantissaBits,
                        .exponentBits = (uint8_t)kBfpBExponentBits};
    quantization_t bQ;
    initBfpQuantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)kBfpBCodes, &bShape, &bQ, NULL);

    tensor_t outTensor;
    float outData[6];
    size_t outDims[] = {(size_t)kBfpOutRows, (size_t)kBfpOutCols};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    matmulBfpTensors(&aTensor, &bTensor, NULL, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpMatmulNoBiasExpected, outTensor.data,
                             kBfpMatmulNoBiasExpected_len * sizeof(float));
}

/* BFP power-of-two twin, the BFP sibling of
 * testMatmulGroupedEqualScalesBitIdenticalToScalar above (see its comment
 * for the exactness argument): identical mantissas; BFP b grouped
 * {numGroups=3, groupSize=6} with every stored exponent 125 (e=8, bias 127
 * -> 2^-2 == 0.25f) <-> SYM twin scales {0.25f, 0.25f, 0.25f}; BFP a
 * per-tensor stored 126 (2^-1 == 0.5f) <-> SYM aScale 0.5f. All products
 * and partials are < 2^24 (exact floats), every BFP fold is a pure exponent
 * shift, and the SYM side's equal-power-of-two rescales are exact round
 * trips -- so the BFP float output must be BIT-IDENTICAL to the SYM path's
 * dequantized output. */
void testMatmulBfpPowerOfTwoBitIdenticalToGroupedSym(void) {
    size_t aDims[] = {2, 6};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    size_t bDims[] = {3, 6};
    size_t bOrder[] = {1, 0}; /* post-transpose logical view: reduction axis first */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    size_t outDims[] = {2, 3};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);

    tensor_t aBfpTensor;
    uint8_t aExponents[] = {126}; /* 2^(126-127) == 0.5f */
    bfpQConfig_t aBfpQC = {.exponents = aExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 8,
                           .exponentBits = 8};
    quantization_t aBfpQ;
    initBfpQuantization(&aBfpQC, &aBfpQ);
    setTensorValues(&aBfpTensor, (uint8_t *)kPerChannelAMantissas, &aShape, &aBfpQ, NULL);

    tensor_t bBfpTensor;
    uint8_t bExponents[] = {125, 125, 125}; /* 2^(125-127) == 0.25f */
    bfpQConfig_t bBfpQC = {.exponents = bExponents,
                           .numGroups = 3,
                           .groupSize = 6,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 8,
                           .exponentBits = 8};
    quantization_t bBfpQ;
    initBfpQuantization(&bBfpQC, &bBfpQ);
    setTensorValues(&bBfpTensor, (uint8_t *)kPerChannelWMantissas, &bShape, &bBfpQ, NULL);

    tensor_t outBfpTensor;
    float outBfpData[6];
    quantization_t outBfpQ;
    initFloat32Quantization(&outBfpQ);
    setTensorValues(&outBfpTensor, (uint8_t *)outBfpData, &outShape, &outBfpQ, NULL);

    matmulBfpTensors(&aBfpTensor, &bBfpTensor, NULL, &outBfpTensor);

    tensor_t aSymTensor;
    symInt32QConfig_t aSymQC;
    initSymInt32QConfig(HALF_AWAY, &aSymQC);
    aSymQC.scale = 0.5f;
    quantization_t aSymQ;
    initSymInt32Quantization(&aSymQC, &aSymQ);
    setTensorValues(&aSymTensor, (uint8_t *)kPerChannelAMantissas, &aShape, &aSymQ, NULL);

    tensor_t bSymTensor;
    symInt32QConfig_t bSymQC;
    initSymInt32QConfig(HALF_AWAY, &bSymQC);
    bSymQC.scale = 1.0f;
    quantization_t bSymQ;
    initSymInt32Quantization(&bSymQC, &bSymQ);
    setTensorValues(&bSymTensor, (uint8_t *)kPerChannelWMantissas, &bShape, &bSymQ, NULL);

    float scales[3] = {0.25f, 0.25f, 0.25f};
    symQConfig_t weightGroups = {
        .scales = scales, .numGroups = 3, .groupSize = 6, .qBits = 8, .roundingMode = HALF_AWAY};

    tensor_t outSymTensor;
    int32_t outSymData[6];
    symInt32QConfig_t outSymQC;
    initSymInt32QConfig(HALF_AWAY, &outSymQC);
    quantization_t outSymQ;
    initSymInt32Quantization(&outSymQC, &outSymQ);
    setTensorValues(&outSymTensor, (uint8_t *)outSymData, &outShape, &outSymQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aSymTensor, &bSymTensor, NULL, &outSymTensor,
                                       &groupedWeight, NULL);

    for (size_t i = 0; i < 6; i++) {
        float symDequant = (float)outSymData[i] * outSymQC.scale;
        TEST_ASSERT_EQUAL_MEMORY(&symDequant, &outBfpData[i], sizeof(float));
    }
}

void testMatmulBfpHeadroomGuardDies(void) {
    /* per-tensor m=16 operands with K = 2 > bfpSegmentLimit(16, 16) == 1 --
     * boundary-tight on purpose: a limit+1 off-by-one in the guard would let
     * exactly this K through (K = 3 would still die under that mutation). */
    tensor_t aWide;
    int32_t aData[] = {1, 1};
    size_t aDims[] = {1, 2};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    uint8_t aExponents[] = {127};
    bfpQConfig_t aQC = {.exponents = aExponents,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 16,
                        .exponentBits = 8};
    quantization_t aQ;
    initBfpQuantization(&aQC, &aQ);
    setTensorValues(&aWide, (uint8_t *)aData, &aShape, &aQ, NULL);

    tensor_t bWide;
    int32_t bData[] = {1, 1};
    size_t bDims[] = {2, 1};
    size_t bOrder[] = {0, 1};
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    uint8_t bExponents[] = {127};
    bfpQConfig_t bQC = {.exponents = bExponents,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 16,
                        .exponentBits = 8};
    quantization_t bQ;
    initBfpQuantization(&bQC, &bQ);
    setTensorValues(&bWide, (uint8_t *)bData, &bShape, &bQ, NULL);

    tensor_t outTensor;
    float outData[1];
    size_t outDims[] = {1, 1};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(matmulBfpTensors(&aWide, &bWide, NULL, &outTensor));
}

void testBfpSegmentLimitTableValues(void) {
    TEST_ASSERT_EQUAL_size_t(131071, bfpSegmentLimit(8, 8));
    TEST_ASSERT_EQUAL_size_t(511, bfpSegmentLimit(12, 12));
    TEST_ASSERT_EQUAL_size_t(1, bfpSegmentLimit(16, 16));
}

/* PR3 sum-headroom sibling of the table above: single-operand SUMS (biasGrad)
 * are bounded by g * 2^(m-1), not the product bound, so the limit is
 * INT32_MAX >> (m-1) -- far roomier than bfpSegmentLimit at equal widths. */
void testBfpSumSegmentLimitTableValues(void) {
    TEST_ASSERT_EQUAL_size_t((size_t)(INT32_MAX >> 7), bfpSumSegmentLimit(8));
    TEST_ASSERT_EQUAL_size_t((size_t)(INT32_MAX >> 15), bfpSumSegmentLimit(16));
    TEST_ASSERT_EQUAL_size_t((size_t)(INT32_MAX >> 1), bfpSumSegmentLimit(2));
}

/* bfpGroupAt: the per-MAC group lookup divides only when the index leaves the
 * cached [start, end) of the current group. The cache is a public struct, so
 * the tests pre-load a WRONG group id to prove which path ran: an in-range
 * index must return the cached id untouched, an out-of-range index must
 * recompute (and overwrite it). */
static const bfpQConfig_t groupAtQc = {
    .exponents = NULL, .numGroups = 3, .groupSize = 4, .mantissaBits = 8, .exponentBits = 8};

void testBfpGroupAtInRangeTrustsCache(void) {
    bfpGroupRange_t r = {.group = 7, .start = 0, .end = 4};
    TEST_ASSERT_EQUAL_size_t(7, bfpGroupAt(&r, &groupAtQc, 2));
    TEST_ASSERT_EQUAL_size_t(7, bfpGroupAt(&r, &groupAtQc, 0));
    TEST_ASSERT_EQUAL_size_t(7, bfpGroupAt(&r, &groupAtQc, 3));
    TEST_ASSERT_EQUAL_size_t(0, r.start);
    TEST_ASSERT_EQUAL_size_t(4, r.end);
}

void testBfpGroupAtForwardJumpRecomputes(void) {
    bfpGroupRange_t r = {.group = 7, .start = 0, .end = 4};
    /* end is exclusive: index 4 is the first element of group 1 */
    TEST_ASSERT_EQUAL_size_t(1, bfpGroupAt(&r, &groupAtQc, 4));
    TEST_ASSERT_EQUAL_size_t(4, r.start);
    TEST_ASSERT_EQUAL_size_t(8, r.end);
    /* a jump that skips a whole group lands on the right one */
    TEST_ASSERT_EQUAL_size_t(2, bfpGroupAt(&r, &groupAtQc, 9));
    TEST_ASSERT_EQUAL_size_t(8, r.start);
    TEST_ASSERT_EQUAL_size_t(12, r.end);
}

void testBfpGroupAtBackwardJumpRecomputes(void) {
    bfpGroupRange_t r = {.group = 2, .start = 8, .end = 12};
    TEST_ASSERT_EQUAL_size_t(0, bfpGroupAt(&r, &groupAtQc, 1));
    TEST_ASSERT_EQUAL_size_t(0, r.start);
    TEST_ASSERT_EQUAL_size_t(4, r.end);
    /* start is inclusive: stepping back exactly onto start stays cached */
    r.group = 7;
    TEST_ASSERT_EQUAL_size_t(7, bfpGroupAt(&r, &groupAtQc, 0));
}

/* A zero-initialised range is EMPTY ([0, 0)), so the very first lookup --
 * index 0 included -- recomputes instead of trusting the zeroed group id. */
void testBfpGroupAtZeroInitIsEmptyRange(void) {
    bfpGroupRange_t r = {0};
    r.group = 7;
    TEST_ASSERT_EQUAL_size_t(0, bfpGroupAt(&r, &groupAtQc, 0));
    TEST_ASSERT_EQUAL_size_t(0, r.start);
    TEST_ASSERT_EQUAL_size_t(4, r.end);
}

/* The last group's range ends exactly at the element count (a partial last
 * group cannot exist: validateBfpQConfigShape enforces numGroups * groupSize
 * == N), so the final storage index resolves to numGroups - 1. */
void testBfpGroupAtLastGroupEndsAtElementCount(void) {
    bfpGroupRange_t r = {0};
    TEST_ASSERT_EQUAL_size_t(2, bfpGroupAt(&r, &groupAtQc, 11));
    TEST_ASSERT_EQUAL_size_t(8, r.start);
    TEST_ASSERT_EQUAL_size_t(12, r.end);
}

/* Per-tensor {numGroups=1, groupSize=0}: everything is group 0 and the first
 * lookup installs a range covering every index, so no later lookup ever
 * recomputes -- a poisoned group id after the first call proves it. */
void testBfpGroupAtPerTensorNeverRecomputes(void) {
    bfpQConfig_t perTensor = groupAtQc;
    perTensor.numGroups = 1;
    perTensor.groupSize = 0;
    bfpGroupRange_t r = {0};
    TEST_ASSERT_EQUAL_size_t(0, bfpGroupAt(&r, &perTensor, 5));
    TEST_ASSERT_EQUAL_size_t(0, r.start);
    TEST_ASSERT_EQUAL_size_t(SIZE_MAX, r.end);
    r.group = 9;
    TEST_ASSERT_EQUAL_size_t(9, bfpGroupAt(&r, &perTensor, 1000000));
    TEST_ASSERT_EQUAL_size_t(9, bfpGroupAt(&r, &perTensor, 0));
}

/* Bit-identity with bfpGroupOf over a strided walk that jumps forward and
 * backward across groups (the backward GEMM's transposed-view access pattern),
 * grouped and per-tensor. */
void testBfpGroupAtMatchesBfpGroupOfOnStridedWalk(void) {
    bfpQConfig_t grouped = groupAtQc;
    grouped.numGroups = 6;
    grouped.groupSize = 4;
    bfpQConfig_t perTensor = groupAtQc;
    perTensor.numGroups = 1;
    perTensor.groupSize = 0;
    bfpGroupRange_t rg = {0};
    bfpGroupRange_t rp = {0};
    for (size_t i = 0; i < 96; i++) {
        size_t idx = (i * 7) % 24;
        TEST_ASSERT_EQUAL_size_t(bfpGroupOf(&grouped, idx), bfpGroupAt(&rg, &grouped, idx));
        TEST_ASSERT_EQUAL_size_t(bfpGroupOf(&perTensor, idx), bfpGroupAt(&rp, &perTensor, idx));
    }
}

/* Integer fold helpers (arithmetic-bfp §13 "The integer fold"): a partial
 * enters the int32 accumulator as partial * 2^-k. k > 0 rounds HALF_AWAY in
 * the overflow-safe magnitude form, k == 0 is the identity, k < 0 is an
 * exact left shift the headroom proof licenses. */
void testBfpShrHalfAwayRoundsHalfAwayFromZero(void) {
    TEST_ASSERT_EQUAL_INT32(4, bfpShrHalfAway(7, 1));   /* 3.5 -> 4 */
    TEST_ASSERT_EQUAL_INT32(-4, bfpShrHalfAway(-7, 1)); /* -3.5 -> -4 */
    TEST_ASSERT_EQUAL_INT32(3, bfpShrHalfAway(5, 1));   /* 2.5 -> 3 */
    TEST_ASSERT_EQUAL_INT32(-3, bfpShrHalfAway(-5, 1));
    TEST_ASSERT_EQUAL_INT32(2, bfpShrHalfAway(6, 2));  /* 1.5 -> 2 */
    TEST_ASSERT_EQUAL_INT32(2, bfpShrHalfAway(9, 2));  /* 2.25 -> 2 */
    TEST_ASSERT_EQUAL_INT32(3, bfpShrHalfAway(11, 2)); /* 2.75 -> 3 */
    TEST_ASSERT_EQUAL_INT32(-3, bfpShrHalfAway(-11, 2));
    TEST_ASSERT_EQUAL_INT32(1, bfpShrHalfAway(1, 1)); /* 0.5 -> 1 */
    TEST_ASSERT_EQUAL_INT32(-1, bfpShrHalfAway(-1, 1));
    TEST_ASSERT_EQUAL_INT32(0, bfpShrHalfAway(1, 2)); /* 0.25 -> 0 */
    TEST_ASSERT_EQUAL_INT32(0, bfpShrHalfAway(0, 5));
    TEST_ASSERT_EQUAL_INT32(12, bfpShrHalfAway(12, 0));
    TEST_ASSERT_EQUAL_INT32(INT32_MIN, bfpShrHalfAway(INT32_MIN, 0));
    TEST_ASSERT_EQUAL_INT32(INT32_MAX, bfpShrHalfAway(INT32_MAX, 0));
}

void testBfpShrHalfAwayInt32Extremes(void) {
    TEST_ASSERT_EQUAL_INT32(1 << 30, bfpShrHalfAway(INT32_MAX, 1)); /* 2^30 - 0.5 -> 2^30 */
    TEST_ASSERT_EQUAL_INT32(-(1 << 30), bfpShrHalfAway(INT32_MIN, 1));
    TEST_ASSERT_EQUAL_INT32(2, bfpShrHalfAway(INT32_MAX, 30)); /* 1.99.. -> 2 */
    TEST_ASSERT_EQUAL_INT32(1, bfpShrHalfAway(INT32_MAX, 31)); /* 0.99.. -> 1 */
    TEST_ASSERT_EQUAL_INT32(-1, bfpShrHalfAway(INT32_MIN, 31));
    TEST_ASSERT_EQUAL_INT32(0, bfpShrHalfAway(INT32_MAX, 32));  /* < 0.5 */
    TEST_ASSERT_EQUAL_INT32(-1, bfpShrHalfAway(INT32_MIN, 32)); /* exactly -0.5 */
    TEST_ASSERT_EQUAL_INT32(0, bfpShrHalfAway(INT32_MAX, 40));
    TEST_ASSERT_EQUAL_INT32(0, bfpShrHalfAway(INT32_MIN, 40));
}

void testBfpAlignPartialLeftShiftIsExactAndRightShiftRounds(void) {
    TEST_ASSERT_EQUAL_INT32(40, bfpAlignPartial(5, -3));
    TEST_ASSERT_EQUAL_INT32(-40, bfpAlignPartial(-5, -3));
    TEST_ASSERT_EQUAL_INT32(3 << 29, bfpAlignPartial(3, -29));
    TEST_ASSERT_EQUAL_INT32(-(1 << 30), bfpAlignPartial(-1, -30));
    TEST_ASSERT_EQUAL_INT32(5, bfpAlignPartial(5, 0));
    TEST_ASSERT_EQUAL_INT32(4, bfpAlignPartial(7, 1));
    TEST_ASSERT_EQUAL_INT32(-4, bfpAlignPartial(-7, 1));
    TEST_ASSERT_EQUAL_INT32(0, bfpAlignPartial(INT32_MAX, 40));
}

/* The static headroom proof in units of 2^E_acc: (N 2^(ma+mb-2) + 2^(mb-1))
 * 2^-g + (N + 1) <= INT32_MAX, g the smallest integer (negative allowed) that
 * satisfies it -- the product bound, not the int32 bound on a partial. */
void testBfpFoldGuardBitsFromTheProductBound(void) {
    TEST_ASSERT_EQUAL_INT(-10, bfpFoldGuardBits(80, 8, 8, 8, "t")); /* HAR conv2/3 widths */
    TEST_ASSERT_EQUAL_INT(-20, bfpFoldGuardBits(6, 6, 4, 8, "t"));  /* the conv gold fixture */
    TEST_ASSERT_EQUAL_INT(-21, bfpFoldGuardBits(2, 6, 4, 0, "t"));
    TEST_ASSERT_EQUAL_INT(-28,
                          bfpFoldGuardBits(1, 2, 2, 0, "t")); /* T = 4: the floor is never hit */
    TEST_ASSERT_EQUAL_INT(1, bfpFoldGuardBits(2, 16, 16, 0, "t")); /* 2 * 2^30 = 2^31 > INT32_MAX */
    TEST_ASSERT_EQUAL_INT(1, bfpFoldGuardBits(600, 16, 8, 0, "t"));
    TEST_ASSERT_EQUAL_INT(0, bfpFoldGuardBits(1, 16, 16, 0, "t")); /* 2^30 + 2 fits exactly once */
    TEST_ASSERT_EQUAL_INT(0, bfpFoldGuardBits(511, 16, 8, 0, "t"));
    TEST_ASSERT_EQUAL_INT(-30, bfpFoldGuardBits(0, 8, 8, 0, "t")); /* empty: only the floor */
    /* g >= 2: the right-shift search must keep going past 1 (a search that
     * walked back from g = 1 with an out-of-range shift returned 1 here). */
    TEST_ASSERT_EQUAL_INT(2, bfpFoldGuardBits(4, 16, 16, 0, "t"));   /* 4 * 2^30 = 2^32 */
    TEST_ASSERT_EQUAL_INT(6, bfpFoldGuardBits(80, 16, 16, 0, "t"));  /* 80 * 2^30 ~ 2^36.3 */
    TEST_ASSERT_EQUAL_INT(3, bfpFoldGuardBits(2400, 16, 8, 0, "t")); /* 2400 * 2^22 ~ 2^33.2 */
    TEST_ASSERT_EQUAL_INT(2, bfpFoldGuardBits(4, 16, 16, 16, "t"));  /* + a m16 bias segment */
    TEST_ASSERT_EQUAL_INT(-15, bfpFoldGuardBits(2, 4, 4, 16, "t"));  /* bias-dominated */
}

/* The proof's inequality checked directly on the returned g over a sweep of
 * (N, ma, mb, mBias): g satisfies it and g - 1 does not (minimality), both
 * sides in exact 64-bit arithmetic with shift counts always in range. */
static bool foldGuardHolds(uint64_t total, uint64_t room, int g) {
    if (g >= 0) {
        return g <= 31 && total <= (room << (unsigned)g);
    }
    return -g <= 30 && (total << (unsigned)(-g)) <= room;
}

void testBfpFoldGuardBitsIsTheSmallestGThatHolds(void) {
    static const uint8_t widths[] = {2, 4, 8, 12, 16};
    static const uint8_t biases[] = {0, 8, 16};
    static const size_t lengths[] = {1, 2, 3, 4, 7, 8, 63, 80, 511, 512, 600, 2400, 100000};
    for (size_t wa = 0; wa < 5; wa++) {
        for (size_t wb = 0; wb < 5; wb++) {
            for (size_t li = 0; li < 13; li++) {
                for (size_t bi = 0; bi < 3; bi++) {
                    size_t n = lengths[li];
                    uint64_t total = ((uint64_t)n << (widths[wa] + widths[wb] - 2u)) +
                                     (biases[bi] ? (uint64_t)1 << (biases[bi] - 1u) : 0u);
                    uint64_t room = (uint64_t)INT32_MAX - (n + 1u);
                    if (total > (room << 31)) {
                        continue; /* the fail-fast regime, covered by the death test */
                    }
                    int g = bfpFoldGuardBits(n, widths[wa], widths[wb], biases[bi], "sweep");
                    TEST_ASSERT_TRUE(foldGuardHolds(total, room, g));
                    TEST_ASSERT_TRUE(g == -30 || !foldGuardHolds(total, room, g - 1));
                }
            }
        }
    }
}

static void bfpFoldGuardBitsOverlongReductionBody(void) {
    (void)bfpFoldGuardBits((size_t)1 << 33, 16, 16, 0, "t"); /* g would be 33: every fold flushes */
}

void testBfpFoldGuardBitsRejectsAReductionNoShiftCanHold(void) {
    ASSERT_EXITS_WITH_FAILURE(bfpFoldGuardBitsOverlongReductionBody());
}

/* ---- Pool helpers of the integer-only forward (arithmetic-bfp.md §13) ---- */

/* Exact reference for value(ma, ea) > value(mb, eb): both sides as int64 on
 * the smaller exponent's grid. A gap beyond 47 binades decides exactly like
 * a gap of 47 (|code| <= 2^15, so 2^47 already dwarfs the other side), which
 * keeps the int64 product in range. */
static bool exactValueGreater(int32_t ma, int ea, int32_t mb, int eb) {
    int lo = ea < eb ? ea : eb;
    int da = ea - lo > 47 ? 47 : ea - lo;
    int db = eb - lo > 47 ? 47 : eb - lo;
    return (int64_t)ma * ((int64_t)1 << da) > (int64_t)mb * ((int64_t)1 << db);
}

void testBfpValueGreaterIsTheExactCompare(void) {
    static const int32_t codes[] = {-32768, -32767, -129, -128, -2, -1, 0, 1, 2, 127, 32767};
    static const int gaps[] = {0, 1, 2, 7, 14, 15, 16, 17, 30, 40, 200};
    const size_t nc = sizeof(codes) / sizeof(codes[0]);
    const size_t ng = sizeof(gaps) / sizeof(gaps[0]);
    for (size_t a = 0; a < nc; a++) {
        for (size_t b = 0; b < nc; b++) {
            for (size_t d = 0; d < ng; d++) {
                int base = -60;
                /* Both orders: a above b by the gap, and b above a. */
                TEST_ASSERT_EQUAL(exactValueGreater(codes[a], base + gaps[d], codes[b], base),
                                  bfpValueGreater(codes[a], base + gaps[d], codes[b], base));
                TEST_ASSERT_EQUAL(exactValueGreater(codes[a], base, codes[b], base + gaps[d]),
                                  bfpValueGreater(codes[a], base, codes[b], base + gaps[d]));
            }
        }
    }
    /* Equal values across groups are NOT greater (the tie keeps the first). */
    TEST_ASSERT_FALSE(bfpValueGreater(8, 0, 4, 1));
    TEST_ASSERT_FALSE(bfpValueGreater(4, 1, 8, 0));
    TEST_ASSERT_FALSE(bfpValueGreater(0, 20, 0, 0));
    TEST_ASSERT_FALSE(bfpValueGreater(-32768, 16, -32768, 16));
}

/* Reference: round half away from zero of a / b in exact int64. */
static int32_t exactDivHalfAway(int64_t a, int64_t b) {
    int64_t mag = a < 0 ? -a : a;
    int64_t q = mag / b;
    if (2 * (mag - q * b) >= b) {
        q++;
    }
    return (int32_t)(a < 0 ? -q : q);
}

void testRoundedDivHalfAwayInt32IsExactOverTheInt32Range(void) {
    static const int32_t sums[] = {
        INT32_MIN, INT32_MIN + 1, -1000001,      -7,       -5, -3, -2, -1, 0, 1, 2, 3, 5,
        7,         1000001,       INT32_MAX - 1, INT32_MAX};
    static const int32_t divs[] = {1, 2, 3, 4, 5, 6, 7, 10, 100, 65535, INT32_MAX};
    for (size_t s = 0; s < sizeof(sums) / sizeof(sums[0]); s++) {
        for (size_t d = 0; d < sizeof(divs) / sizeof(divs[0]); d++) {
            TEST_ASSERT_EQUAL_INT32(exactDivHalfAway(sums[s], divs[d]),
                                    roundedDivHalfAwayInt32(sums[s], divs[d]));
        }
    }
    /* The ties, both signs: 5/2 -> 3, -5/2 -> -3, 3/6 -> 1, -3/6 -> -1. */
    TEST_ASSERT_EQUAL_INT32(3, roundedDivHalfAwayInt32(5, 2));
    TEST_ASSERT_EQUAL_INT32(-3, roundedDivHalfAwayInt32(-5, 2));
    TEST_ASSERT_EQUAL_INT32(1, roundedDivHalfAwayInt32(3, 6));
    TEST_ASSERT_EQUAL_INT32(-1, roundedDivHalfAwayInt32(-3, 6));
}

/* The SUM guard is the product proof with the second factor bounded by 1:
 * K codes, each |code| <= 2^(m-1), so K 2^(m-1) 2^-g + (K + 1) <= INT32_MAX,
 * smallest g -- checked on the inequality itself, in 64-bit arithmetic. */
void testBfpSumFoldGuardBitsIsTheSmallestGThatHolds(void) {
    static const uint8_t widths[] = {2, 4, 6, 8, 12, 16};
    static const size_t lengths[] = {1, 2, 3, 4, 7, 32, 100, 511, 65536, 1000000};
    for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
        for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); l++) {
            size_t n = lengths[l];
            uint64_t total = (uint64_t)n << (widths[w] - 1u);
            uint64_t room = (uint64_t)INT32_MAX - (n + 1u);
            int g = bfpSumFoldGuardBits(n, widths[w], "sweep");
            TEST_ASSERT_TRUE(foldGuardHolds(total, room, g));
            TEST_ASSERT_TRUE(g == -30 || !foldGuardHolds(total, room, g - 1));
        }
    }
    TEST_ASSERT_EQUAL_INT(-18, bfpSumFoldGuardBits(32, 8, "t")); /* HAR AvgPool K = 32 */
}

/* Group-shape fail-fast: bfpGroupOf divides by groupSize
 * with no relation to numGroups, so a mismatched config ({numGroups=2,
 * groupSize=4} on 12 elements: 2*4 == 8 != 12) would silently read
 * exponents[] out of bounds. The kernel must route every operand through
 * validateBfpQConfigShape (the SYM grouped entry's validateSymQConfigShape
 * precedent) before touching data. */
void testMatmulBfpRejectsMismatchedGroupShape(void) {
    tensor_t aTensor;
    int32_t aData[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    size_t aDims[] = {2, 6};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    uint8_t aExponents[] = {127, 127};
    bfpQConfig_t aQC = {.exponents = aExponents,
                        .numGroups = 2,
                        .groupSize = 4, /* 2*4 == 8 != 12 elements */
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 6,
                        .exponentBits = 8};
    quantization_t aQ;
    initBfpQuantization(&aQC, &aQ);
    setTensorValues(&aTensor, (uint8_t *)aData, &aShape, &aQ, NULL);

    tensor_t bTensor;
    int32_t bData[] = {1, 1, 1, 1, 1, 1};
    size_t bDims[] = {6, 1};
    size_t bOrder[] = {0, 1};
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    uint8_t bExponents[] = {127};
    bfpQConfig_t bQC = {.exponents = bExponents,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 6,
                        .exponentBits = 8};
    quantization_t bQ;
    initBfpQuantization(&bQC, &bQ);
    setTensorValues(&bTensor, (uint8_t *)bData, &bShape, &bQ, NULL);

    tensor_t outTensor;
    float outData[2];
    size_t outDims[] = {2, 1};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(matmulBfpTensors(&aTensor, &bTensor, NULL, &outTensor));
}

/* PR3 kernel-reuse pin: Linear dx = loss @ W with W in RAW [outF, inF] storage
 * -- the reduction walks W strided by inF, hopping weight groups mid-reduction.
 * matmulBfpTensors' per-element lookup + either-boundary fold must handle the
 * strided walk exactly (BFP D8 amendment: no re-quantize, folds just more often). */
void testMatmulBfpDxStridedWeightWalkMatchesGold(void) {
    tensor_t lossT;
    size_t lossDims[] = {(size_t)kBfpBwdBatch, (size_t)kBfpBwdOutF};
    size_t lossOrder[] = {0, 1};
    shape_t lossShape;
    setShape(&lossShape, lossDims, 2, lossOrder);
    uint8_t lossExps[sizeof(kBfpLossExponents)];
    memcpy(lossExps, kBfpLossExponents, sizeof(lossExps));
    bfpQConfig_t lossQC = {.exponents = lossExps,
                           .numGroups = (size_t)kBfpLossNumGroups,
                           .groupSize = (size_t)kBfpLossGroupSize,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpLossMantissaBits,
                           .exponentBits = (uint8_t)kBfpLossExponentBits};
    quantization_t lossQ;
    initBfpQuantization(&lossQC, &lossQ);
    setTensorValues(&lossT, (uint8_t *)kBfpLossCodes, &lossShape, &lossQ, NULL);

    tensor_t wT;
    size_t wDims[] = {(size_t)kBfpBwdOutF, (size_t)kBfpBwdInF};
    size_t wOrder[] = {0, 1}; /* RAW storage order: reduction axis strided */
    shape_t wShape;
    setShape(&wShape, wDims, 2, wOrder);
    uint8_t wExps[sizeof(kBfpWbExponents)];
    memcpy(wExps, kBfpWbExponents, sizeof(wExps));
    bfpQConfig_t wQC = {.exponents = wExps,
                        .numGroups = (size_t)kBfpWbNumGroups,
                        .groupSize = (size_t)kBfpWbGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpWbMantissaBits,
                        .exponentBits = (uint8_t)kBfpWbExponentBits};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wT, (uint8_t *)kBfpWbCodes, &wShape, &wQ, NULL);

    tensor_t outT;
    float outData[12];
    size_t outDims[] = {(size_t)kBfpBwdBatch, (size_t)kBfpBwdInF};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outT, (uint8_t *)outData, &outShape, &outQ, NULL);

    matmulBfpTensors(&lossT, &wT, NULL, &outT);

    TEST_ASSERT_EQUAL_MEMORY(kBfpDxExpected, outT.data, kBfpDxExpected_len * sizeof(float));
}

/* PR3 kernel-reuse pin: Linear weightGrad = loss^T @ x with loss behind the
 * zero-copy transposeTensor view [outF, batch] -- the a-walk reads loss
 * storage strided by outF and crosses loss groups mid-reduction while x's
 * groups (groupSize 6 > inF) sometimes stay put, so the a-side of the
 * either-boundary fold clause carries this test alone. */
void testMatmulBfpWeightGradTransposedLossViewMatchesGold(void) {
    tensor_t lossT;
    size_t lossDims[] = {(size_t)kBfpBwdBatch, (size_t)kBfpBwdOutF};
    size_t lossOrder[] = {0, 1};
    shape_t lossShape;
    setShape(&lossShape, lossDims, 2, lossOrder);
    uint8_t lossExps[sizeof(kBfpLossExponents)];
    memcpy(lossExps, kBfpLossExponents, sizeof(lossExps));
    bfpQConfig_t lossQC = {.exponents = lossExps,
                           .numGroups = (size_t)kBfpLossNumGroups,
                           .groupSize = (size_t)kBfpLossGroupSize,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpLossMantissaBits,
                           .exponentBits = (uint8_t)kBfpLossExponentBits};
    quantization_t lossQ;
    initBfpQuantization(&lossQC, &lossQ);
    setTensorValues(&lossT, (uint8_t *)kBfpLossCodes, &lossShape, &lossQ, NULL);
    transposeTensor(&lossT, 0, 1); /* zero-copy loss^T view [outF, batch] */

    tensor_t xT;
    size_t xDims[] = {(size_t)kBfpBwdBatch, (size_t)kBfpBwdInF};
    size_t xOrder[] = {0, 1};
    shape_t xShape;
    setShape(&xShape, xDims, 2, xOrder);
    uint8_t xExps[sizeof(kBfpXbExponents)];
    memcpy(xExps, kBfpXbExponents, sizeof(xExps));
    bfpQConfig_t xQC = {.exponents = xExps,
                        .numGroups = (size_t)kBfpXbNumGroups,
                        .groupSize = (size_t)kBfpXbGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpXbMantissaBits,
                        .exponentBits = (uint8_t)kBfpXbExponentBits};
    quantization_t xQ;
    initBfpQuantization(&xQC, &xQ);
    setTensorValues(&xT, (uint8_t *)kBfpXbCodes, &xShape, &xQ, NULL);

    tensor_t outT;
    float outData[16];
    size_t outDims[] = {(size_t)kBfpBwdOutF, (size_t)kBfpBwdInF};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outT, (uint8_t *)outData, &outShape, &outQ, NULL);

    matmulBfpTensors(&lossT, &xT, NULL, &outT);
    transposeTensor(&lossT, 0, 1); /* restore the storage view */

    TEST_ASSERT_EQUAL_MEMORY(kBfpWgExpected, outT.data, kBfpWgExpected_len * sizeof(float));
}

/* PR3 dx power-of-two twin (the power-of-two bit-identity twin in the backward, BFP D8 amendment):
 * the dx sibling of testMatmulBfpPowerOfTwoBitIdenticalToGroupedSym above (see its comment for the
 * exactness argument), in the RAW-weight-storage orientation of
 * testMatmulGroupedDxEqualScalesBitIdenticalToScalar (same mantissas, loss per-tensor stored 126
 * <-> SYM aScale 0.5f, W grouped all-125 <-> weightGroups scales 0.25f): the BFP float output must
 * be BIT-IDENTICAL to the grouped-SYM path's dequantized output. */
void testMatmulBfpDxPowerOfTwoBitIdenticalToGroupedSym(void) {
    size_t aDims[] = {(size_t)kDxPerChannelOutRows, (size_t)kDxPerChannelReduceLen};
    size_t aOrder[] = {0, 1};
    shape_t aShape;
    setShape(&aShape, aDims, 2, aOrder);
    size_t bDims[] = {(size_t)kDxPerChannelReduceLen, (size_t)kDxPerChannelOutCols};
    size_t bOrder[] = {0, 1}; /* RAW storage view: reduction axis strided by outCols */
    shape_t bShape;
    setShape(&bShape, bDims, 2, bOrder);
    size_t outDims[] = {(size_t)kDxPerChannelOutRows, (size_t)kDxPerChannelOutCols};
    size_t outOrder[] = {0, 1};
    shape_t outShape;
    setShape(&outShape, outDims, 2, outOrder);

    tensor_t aBfpTensor;
    uint8_t aExponents[] = {126}; /* 2^(126-127) == 0.5f */
    bfpQConfig_t aBfpQC = {.exponents = aExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 8,
                           .exponentBits = 8};
    quantization_t aBfpQ;
    initBfpQuantization(&aBfpQC, &aBfpQ);
    setTensorValues(&aBfpTensor, (uint8_t *)kDxPerChannelLossMantissas, &aShape, &aBfpQ, NULL);

    tensor_t bBfpTensor;
    uint8_t bExponents[] = {125, 125, 125}; /* 2^(125-127) == 0.25f */
    bfpQConfig_t bBfpQC = {.exponents = bExponents,
                           .numGroups = 3,
                           .groupSize = 6,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 8,
                           .exponentBits = 8};
    quantization_t bBfpQ;
    initBfpQuantization(&bBfpQC, &bBfpQ);
    setTensorValues(&bBfpTensor, (uint8_t *)kDxPerChannelWMantissas, &bShape, &bBfpQ, NULL);

    tensor_t outBfpTensor;
    float outBfpData[12];
    quantization_t outBfpQ;
    initFloat32Quantization(&outBfpQ);
    setTensorValues(&outBfpTensor, (uint8_t *)outBfpData, &outShape, &outBfpQ, NULL);

    matmulBfpTensors(&aBfpTensor, &bBfpTensor, NULL, &outBfpTensor);

    tensor_t aSymTensor;
    symInt32QConfig_t aSymQC;
    initSymInt32QConfig(HALF_AWAY, &aSymQC);
    aSymQC.scale = 0.5f;
    quantization_t aSymQ;
    initSymInt32Quantization(&aSymQC, &aSymQ);
    setTensorValues(&aSymTensor, (uint8_t *)kDxPerChannelLossMantissas, &aShape, &aSymQ, NULL);

    tensor_t bSymTensor;
    symInt32QConfig_t bSymQC;
    initSymInt32QConfig(HALF_AWAY, &bSymQC);
    bSymQC.scale = 1.0f;
    quantization_t bSymQ;
    initSymInt32Quantization(&bSymQC, &bSymQ);
    setTensorValues(&bSymTensor, (uint8_t *)kDxPerChannelWMantissas, &bShape, &bSymQ, NULL);

    float scales[3] = {0.25f, 0.25f, 0.25f};
    symQConfig_t weightGroups = {
        .scales = scales, .numGroups = 3, .groupSize = 6, .qBits = 8, .roundingMode = HALF_AWAY};

    tensor_t outSymTensor;
    int32_t outSymData[12];
    symInt32QConfig_t outSymQC;
    initSymInt32QConfig(HALF_AWAY, &outSymQC);
    quantization_t outSymQ;
    initSymInt32Quantization(&outSymQC, &outSymQ);
    setTensorValues(&outSymTensor, (uint8_t *)outSymData, &outShape, &outSymQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    matmulSymInt32TensorsGroupedWeight(&aSymTensor, &bSymTensor, NULL, &outSymTensor,
                                       &groupedWeight, NULL);

    for (size_t i = 0; i < 12; i++) {
        float symDequant = (float)outSymData[i] * outSymQC.scale;
        TEST_ASSERT_EQUAL_MEMORY(&symDequant, &outBfpData[i], sizeof(float));
    }
}

/* ---- Integer-only forward (arithmetic-bfp.md §13): matmulBfpRawTensors ----
 *
 * The Linear forward's integer fold, operands in the funnel's unpacked-BFP
 * scratch form, output BFP_RAW. Every reference below is computed HERE from
 * the §13 definition in int64 (division + remainder for HALF_AWAY, the guard
 * by its defining inequality), never from the kernel's bit forms. `a` is
 * [rows, k] row-major, the weight is STORED [cols, k] and handed to the
 * kernel as the transposed [k, cols] view (Linear.c's transpose dance). */

#define LIN_RAW_MAX_K 2400
#define LIN_RAW_MAX_OUT 16

typedef struct linRawCase {
    size_t rows, cols, k;
    const int32_t *a;
    const uint8_t *aExps;
    size_t aGroups, aGs;
    uint8_t ma;
    const int32_t *w;
    const uint8_t *wExps;
    size_t wGroups, wGs;
    uint8_t mw;
    const int32_t *bias; /* NULL = no bias */
    const uint8_t *biasExps;
    size_t biasGroups, biasGs;
    uint8_t mBias;
    size_t rawGroups, rawGs;
} linRawCase_t;

typedef struct linRawRef {
    int32_t mant[LIN_RAW_MAX_OUT];
    int16_t exps[LIN_RAW_MAX_OUT];
    double exact[LIN_RAW_MAX_OUT];
    double bound[LIN_RAW_MAX_OUT]; /* 1/2 (S + 1) 2^E_acc, §13 item 7 */
    int g;
    size_t rounded; /* right-shift folds that dropped a nonzero remainder */
    size_t splits;  /* segments closed by the segment limit */
} linRawRef_t;

static int linUnbiased(const uint8_t *exps, size_t gs, size_t idx) {
    return (int)exps[gs == 0 ? 0 : idx / gs] - 127;
}

static int64_t refShrHalfAway(int64_t v, int k) {
    if (k == 0) {
        return v;
    }
    if (k > 40) {
        return 0; /* |v| < 2^32 <= 2^(k-1): below one half */
    }
    int64_t d = (int64_t)1 << k;
    int64_t mag = v < 0 ? -v : v;
    int64_t q = mag / d;
    if (2 * (mag % d) >= d) {
        q++;
    }
    return v < 0 ? -q : q;
}

static int64_t refAlign(int64_t v, int k, size_t *rounded) {
    if (k < 0) {
        int64_t r = v * ((int64_t)1 << -k);
        TEST_ASSERT_TRUE(r >= INT32_MIN && r <= INT32_MAX);
        return r;
    }
    int64_t r = refShrHalfAway(v, k);
    if (k > 0 && (k > 40 || r * ((int64_t)1 << k) != v)) {
        (*rounded)++;
    }
    return r;
}

/* Smallest g in [-30, 31] with (N 2^(ma+mb-2) + 2^(mBias-1)) 2^-g + (N + 1) <=
 * INT32_MAX, compared without shifting the total left (it is an integer, so
 * total 2^-g <= R iff total <= floor(R 2^g)). */
static int refGuard(size_t n, uint8_t ma, uint8_t mb, uint8_t mBias) {
    int64_t total =
        (int64_t)n * ((int64_t)1 << (ma + mb - 2)) + (mBias > 0 ? ((int64_t)1 << (mBias - 1)) : 0);
    int64_t room = (int64_t)INT32_MAX - (int64_t)n - 1;
    for (int g = -30; g <= 31; g++) {
        int64_t lim = g <= 0 ? room >> -g : room * ((int64_t)1 << g);
        if (total <= lim) {
            return g;
        }
    }
    TEST_FAIL_MESSAGE("no guard holds");
    return 0;
}

static void refLinearRaw(const linRawCase_t *c, linRawRef_t *r) {
    size_t nOut = c->rows * c->cols;
    TEST_ASSERT_TRUE(nOut <= LIN_RAW_MAX_OUT);
    unsigned shift = (unsigned)c->ma + c->mw - 2u;
    size_t limit = shift >= 31u ? 1u : (size_t)(INT32_MAX >> shift);
    r->g = refGuard(c->k, c->ma, c->mw, c->bias != NULL ? c->mBias : 0);
    r->rounded = 0;
    r->splits = 0;
    size_t gsz = c->rawGs == 0 ? nOut : c->rawGs;
    for (size_t t = 0; t * gsz < nOut; t++) {
        int top = INT32_MIN;
        for (size_t o = t * gsz; o < (t + 1) * gsz; o++) {
            size_t row = o / c->cols, col = o % c->cols;
            int aMax = INT32_MIN, wMax = INT32_MIN;
            for (size_t i = 0; i < c->k; i++) {
                int ea = linUnbiased(c->aExps, c->aGs, row * c->k + i);
                int ew = linUnbiased(c->wExps, c->wGs, col * c->k + i);
                aMax = ea > aMax ? ea : aMax;
                wMax = ew > wMax ? ew : wMax;
            }
            int e = aMax + wMax;
            if (c->bias != NULL) {
                int eb = linUnbiased(c->biasExps, c->biasGs, col);
                e = eb > e ? eb : e;
            }
            top = e > top ? e : top;
        }
        r->exps[t] = (int16_t)(top + r->g);
    }
    for (size_t o = 0; o < nOut; o++) {
        size_t row = o / c->cols, col = o % c->cols;
        int eAcc = r->exps[o / gsz];
        int64_t acc = 0;
        double exact = 0.0;
        size_t segments = 0;
        if (c->bias != NULL) {
            int eb = linUnbiased(c->biasExps, c->biasGs, col);
            acc = refAlign(c->bias[col], eAcc - eb, &r->rounded);
            exact = ldexp((double)c->bias[col], eb);
            segments = 1;
        }
        int64_t partial = 0;
        size_t count = 0;
        size_t curA = SIZE_MAX, curW = SIZE_MAX;
        int curE = 0;
        for (size_t i = 0; i < c->k; i++) {
            size_t ia = row * c->k + i, iw = col * c->k + i;
            size_t ga = c->aGs == 0 ? 0 : ia / c->aGs;
            size_t gw = c->wGs == 0 ? 0 : iw / c->wGs;
            if (curA != SIZE_MAX && (ga != curA || gw != curW || count == limit)) {
                if (ga == curA && gw == curW) {
                    r->splits++;
                }
                acc += refAlign(partial, eAcc - curE, &r->rounded);
                TEST_ASSERT_TRUE(acc >= INT32_MIN && acc <= INT32_MAX);
                segments++;
                partial = 0;
                count = 0;
            }
            curA = ga;
            curW = gw;
            curE = linUnbiased(c->aExps, c->aGs, ia) + linUnbiased(c->wExps, c->wGs, iw);
            int64_t prod = (int64_t)c->a[ia] * c->w[iw];
            partial += prod;
            TEST_ASSERT_TRUE(partial >= INT32_MIN && partial <= INT32_MAX);
            count++;
            exact += ldexp((double)prod, curE);
        }
        if (curA != SIZE_MAX) {
            acc += refAlign(partial, eAcc - curE, &r->rounded);
            segments++;
        }
        TEST_ASSERT_TRUE(acc >= INT32_MIN && acc <= INT32_MAX);
        r->mant[o] = (int32_t)acc;
        r->exact[o] = exact;
        r->bound[o] = 0.5 * (double)(segments + 1) * ldexp(1.0, eAcc);
    }
}

/* The kernel's operands for one case: a [rows, k] identity order; the weight
 * stored [cols, k] under the transposed view (order {1, 0}), exactly what
 * linearForwardBfpRaw hands the kernel. */
typedef struct linRawOperands {
    bfpScratchFixture_t a, w, bias;
    bfpRawFixture_t out;
} linRawOperands_t;

static void linRawBuild(const linRawCase_t *c, linRawOperands_t *ops) {
    size_t aDims[] = {c->rows, c->k};
    bfpScratchInit(&ops->a, 2, aDims, c->a, c->aExps, c->aGroups, c->aGs, c->ma, 8);
    size_t wDims[] = {c->cols, c->k};
    bfpScratchInit(&ops->w, 2, wDims, c->w, c->wExps, c->wGroups, c->wGs, c->mw, 8);
    ops->w.order[0] = 1;
    ops->w.order[1] = 0;
    if (c->bias != NULL) {
        size_t biasDims[] = {c->cols};
        bfpScratchInit(&ops->bias, 1, biasDims, c->bias, c->biasExps, c->biasGroups, c->biasGs,
                       c->mBias, 8);
    }
    size_t outDims[] = {c->rows, c->cols};
    bfpRawInit(&ops->out, 2, outDims, c->rawGroups, c->rawGs);
}

static void runLinRawCase(const linRawCase_t *c, linRawRef_t *ref, linRawOperands_t *ops) {
    refLinearRaw(c, ref);
    linRawBuild(c, ops);
    matmulBfpRawTensors(&ops->a.tensor, &ops->w.tensor, c->bias != NULL ? &ops->bias.tensor : NULL,
                        &ops->out.tensor);
    size_t nOut = c->rows * c->cols;
    assertRawEquals(&ops->out, ref->mant, nOut, ref->exps, c->rawGroups);
    size_t gsz = c->rawGs == 0 ? nOut : c->rawGs;
    for (size_t o = 0; o < nOut; o++) {
        double got = ldexp((double)ops->out.mant[o], ops->out.exps[o / gsz]);
        TEST_ASSERT_TRUE_MESSAGE(fabs(got - ref->exact[o]) <= ref->bound[o],
                                 "raw value outside the 1/2 (S+1) 2^E_acc fold bound");
    }
}

/* Grouped operands whose groups straddle rows (a gs 4 over k = 6, the weight
 * gs 9 over its [3, 6] storage), per-column bias groups with one exponent
 * ABOVE every product (it sets E_top for column 0), and raw target groups of
 * 2 that straddle the two output rows. The input exponents span 22 binades
 * against g = -14, so some segments are left-shifted exactly and others
 * rounded HALF_AWAY. */
void testMatmulBfpRawMatchesTheFoldReferenceOnStraddlingGroups(void) {
    static const int32_t a[] = {100, -128, 127, -3, 55, -77, 64, -1, 90, -100, 33, 7};
    static const uint8_t aExps[] = {127 + 12, 127 - 10, 127 + 3};
    static const int32_t w[] = {-90, 17, 127, -128, 5,    60,  31, -45, 99,
                                -7,  88, -66, 12,   -128, 127, 1,  -2,  43};
    static const uint8_t wExps[] = {127 - 4, 127 + 9};
    static const int32_t bias[] = {-101, 77, 3};
    static const uint8_t biasExps[] = {127 + 25, 127 - 30, 127};
    linRawCase_t c = {.rows = 2,
                      .cols = 3,
                      .k = 6,
                      .a = a,
                      .aExps = aExps,
                      .aGroups = 3,
                      .aGs = 4,
                      .ma = 8,
                      .w = w,
                      .wExps = wExps,
                      .wGroups = 2,
                      .wGs = 9,
                      .mw = 8,
                      .bias = bias,
                      .biasExps = biasExps,
                      .biasGroups = 3,
                      .biasGs = 1,
                      .mBias = 8,
                      .rawGroups = 3,
                      .rawGs = 2};
    linRawRef_t ref;
    linRawOperands_t ops;
    runLinRawCase(&c, &ref, &ops);
    TEST_ASSERT_EQUAL_INT(-14, ref.g);
    TEST_ASSERT_TRUE_MESSAGE(ref.rounded > 0, "fixture too weak: no fold rounds");
    /* column 0's bias exponent (+25) sets E_top of both groups holding a
     * column-0 output; group 2 is row 1 x weight row 2 (3 + 9). */
    TEST_ASSERT_EQUAL_INT16(25 - 14, ops.out.exps[0]);
    TEST_ASSERT_EQUAL_INT16(25 - 14, ops.out.exps[1]);
    TEST_ASSERT_EQUAL_INT16(12 - 14, ops.out.exps[2]);

    /* Same operands without bias, onto the per-tensor raw. */
    c.bias = NULL;
    c.rawGroups = 1;
    c.rawGs = 0;
    runLinRawCase(&c, &ref, &ops);
    TEST_ASSERT_EQUAL_INT(-14, ref.g);
    TEST_ASSERT_TRUE_MESSAGE(ref.rounded > 0, "fixture too weak: no fold rounds");
}

/* Worst-case headroom: every code at a grid edge, at the proof's limits, one
 * output of a [1, k] x [k, 1] reduction, every exponent unbiased 0. */
static int32_t gLinWorstA[LIN_RAW_MAX_K];
static int32_t gLinWorstW[LIN_RAW_MAX_K];

static void runLinWorst(size_t k, int32_t aCode, uint8_t ma, int32_t wCode, uint8_t mw,
                        const int32_t *biasCode, uint8_t mBias, uint8_t biasStored,
                        linRawRef_t *ref) {
    TEST_ASSERT_TRUE(k <= LIN_RAW_MAX_K);
    for (size_t i = 0; i < k; i++) {
        gLinWorstA[i] = aCode;
        gLinWorstW[i] = wCode;
    }
    static const uint8_t unit[] = {127};
    uint8_t biasExps[] = {biasStored};
    linRawCase_t c = {.rows = 1,
                      .cols = 1,
                      .k = k,
                      .a = gLinWorstA,
                      .aExps = unit,
                      .aGroups = 1,
                      .aGs = 0,
                      .ma = ma,
                      .w = gLinWorstW,
                      .wExps = unit,
                      .wGroups = 1,
                      .wGs = 0,
                      .mw = mw,
                      .bias = biasCode,
                      .biasExps = biasExps,
                      .biasGroups = 1,
                      .biasGs = 0,
                      .mBias = mBias,
                      .rawGroups = 1,
                      .rawGs = 0};
    linRawOperands_t ops;
    runLinRawCase(&c, ref, &ops);
}

/* m16 x m16, N = 80: g = 6 and the segment limit is 1 (two 2^30 products
 * already overflow), so every product is its own segment; floor x floor and
 * floor x ceiling. */
void testMatmulBfpRawWorstCaseM16xM16AllCodesAtTheEdges(void) {
    linRawRef_t ref;
    runLinWorst(80, -32768, 16, -32768, 16, NULL, 0, 0, &ref);
    TEST_ASSERT_EQUAL_INT(6, ref.g);
    TEST_ASSERT_EQUAL_INT32(80 << 24, ref.mant[0]); /* 80 * 2^30 exactly */
    TEST_ASSERT_EQUAL_size_t(79, ref.splits);
    runLinWorst(80, -32768, 16, 32767, 16, NULL, 0, 0, &ref);
    TEST_ASSERT_EQUAL_INT(6, ref.g);
}

/* m16 x m8, N = 2400 (g = 3): the run splits at 511 products (four full
 * segments + 356) instead of failing fast as the backward kernel does. */
void testMatmulBfpRawWorstCaseM16xM8SplitsAtTheSegmentLimit(void) {
    linRawRef_t ref;
    runLinWorst(2400, -32768, 16, -128, 8, NULL, 0, 0, &ref);
    TEST_ASSERT_EQUAL_INT(3, ref.g);
    TEST_ASSERT_EQUAL_size_t(4, ref.splits);
    TEST_ASSERT_EQUAL_INT32(2400 << 19, ref.mant[0]); /* 2400 * 2^22 / 2^3 */
    runLinWorst(2400, 32767, 16, -128, 8, NULL, 0, 0, &ref);
}

/* m16 x m16, N = 4, a m16 bias at its floor (g = 2); and a m8 bias whose
 * exponent (+20) is ABOVE the products' (m8 x m8, N = 80, g = -10): E_top is
 * the bias exponent and the products shift right by 10 exactly. */
void testMatmulBfpRawWorstCaseBiasAtTheFloorAndAboveTheProducts(void) {
    linRawRef_t ref;
    int32_t biasFloor = -32768;
    runLinWorst(4, -32768, 16, -32768, 16, &biasFloor, 16, 127, &ref);
    TEST_ASSERT_EQUAL_INT(2, ref.g);
    TEST_ASSERT_EQUAL_INT64((int64_t)4 * ((int64_t)1 << 30) - 32768,
                            (int64_t)ref.mant[0] * ((int64_t)1 << 2));

    int32_t biasHigh = -128;
    runLinWorst(80, -128, 8, -128, 8, &biasHigh, 8, 127 + 20, &ref);
    TEST_ASSERT_EQUAL_INT(-10, ref.g);
    TEST_ASSERT_EQUAL_INT16(10, ref.exps[0]);
    TEST_ASSERT_EQUAL_INT32(-129792, ref.mant[0]);
}

/* A weight view whose reduction axis is NOT the physically innermost one
 * (identity-ordered [k, cols] storage): a weight column is no contiguous run,
 * so E_top uses the weight's tensor-wide max exponent -- coarser, still
 * sound. Every fold here is an exact left shift, so the raw equals the int64
 * sum and E_acc = rowmax(a) + max(w) + g. */
void testMatmulBfpRawStridedWeightViewUsesTheTensorWideMax(void) {
    static const int32_t a[] = {3, -5, 7, 11};
    static const uint8_t aExps[] = {127 - 2};
    /* storage [k = 4, cols = 2], gs 2: storage row i is group i */
    static const int32_t w[] = {1, -2, 4, 8, -16, 32, 64, -128};
    static const uint8_t wExps[] = {127 + 1, 127 - 6, 127 + 5, 127};
    bfpScratchFixture_t aF, wF;
    size_t aDims[] = {1, 4};
    bfpScratchInit(&aF, 2, aDims, a, aExps, 1, 0, 8, 8);
    size_t wDims[] = {4, 2};
    bfpScratchInit(&wF, 2, wDims, w, wExps, 4, 2, 8, 8);
    bfpRawFixture_t out;
    size_t outDims[] = {1, 2};
    bfpRawInit(&out, 2, outDims, 2, 1);

    matmulBfpRawTensors(&aF.tensor, &wF.tensor, NULL, &out.tensor);

    /* one raw group per output: a column read as a contiguous run would cover
     * storage groups {0, 1} (max +1) for column 0 instead of the +5 it visits */
    int eAcc = -2 + 5 + refGuard(4, 8, 8, 0);
    TEST_ASSERT_EQUAL_INT16((int16_t)eAcc, out.exps[0]);
    TEST_ASSERT_EQUAL_INT16((int16_t)eAcc, out.exps[1]);
    for (size_t col = 0; col < 2; col++) {
        int64_t exactScaled = 0; /* in units of 2^eAcc; every shift is left */
        for (size_t i = 0; i < 4; i++) {
            int e = -2 + ((int)wExps[i] - 127);
            exactScaled += (int64_t)a[i] * w[i * 2 + col] * ((int64_t)1 << (e - eAcc));
        }
        TEST_ASSERT_EQUAL_INT64(exactScaled, (int64_t)out.mant[col]);
    }
}

/* Exact power-of-two data (the matmulBfpTensors gold fixture): the raw's
 * dequant (float)mant * 2^E is bit-identical to the float kernel's
 * rounding-free fold, bias included. */
void testMatmulBfpRawPowerOfTwoDequantMatchesFloatKernel(void) {
    bfpScratchFixture_t a, b, bias;
    size_t aDims[] = {(size_t)kBfpOutRows, (size_t)kBfpReduceLen};
    bfpScratchInit(&a, 2, aDims, kBfpACodes, kBfpAExponents, (size_t)kBfpANumGroups,
                   (size_t)kBfpAGroupSize, (uint8_t)kBfpAMantissaBits, (uint8_t)kBfpAExponentBits);
    size_t bDims[] = {(size_t)kBfpOutCols, (size_t)kBfpReduceLen};
    bfpScratchInit(&b, 2, bDims, kBfpBCodes, kBfpBExponents, (size_t)kBfpBNumGroups,
                   (size_t)kBfpBGroupSize, (uint8_t)kBfpBMantissaBits, (uint8_t)kBfpBExponentBits);
    b.order[0] = 1;
    b.order[1] = 0;
    size_t biasDims[] = {(size_t)kBfpOutCols};
    bfpScratchInit(&bias, 1, biasDims, kBfpBiasCodes, kBfpBiasExponents, 1, 0,
                   (uint8_t)kBfpBiasMantissaBits, (uint8_t)kBfpBiasExponentBits);
    bfpRawFixture_t out;
    size_t outDims[] = {(size_t)kBfpOutRows, (size_t)kBfpOutCols};
    bfpRawInit(&out, 2, outDims, 1, 0);

    matmulBfpRawTensors(&a.tensor, &b.tensor, &bias.tensor, &out.tensor);

    float deq[6];
    TEST_ASSERT_TRUE(kBfpMatmulExpected_len <= 6);
    for (size_t i = 0; i < kBfpMatmulExpected_len; i++) {
        deq[i] = ldexpf((float)out.mant[i], out.exps[0]);
    }
    TEST_ASSERT_EQUAL_MEMORY(kBfpMatmulExpected, deq, kBfpMatmulExpected_len * sizeof(float));
}

static bfpScratchFixture_t gLinDieA, gLinDieW;
static bfpRawFixture_t gLinDieRaw;
static void linRawDieSetup(void) {
    static const int32_t codes[] = {1, 2, 3, 4};
    static const uint8_t unit[] = {127};
    size_t aDims[] = {1, 2};
    bfpScratchInit(&gLinDieA, 2, aDims, codes, unit, 1, 0, 8, 8);
    size_t wDims[] = {2, 2};
    bfpScratchInit(&gLinDieW, 2, wDims, codes, unit, 1, 0, 8, 8);
    gLinDieW.order[0] = 1;
    gLinDieW.order[1] = 0;
}
static void linRawIntoFloatBody(void) {
    linRawDieSetup();
    float outData[2];
    tensor_t outT;
    shape_t outShape;
    size_t outDims[] = {1, 2};
    size_t outOrder[] = {0, 1};
    setShape(&outShape, outDims, 2, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outT, (uint8_t *)outData, &outShape, &outQ, NULL);
    matmulBfpRawTensors(&gLinDieA.tensor, &gLinDieW.tensor, NULL, &outT);
}
static void linRawBadGeometryBody(void) {
    linRawDieSetup();
    size_t outDims[] = {1, 2};
    bfpRawInit(&gLinDieRaw, 2, outDims, 3, 1); /* 3 groups of 1 over 2 outputs */
    matmulBfpRawTensors(&gLinDieA.tensor, &gLinDieW.tensor, NULL, &gLinDieRaw.tensor);
}

void testMatmulBfpRawRejectsANonRawOutputAndBadRawGeometry(void) {
    ASSERT_EXITS_WITH_FAILURE(linRawIntoFloatBody());
    ASSERT_EXITS_WITH_FAILURE(linRawBadGeometryBody());
}

void setUp() {}
void tearDown() {}

/* ---- SYM int32 accumulator guard (SymAccumGuard.h) ---------------------- */

#define GUARD_MAXN 1024
static int32_t gGuardA[GUARD_MAXN];
static int32_t gGuardB[GUARD_MAXN];

static void guardFill(size_t n, int32_t aCode, int32_t bCode, bool bAlternates) {
    for (size_t i = 0; i < n; i++) {
        gGuardA[i] = aCode;
        gGuardB[i] = (bAlternates && (i % 2 == 1)) ? -bCode : bCode;
    }
}

/* a:[1,n] . b:[n,1] -> one SYM_INT32 output, all operands int12 (qMaxBits 12),
 * scales 1; biasCode != NULL adds a SYM_INT32 bias of that code (scale 1). */
static int32_t guardDotBits(size_t n, uint8_t aBits, uint8_t bBits, const int32_t *biasCode) {
    shape_t aS, bS, oS, biasS;
    size_t aDims[] = {1, n}, bDims[] = {n, 1}, oDims[] = {1, 1}, biasDims[] = {1};
    size_t order[] = {0, 1};
    setShape(&aS, aDims, 2, order);
    setShape(&bS, bDims, 2, order);
    setShape(&oS, oDims, 2, order);
    setShape(&biasS, biasDims, 1, order);
    symInt32QConfig_t aQC, bQC, oQC, biasQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &aQC, aBits);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &bQC, bBits);
    initSymInt32QConfig(HALF_AWAY, &oQC);
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    quantization_t aQ, bQ, oQ, biasQ;
    initSymInt32Quantization(&aQC, &aQ);
    initSymInt32Quantization(&bQC, &bQ);
    initSymInt32Quantization(&oQC, &oQ);
    initSymInt32Quantization(&biasQC, &biasQ);
    int32_t out = 0;
    int32_t biasData = biasCode ? *biasCode : 0;
    tensor_t a, b, o, bias;
    setTensorValues(&a, (uint8_t *)gGuardA, &aS, &aQ, NULL);
    setTensorValues(&b, (uint8_t *)gGuardB, &bS, &bQ, NULL);
    setTensorValues(&o, (uint8_t *)&out, &oS, &oQ, NULL);
    setTensorValues(&bias, (uint8_t *)&biasData, &biasS, &biasQ, NULL);
    if (biasCode) {
        matmulSymInt32TensorsWithBias(&a, &b, &o, &bias, NULL);
    } else {
        matmulSymInt32Tensors(&a, &b, &o);
    }
    return out;
}

static int32_t guardDot(size_t n, const int32_t *biasCode) {
    return guardDotBits(n, 12, 12, biasCode);
}

static int32_t guardExpected(size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (int64_t)gGuardA[i] * gGuardB[i];
    }
    return (int32_t)sum;
}

void testMatmulSymGuardProvenSafeBoundaryN511RunsExact(void) {
    /* -2048 x -2048 is reachable (fixed-scale requant saturation); 511 * 2^22 fits */
    guardFill(511, -2048, -2048, false);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, guardDot(511, NULL));
}

void testMatmulSymGuardDiesAtFirstWrappingNFullScaleNegativeCodes(void) {
    guardFill(512, -2048, -2048, false); /* 512 * 2^22 = 2^31: first wrapping N */
    ASSERT_EXITS_WITH_OUTPUT(1, "matmul SYM_INT32", (void)guardDot(512, NULL));
}

void testMatmulSymGuardDiesAtN513FullScalePositiveCodes(void) {
    guardFill(513, 2047, 2047, false); /* the original sign-flip: absmax-encoded +-2047 */
    ASSERT_EXITS_WITH_OUTPUT(1, "N=513", (void)guardDot(513, NULL));
}

void testMatmulSymGuardDiesOnNegativeOverflow(void) {
    guardFill(513, 2047, -2048, false); /* 513 * -4192256 < INT32_MIN */
    ASSERT_EXITS_WITH_OUTPUT(1, "matmul SYM_INT32", (void)guardDot(513, NULL));
}

void testMatmulSymGuardCheckedPathStaysExactWhenValuesFit(void) {
    /* N = 601 > 511: the bound cannot prove it, so the checked loop runs; the
     * alternating b keeps every partial sum tiny, so no false positive. */
    guardFill(601, 2047, 2047, true);
    TEST_ASSERT_EQUAL_INT32(guardExpected(601), guardDot(601, NULL));
    TEST_ASSERT_EQUAL_INT32(2047 * 2047, guardDot(601, NULL));
}

void testMatmulSymGuardBiasSeedThatTipsTheBoundIsCounted(void) {
    int32_t seed = 4194304; /* 2^22: 511 products + this seed = 2^31 */
    guardFill(511, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "max|seed|=4194304", (void)guardDot(511, &seed));
    int32_t small = 4194303; /* one short of the tip: exact */
    TEST_ASSERT_EQUAL_INT32(511 * 4194304 + 4194303, guardDot(511, &small));
}

/* a:[1,n] . grouped b (storage [1,n], numGroups x groupSize), scales all 1;
 * `domain` names the weight's code domain (a stored SYM or ASYM weight). */
static int32_t guardGroupedDotIn(size_t n, size_t numGroups, uint8_t weightQBits,
                                 symCodeDomain_t domain) {
    shape_t aS, bS, oS;
    size_t aDims[] = {1, n}, bDims[] = {1, n}, oDims[] = {1, 1};
    size_t aOrder[] = {0, 1}, bOrder[] = {1, 0};
    setShape(&aS, aDims, 2, aOrder);
    setShape(&bS, bDims, 2, bOrder);
    setShape(&oS, oDims, 2, aOrder);
    symInt32QConfig_t aQC, bQC, oQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    initSymInt32QConfig(HALF_AWAY, &bQC);
    initSymInt32QConfig(HALF_AWAY, &oQC);
    quantization_t aQ, bQ, oQ;
    initSymInt32Quantization(&aQC, &aQ);
    initSymInt32Quantization(&bQC, &bQ);
    initSymInt32Quantization(&oQC, &oQ);
    int32_t out = 0;
    tensor_t a, b, o;
    setTensorValues(&a, (uint8_t *)gGuardA, &aS, &aQ, NULL);
    setTensorValues(&b, (uint8_t *)gGuardB, &bS, &bQ, NULL);
    setTensorValues(&o, (uint8_t *)&out, &oS, &oQ, NULL);
    float scales[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    symQConfig_t groups = {.scales = scales,
                           .numGroups = numGroups,
                           .groupSize = n / numGroups,
                           .qBits = weightQBits,
                           .roundingMode = HALF_AWAY};
    symGroupedWeight_t groupedWeight = {.groups = &groups, .domain = domain};
    matmulSymInt32TensorsGroupedWeight(&a, &b, NULL, &o, &groupedWeight, NULL);
    return out;
}

static int32_t guardGroupedDotBits(size_t n, size_t numGroups, uint8_t weightQBits) {
    return guardGroupedDotIn(n, numGroups, weightQBits, SYM_CODES_TWOS_COMPLEMENT);
}

static int32_t guardGroupedDot(size_t n, size_t numGroups) {
    return guardGroupedDotBits(n, numGroups, 12);
}

void testMatmulGroupedGuardDiesOnRawGroupPartialOverflow(void) {
    guardFill(1024, -2048, -2048, false); /* 2 groups x 512 terms: 512 * 2^22 = 2^31 per group */
    ASSERT_EXITS_WITH_OUTPUT(1, "matmul SYM_INT32 grouped", (void)guardGroupedDot(1024, 2));
}

void testMatmulGroupedGuardDiesOnRescaledAccumulatorOverflow(void) {
    guardFill(1024, -2048, -2048,
              false); /* 4 x 256 terms: each partial 2^30 fits, two sum to 2^31 */
    ASSERT_EXITS_WITH_OUTPUT(1, "matmul SYM_INT32 grouped", (void)guardGroupedDot(1024, 4));
}

void testMatmulGroupedGuardCheckedPathStaysExactWhenValuesFit(void) {
    guardFill(600, 2047, 2047, true); /* 3 groups x 200: unprovable, values tiny */
    TEST_ASSERT_EQUAL_INT32(guardExpected(600), guardGroupedDot(600, 3));
}

/* 7 x 73 = 511 full-scale int12 terms: a stored SYM weight's exact range proves
 * them (511 * 2^22 fits), so the safe loop runs; the same codes under a stored
 * ASYM weight's image (+-4095 reachable) are unprovable, so the checked loop
 * runs -- both must stay exact. */
void testMatmulGroupedGuardFullScaleN511RunsExact(void) {
    guardFill(511, -2048, -2048, false);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, guardGroupedDot(511, 7));
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, guardGroupedDotIn(511, 7, 12, SYM_CODES_ASYM_IMAGE));
}

/* A ctx whose .domain was never set (zero) names no code domain: the kernel
 * must abort naming the tag before any accumulation, never run the SYM range
 * over what may be an ASYM image -- 511 terms of 2047 x 4095 would be proven
 * safe under the SYM range and wrap. */
void testMatmulGroupedZeroedCodeDomainAbortsBeforeTheLoop(void) {
    guardFill(511, 2047, 4095, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "symGroupedWeightRange: unknown code domain tag 0",
                             (void)guardGroupedDotIn(511, 7, 12, (symCodeDomain_t)0));
}

/* A stored grouped ASYM weight unpacks to code - zp[g], which reaches
 * +-(2^qBits - 1) = +-4095 at qBits 12 -- one code past the two's-complement
 * range a SYM weight of that width can hold. The kernel takes the range from
 * the ctx's code domain: 511 terms of 2047 x 4095 exceed INT32_MAX and abort
 * (the SYM range, 511 int12 terms, would have run them unchecked and wrapped);
 * 256 terms (2145911040) are proven at the exact image range and run exact
 * (the image's holding width 13 proves only 255). */
void testMatmulGroupedGuardAsymImageWeightKeepsItsFullRange(void) {
    guardFill(511, 2047, 4095, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "matmul SYM_INT32 grouped",
                             (void)guardGroupedDotIn(511, 7, 12, SYM_CODES_ASYM_IMAGE));
    guardFill(256, 2047, 4095, false);
    TEST_ASSERT_EQUAL_INT32(256 * 2047 * 4095, guardGroupedDotIn(256, 2, 12, SYM_CODES_ASYM_IMAGE));
}

/* Operands wider than the int12 default are not rejected; the guard is exact
 * for them: 256 terms of 4095 x 2047 = 2145911040 still fit int32, the 257th
 * term does not. */
void testMatmulSymGuardWiderOperandRunsExactWhenItFitsAndAbortsWhenItWraps(void) {
    guardFill(256, 4095, 2047, false);
    TEST_ASSERT_EQUAL_INT32(256 * 4095 * 2047, guardDotBits(256, 13, 12, NULL));
    guardFill(257, 4095, 2047, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (13, 12)", (void)guardDotBits(257, 13, 12, NULL));
}

/* A grouped weight wider than int12 used to exit at the validator; now it takes
 * the checked path (int12 x int16 products over 2 x 64 terms are unprovable).
 * Small codes stay exact (the rescale goes through float, so keep the partials
 * under 2^24); 2 x 64 terms of 2047 x 32767 wrap the raw group partial. */
void testMatmulGroupedGuardWideWeightRunsExactWhenItFitsAndAbortsWhenItWraps(void) {
    guardFill(32, 3, 32767, false);
    TEST_ASSERT_EQUAL_INT32(32 * 3 * 32767, guardGroupedDotBits(32, 2, 16));
    guardFill(128, 2047, 32767, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "matmul SYM_INT32 grouped (raw group partial)",
                             (void)guardGroupedDotBits(128, 2, 16));
}

/* Two groups of 40 terms of 1024 x 32767: each raw partial is 1342136320, above
 * the #189 seed guard's 1.07e9 reserve but inside int32. Group 2 negated with one
 * zero term: the sum fits (1024 * 32767) and every value survives the float
 * rescale exactly. Both positive: the running accumulator wraps. */
static void wideGroupedFill(bool secondGroupNegative) {
    guardFill(80, 1024, 32767, false);
    if (secondGroupNegative) {
        for (size_t i = 40; i < 80; i++) {
            gGuardB[i] = -32767;
        }
        gGuardB[79] = 0;
    }
}

void testMatmulGroupedWidePartialAboveSeedGuardRunsExactWhenSumFits(void) {
    wideGroupedFill(true);
    TEST_ASSERT_EQUAL_INT32(1024 * 32767, guardGroupedDotBits(80, 2, 16));
}

void testMatmulGroupedWidePartialSumThatWrapsHitsTheF9Guard(void) {
    wideGroupedFill(false);
    ASSERT_EXITS_WITH_OUTPUT(1, "matmul SYM_INT32 grouped (rescaled accumulator)",
                             (void)guardGroupedDotBits(80, 2, 16));
}

/* ---- #516: raw SYM outputs declare their accumulator width ------------- */

void testSymAccumResultBitsFollowsTheSiteBound(void) {
    /* exact site, int12 x int12: the positive corner N * 2^22 + seed decides */
    symAccumSite_t n255 = symAccumSiteInit("t", 255, 12, 12, 0);
    symAccumSite_t n256 = symAccumSiteInit("t", 256, 12, 12, 0);
    symAccumSite_t n512 = symAccumSiteInit("t", 512, 12, 12, 0); /* unprovable */
    symAccumSite_t seedOnly = symAccumSiteInit("t", 0, 12, 12, 1000);
    symAccumSite_t int32MinSeedOnly = symAccumSiteInit("t", 0, 32, 32, 2147483648ull);
    TEST_ASSERT_EQUAL_UINT8(31, symAccumResultBits(&n255));
    TEST_ASSERT_EQUAL_UINT8(32, symAccumResultBits(&n256));
    TEST_ASSERT_EQUAL_UINT8(32, symAccumResultBits(&n512));
    TEST_ASSERT_EQUAL_UINT8(11, symAccumResultBits(&seedOnly));
    TEST_ASSERT_EQUAL_UINT8(32, symAccumResultBits(&int32MinSeedOnly));
    /* rescaled site, (4, 5) bits, N = 8, seed 1023: the float slack
     * 8 * (2^7 + 1) + ceil(8 * 2^7 / 2^22) = 1033 tips 1024 + 1023 = 2047 (12 bits)
     * over to 2056 (13 bits). */
    symAccumSite_t exact = symAccumSiteInit("t", 8, 4, 5, 1023);
    symAccumSite_t rescaled = symAccumSiteInitRescaled("t", 8, 4, 5, 1023);
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&exact));
    TEST_ASSERT_EQUAL_UINT8(13, symAccumResultBits(&rescaled));
}

/* A 2-D SYM_INT32 wire over caller-owned codes. The struct owns the shape and
 * qConfig storage that setShape / setTensorValues keep raw pointers to, so it
 * is initialised in place and never copied. */
typedef struct symWire {
    tensor_t t;
    shape_t shape;
    size_t dims[2];
    size_t order[2];
    symInt32QConfig_t qc;
    quantization_t q;
} symWire_t;

static void symWireInit(symWire_t *w, int32_t *codes, size_t rows, size_t cols, uint8_t qMaxBits) {
    w->dims[0] = rows;
    w->dims[1] = cols;
    w->order[0] = 0;
    w->order[1] = 1;
    setShape(&w->shape, w->dims, 2, w->order);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &w->qc, qMaxBits);
    initSymInt32Quantization(&w->qc, &w->q);
    setTensorValues(&w->t, (uint8_t *)codes, &w->shape, &w->q, NULL);
}

/* Output wires start at a width no stamp test expects, so a missing stamp can
 * never pass as the right one. */
#define UNSTAMPED_WIDTH 5

void testMatmulSymStampsResultWidthInt12K1(void) {
    int32_t aCode = -2048, bCode = -2048, outCode = 0;
    symWire_t a, b, out;
    symWireInit(&a, &aCode, 1, 1, 12);
    symWireInit(&b, &bCode, 1, 1, 12);
    symWireInit(&out, &outCode, 1, 1, UNSTAMPED_WIDTH);
    matmulSymInt32Tensors(&a.t, &b.t, &out.t);
    TEST_ASSERT_EQUAL_INT32(4194304, outCode);
    TEST_ASSERT_EQUAL_UINT8(24, out.qc.qMaxBits); /* B = 2^11 * 2^11 = 2^22 -> 24 */
}

/* a:[1,k] . b:[k,1] of int12 codes 1: the stamp is the static bound k * 2^22,
 * never the data (the product is k). */
static uint8_t stampedWidthInt12(size_t k) {
    guardFill(k, 1, 1, false);
    int32_t outCode = 0;
    symWire_t a, b, out;
    symWireInit(&a, gGuardA, 1, k, 12);
    symWireInit(&b, gGuardB, k, 1, 12);
    symWireInit(&out, &outCode, 1, 1, UNSTAMPED_WIDTH);
    matmulSymInt32Tensors(&a.t, &b.t, &out.t);
    TEST_ASSERT_EQUAL_INT32((int32_t)k, outCode);
    return out.qc.qMaxBits;
}

void testMatmulSymStampsResultWidthBoundaries(void) {
    TEST_ASSERT_EQUAL_UINT8(31, stampedWidthInt12(255)); /* 255 * 2^22 < 2^30 */
    TEST_ASSERT_EQUAL_UINT8(32, stampedWidthInt12(256)); /* 2^30 is not < 2^30 */
    TEST_ASSERT_EQUAL_UINT8(32, stampedWidthInt12(511)); /* provable, still int32-wide */
    TEST_ASSERT_EQUAL_UINT8(32, stampedWidthInt12(512)); /* unprovable: raw int32 range */
}

void testMatmulSymWithBiasStampIncludesSeed(void) {
    int32_t aCode = 1, bCode = 1, biasCode = 1000, outCode = 0;
    symWire_t a, b, bias, out;
    symWireInit(&a, &aCode, 1, 1, 4);
    symWireInit(&b, &bCode, 1, 1, 4);
    symWireInit(&bias, &biasCode, 1, 1, 12);
    symWireInit(&out, &outCode, 1, 1, UNSTAMPED_WIDTH);
    matmulSymInt32TensorsWithBias(&a.t, &b.t, &out.t, NULL, NULL);
    TEST_ASSERT_EQUAL_INT32(1, outCode);
    TEST_ASSERT_EQUAL_UINT8(8, out.qc.qMaxBits); /* B = 2^3 * 2^3 = 64 -> 8 */
    /* the bias is rescaled 1:1 into the accumulator scale, so the seed is 1000 */
    symWireInit(&out, &outCode, 1, 1, UNSTAMPED_WIDTH);
    matmulSymInt32TensorsWithBias(&a.t, &b.t, &out.t, &bias.t, NULL);
    TEST_ASSERT_EQUAL_INT32(1001, outCode);
    TEST_ASSERT_EQUAL_UINT8(12, out.qc.qMaxBits); /* B = 64 + 1000 = 1064 -> 12 */
}

/* The bias seed's sign picks the side it widens: 3 int4 x int4 terms span
 * [-168, 192]; a bias of -1870 at the accumulator scale shifts that to
 * [-2038, -1678] -> 12, where max|seed| on both sides (2062) would say 13.
 * Four terms of codes 7 with bias -1: [-225, 255] -> 9 (seedless: 256 -> 10). */
void testMatmulSymWithBiasStampFollowsTheSeedSign(void) {
    guardFill(3, 1, 1, false);
    int32_t biasCode = -1870, outCode = 0;
    symWire_t a, b, bias, out;
    symWireInit(&a, gGuardA, 1, 3, 4);
    symWireInit(&b, gGuardB, 3, 1, 4);
    symWireInit(&bias, &biasCode, 1, 1, 12);
    symWireInit(&out, &outCode, 1, 1, UNSTAMPED_WIDTH);
    matmulSymInt32TensorsWithBias(&a.t, &b.t, &out.t, &bias.t, NULL);
    TEST_ASSERT_EQUAL_INT32(3 - 1870, outCode);
    TEST_ASSERT_EQUAL_UINT8(12, out.qc.qMaxBits);
    guardFill(4, 7, 7, false);
    biasCode = -1;
    symWireInit(&a, gGuardA, 1, 4, 4);
    symWireInit(&b, gGuardB, 4, 1, 4);
    symWireInit(&out, &outCode, 1, 1, UNSTAMPED_WIDTH);
    matmulSymInt32TensorsWithBias(&a.t, &b.t, &out.t, &bias.t, NULL);
    TEST_ASSERT_EQUAL_INT32(195, outCode);
    TEST_ASSERT_EQUAL_UINT8(9, out.qc.qMaxBits);
}

/* a:[1,8] int4 . grouped b (storage [1,8] as 2 groups x 4, qBits 4) + a bias
 * at the accumulator scale. The output is the RESCALED accumulator, so its
 * width is the rescaled site's over the weight's EXACT code range (Matmul.c,
 * SymGroupedWeight.h). A stored SYM int4 weight is [-8, 7]: one term spans
 * [-56, 64], 8 slacked terms [-457, 521] (slack 8 + ceil(512 / 2^22) = 9 per
 * side); bias 1023 -> 1544 -> 12, bias 500 -> 1021 -> 11, bias -1023 ->
 * 1480 on the negative side -> 12. A stored ASYM weight's image is [-15, 15]:
 * +-120 per term, 969 per side; bias 54 -> 1023 -> 11 and bias 500 -> 1469
 * -> 12. Counting the image one bit wide ([-16, 15], 1033 per side) would
 * say 13 for the SYM bias 1023 and 12 for the image bias 54. */
static uint8_t groupedInt4StampBits(symCodeDomain_t domain, int32_t biasCode, int32_t *outCode) {
    guardFill(8, 1, 1, false);
    symWire_t a, bias, out;
    symWireInit(&a, gGuardA, 1, 8, 4);
    symWireInit(&bias, &biasCode, 1, 1, 12);
    symWireInit(&out, outCode, 1, 1, UNSTAMPED_WIDTH);
    shape_t bS;
    size_t bDims[] = {1, 8}, bOrder[] = {1, 0}; /* storage [1,8] viewed as [8,1] */
    setShape(&bS, bDims, 2, bOrder);
    symInt32QConfig_t bQC;
    initSymInt32QConfig(HALF_AWAY, &bQC);
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    tensor_t b;
    setTensorValues(&b, (uint8_t *)gGuardB, &bS, &bQ, NULL);
    float scales[2] = {1.f, 1.f};
    symQConfig_t groups = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .qBits = 4, .roundingMode = HALF_AWAY};
    symGroupedWeight_t groupedWeight = {.groups = &groups, .domain = domain};
    matmulSymInt32TensorsGroupedWeight(&a.t, &b, &bias.t, &out.t, &groupedWeight, NULL);
    return out.qc.qMaxBits;
}

void testMatmulSymGroupedStampsRescaledWidth(void) {
    int32_t outCode = 0;
    TEST_ASSERT_EQUAL_UINT8(12, groupedInt4StampBits(SYM_CODES_TWOS_COMPLEMENT, 1023, &outCode));
    TEST_ASSERT_EQUAL_INT32(1023 + 8, outCode);
    symAccumSite_t rescaled = symAccumSiteInitRescaled("t", 8, 4, 4, 1023);
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&rescaled));
    TEST_ASSERT_EQUAL_UINT8(11, groupedInt4StampBits(SYM_CODES_TWOS_COMPLEMENT, 500, &outCode));
    TEST_ASSERT_EQUAL_INT32(500 + 8, outCode);
    TEST_ASSERT_EQUAL_UINT8(12, groupedInt4StampBits(SYM_CODES_TWOS_COMPLEMENT, -1023, &outCode));
    TEST_ASSERT_EQUAL_INT32(8 - 1023, outCode);
}

void testMatmulSymGroupedAsymImageStampsItsFullRange(void) {
    int32_t outCode = 0;
    TEST_ASSERT_EQUAL_UINT8(11, groupedInt4StampBits(SYM_CODES_ASYM_IMAGE, 54, &outCode));
    TEST_ASSERT_EQUAL_INT32(54 + 8, outCode);
    TEST_ASSERT_EQUAL_UINT8(12, groupedInt4StampBits(SYM_CODES_ASYM_IMAGE, 500, &outCode));
    TEST_ASSERT_EQUAL_INT32(500 + 8, outCode);
}

/* The #516 repro: the raw 1x1 product -2048 * -2048 = 2^22 re-enters a SYM
 * matmul as an operand. Declared int12 (the pre-stamp default) the guard proved
 * one term safe and the unchecked product 2^22 * -2048 = -2^33 wrapped
 * silently; declared 24 bits the bound is unprovable and the checked product
 * aborts, naming the stamped width in the banner. */
static void chainInt12FullScale(void) {
    int32_t aCode = -2048, bCode = -2048, cCode = -2048, out1Code = 0, out2Code = 0;
    symWire_t a, b, c, out1, out2;
    symWireInit(&a, &aCode, 1, 1, 12);
    symWireInit(&b, &bCode, 1, 1, 12);
    symWireInit(&c, &cCode, 1, 1, 12);
    symWireInit(&out1, &out1Code, 1, 1, UNSTAMPED_WIDTH);
    symWireInit(&out2, &out2Code, 1, 1, UNSTAMPED_WIDTH);
    matmulSymInt32Tensors(&a.t, &b.t, &out1.t);
    matmulSymInt32Tensors(&out1.t, &c.t, &out2.t);
}

void testMatmulSymChainedRawOutputOverflowAborts(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (24, 12)", chainInt12FullScale());
}

void testMatmulSymNarrowChainStaysExact(void) {
    /* a:[1,4] . b:[4,1] of int4 codes 7 -> 196 under the range [-224, 256] ->
     * 10 (the positive corner 256 needs the tenth bit). Chained with an int4
     * code 7 the 1-term (10, 4) site is still provable and the product 1372
     * exact. */
    guardFill(4, 7, 7, false);
    int32_t cCode = 7, out1Code = 0, out2Code = 0;
    symWire_t a, b, c, out1, out2;
    symWireInit(&a, gGuardA, 1, 4, 4);
    symWireInit(&b, gGuardB, 4, 1, 4);
    symWireInit(&c, &cCode, 1, 1, 4);
    symWireInit(&out1, &out1Code, 1, 1, UNSTAMPED_WIDTH);
    symWireInit(&out2, &out2Code, 1, 1, UNSTAMPED_WIDTH);
    matmulSymInt32Tensors(&a.t, &b.t, &out1.t);
    TEST_ASSERT_EQUAL_INT32(196, out1Code);
    TEST_ASSERT_EQUAL_UINT8(symAccumWidthForRange(256, 224), out1.qc.qMaxBits);
    TEST_ASSERT_EQUAL_UINT8(10, out1.qc.qMaxBits);
    matmulSymInt32Tensors(&out1.t, &c.t, &out2.t);
    TEST_ASSERT_EQUAL_INT32(1372, out2Code);
    TEST_ASSERT_EQUAL_UINT8(14, out2.qc.qMaxBits); /* 2^9 * 2^3 = 2^12 -> 14 */
}

int main(void) {

    UNITY_BEGIN();
    RUN_TEST(testSymAccumResultBitsFollowsTheSiteBound);
    RUN_TEST(testMatmulSymStampsResultWidthInt12K1);
    RUN_TEST(testMatmulSymStampsResultWidthBoundaries);
    RUN_TEST(testMatmulSymWithBiasStampIncludesSeed);
    RUN_TEST(testMatmulSymWithBiasStampFollowsTheSeedSign);
    RUN_TEST(testMatmulSymGroupedStampsRescaledWidth);
    RUN_TEST(testMatmulSymGroupedAsymImageStampsItsFullRange);
    RUN_TEST(testMatmulSymChainedRawOutputOverflowAborts);
    RUN_TEST(testMatmulSymNarrowChainStaysExact);
    RUN_TEST(testMatmulGroupedWidePartialAboveSeedGuardRunsExactWhenSumFits);
    RUN_TEST(testMatmulGroupedWidePartialSumThatWrapsHitsTheF9Guard);
    RUN_TEST(testMatmulSymGuardWiderOperandRunsExactWhenItFitsAndAbortsWhenItWraps);
    RUN_TEST(testMatmulGroupedGuardWideWeightRunsExactWhenItFitsAndAbortsWhenItWraps);
    RUN_TEST(testMatmulGroupedGuardAsymImageWeightKeepsItsFullRange);
    RUN_TEST(testMatmulGroupedZeroedCodeDomainAbortsBeforeTheLoop);
    RUN_TEST(testMatmulSymGuardProvenSafeBoundaryN511RunsExact);
    RUN_TEST(testMatmulSymGuardDiesAtFirstWrappingNFullScaleNegativeCodes);
    RUN_TEST(testMatmulSymGuardDiesAtN513FullScalePositiveCodes);
    RUN_TEST(testMatmulSymGuardDiesOnNegativeOverflow);
    RUN_TEST(testMatmulSymGuardCheckedPathStaysExactWhenValuesFit);
    RUN_TEST(testMatmulSymGuardBiasSeedThatTipsTheBoundIsCounted);
    RUN_TEST(testMatmulGroupedGuardDiesOnRawGroupPartialOverflow);
    RUN_TEST(testMatmulGroupedGuardDiesOnRescaledAccumulatorOverflow);
    RUN_TEST(testMatmulGroupedGuardCheckedPathStaysExactWhenValuesFit);
    RUN_TEST(testMatmulGroupedGuardFullScaleN511RunsExact);
    RUN_TEST(testMatmulInt32);
    RUN_TEST(testMatmulInt32WithVector);
    RUN_TEST(testMatmulFloatVectors);
    RUN_TEST(testMatmulSymInt32Tensors);
    RUN_TEST(testMatmulFloat32TensorsWithBiasBroadcastsOverRows);
    RUN_TEST(testMatmulFloat32TensorsWithBiasNullEqualsPlain);
    RUN_TEST(testMatmulSymInt32TensorsWithBiasRescalesBias);
    RUN_TEST(testMatmulSymInt32WiderOperandRunsExact);
    RUN_TEST(testMatmulSymInt32TensorsWithBiasRejectsNonSymInt32Bias);
    RUN_TEST(testMatmulGroupedWeightPerChannelMatchesGold);
    RUN_TEST(testMatmulGroupedWeightGeneralGroupsMatchesGold);
    RUN_TEST(testMatmulGroupedEqualScalesBitIdenticalToScalar);
    RUN_TEST(testMatmulGroupedHonorsOpRoundingMode);
    RUN_TEST(testMatmulGroupedWeightStridedReductionMatchesGold);
    RUN_TEST(testMatmulGroupedWeightStridedGeneralGroupsMatchesGold);
    RUN_TEST(testMatmulGroupedDxEqualScalesBitIdenticalToScalar);
    RUN_TEST(testMatmulGroupedWeightRejectsPerTensorSentinel);
    RUN_TEST(testMatmulBfpMatchesGold);
    RUN_TEST(testMatmulBfpGroupedBiasBindsPerGroupExponent);
    RUN_TEST(testMatmulBfpNoBiasZeroSeeds);
    RUN_TEST(testMatmulBfpPowerOfTwoBitIdenticalToGroupedSym);
    RUN_TEST(testMatmulBfpHeadroomGuardDies);
    RUN_TEST(testBfpSegmentLimitTableValues);
    RUN_TEST(testBfpSumSegmentLimitTableValues);
    RUN_TEST(testBfpGroupAtInRangeTrustsCache);
    RUN_TEST(testBfpGroupAtForwardJumpRecomputes);
    RUN_TEST(testBfpGroupAtBackwardJumpRecomputes);
    RUN_TEST(testBfpGroupAtZeroInitIsEmptyRange);
    RUN_TEST(testBfpGroupAtLastGroupEndsAtElementCount);
    RUN_TEST(testBfpGroupAtPerTensorNeverRecomputes);
    RUN_TEST(testBfpGroupAtMatchesBfpGroupOfOnStridedWalk);
    RUN_TEST(testBfpShrHalfAwayRoundsHalfAwayFromZero);
    RUN_TEST(testBfpShrHalfAwayInt32Extremes);
    RUN_TEST(testBfpAlignPartialLeftShiftIsExactAndRightShiftRounds);
    RUN_TEST(testBfpFoldGuardBitsFromTheProductBound);
    RUN_TEST(testBfpFoldGuardBitsIsTheSmallestGThatHolds);
    RUN_TEST(testBfpFoldGuardBitsRejectsAReductionNoShiftCanHold);
    RUN_TEST(testBfpValueGreaterIsTheExactCompare);
    RUN_TEST(testRoundedDivHalfAwayInt32IsExactOverTheInt32Range);
    RUN_TEST(testBfpSumFoldGuardBitsIsTheSmallestGThatHolds);
    RUN_TEST(testMatmulBfpRejectsMismatchedGroupShape);
    RUN_TEST(testMatmulBfpDxStridedWeightWalkMatchesGold);
    RUN_TEST(testMatmulBfpWeightGradTransposedLossViewMatchesGold);
    RUN_TEST(testMatmulBfpDxPowerOfTwoBitIdenticalToGroupedSym);
    RUN_TEST(testMatmulBfpRawMatchesTheFoldReferenceOnStraddlingGroups);
    RUN_TEST(testMatmulBfpRawWorstCaseM16xM16AllCodesAtTheEdges);
    RUN_TEST(testMatmulBfpRawWorstCaseM16xM8SplitsAtTheSegmentLimit);
    RUN_TEST(testMatmulBfpRawWorstCaseBiasAtTheFloorAndAboveTheProducts);
    RUN_TEST(testMatmulBfpRawStridedWeightViewUsesTheTensorWideMax);
    RUN_TEST(testMatmulBfpRawPowerOfTwoDequantMatchesFloatKernel);
    RUN_TEST(testMatmulBfpRawRejectsANonRawOutputAndBadRawGeometry);

    return UNITY_END();
}
