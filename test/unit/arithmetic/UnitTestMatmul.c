#include <string.h>

#include "Arithmetic.h"
#include "BfpKernelSupport.h"
#include "DeathTest.h"
#include "Matmul.h"
#include "RNG.h"
#include "Tensor.h"
#include "unity.h"

#include <DTypes.h>
#include <TensorConversion.h>

#include "expected_bfp_matmul.h"
#include "expected_group_matmul.h"

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

void testMatmulSymInt32RejectsOperandWiderThanInt12() {
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
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &aQC, 13); /* violates int12 contract */
    initSymInt32QConfig(HALF_AWAY, &bQC);                 /* default int12 */
    initSymInt32QConfig(HALF_AWAY, &oQC);
    quantization_t aQ, bQ, oQ;
    initSymInt32Quantization(&aQC, &aQ);
    initSymInt32Quantization(&bQC, &bQ);
    initSymInt32Quantization(&oQC, &oQ);
    setTensorValues(&a, (uint8_t *)aData, &aS, &aQ, NULL);
    setTensorValues(&b, (uint8_t *)bData, &bS, &bQ, NULL);
    setTensorValues(&out, (uint8_t *)outData, &oS, &oQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(matmulSymInt32Tensors(&a, &b, &out));
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

    ASSERT_EXITS_WITH_FAILURE(matmulSymInt32TensorsWithBias(&a, &b, &out, &bias));
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

    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, &biasTensor, &outputTensor,
                                       &weightGroups);

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

    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, &biasTensor, &outputTensor,
                                       &weightGroups);

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

    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensorGrouped, NULL, &outGrouped, &weightGroups);

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

    matmulSymInt32TensorsWithBias(&aTensor, &bTensorScalar, &outScalar, NULL);

    TEST_ASSERT_EQUAL_INT32_ARRAY(outScalarData, outGroupedData, 6);
    TEST_ASSERT_EQUAL_FLOAT(outScalarQC.scale, outGroupedQC.scale);
}

/* Rounding-mode-honored check (substitutes for a Python-emulated SR-mode
 * gold fixture, which would need the C-side seeded RNG stream — no existing
 * goldgen script emulates SR_HALF_AWAY, see generate_expected_group_matmul.py's
 * docstring). Reuses the general-groups fixture (known non-exact combine
 * quotients) with SR_HALF_AWAY as the op's rounding mode (carried via b's own
 * qConfig — the same field the executeOp prologue would set from
 * arithmetic.roundingMode). If the combine hardcoded HALF_AWAY instead of
 * honoring roundingMode, two runs under different RNG seeds would be BIT-
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

    rngSetSeed(1);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &out1, &weightGroups);
    rngSetSeed(2);
    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &out2, &weightGroups);

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

    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &outputTensor, &weightGroups);

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

    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &outputTensor, &weightGroups);

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

    matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensorGrouped, NULL, &outGrouped, &weightGroups);

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

    ASSERT_EXITS_WITH_FAILURE(
        matmulSymInt32TensorsGroupedWeight(&aTensor, &bTensor, NULL, &outputTensor, &weightGroups));
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

/* PR2 self-review finding 3: the bias seed must dequantize through ITS OWN
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

/* BFP power-of-two twin (spec §8c), the BFP sibling of
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

    matmulSymInt32TensorsGroupedWeight(&aSymTensor, &bSymTensor, NULL, &outSymTensor,
                                       &weightGroups);

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

/* Group-shape fail-fast (review finding 2): bfpGroupOf divides by groupSize
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
 * strided walk exactly (D8 amendment: no re-quantize, folds just more often). */
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

/* PR3 dx power-of-two twin (spec §8c in the backward, D8 amendment): the dx
 * sibling of testMatmulBfpPowerOfTwoBitIdenticalToGroupedSym above (see its
 * comment for the exactness argument), in the RAW-weight-storage orientation
 * of testMatmulGroupedDxEqualScalesBitIdenticalToScalar (same mantissas,
 * loss per-tensor stored 126 <-> SYM aScale 0.5f, W grouped all-125 <->
 * weightGroups scales 0.25f): the BFP float output must be BIT-IDENTICAL to
 * the grouped-SYM path's dequantized output. */
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

    matmulSymInt32TensorsGroupedWeight(&aSymTensor, &bSymTensor, NULL, &outSymTensor,
                                       &weightGroups);

    for (size_t i = 0; i < 12; i++) {
        float symDequant = (float)outSymData[i] * outSymQC.scale;
        TEST_ASSERT_EQUAL_MEMORY(&symDequant, &outBfpData[i], sizeof(float));
    }
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
    RUN_TEST(testMatmulSymInt32RejectsOperandWiderThanInt12);
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
    RUN_TEST(testMatmulBfpRejectsMismatchedGroupShape);
    RUN_TEST(testMatmulBfpDxStridedWeightWalkMatchesGold);
    RUN_TEST(testMatmulBfpWeightGradTransposedLossViewMatchesGold);
    RUN_TEST(testMatmulBfpDxPowerOfTwoBitIdenticalToGroupedSym);

    return UNITY_END();
}
