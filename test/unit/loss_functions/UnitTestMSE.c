#define SOURCE_FILE "UnitTestMSE"

#include "DeathTest.h"
#include "MSE.h"
#include "QuantizationApi.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"
#include <string.h>

void testMSEForward_MeanReturnsPerSampleMean() {
    /* Output [1, 3]: B = 1, F = 3. */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(output, (float[]){1.f, 2.f, 3.f}, 3);

    size_t *labelDims = reserveMemory(2 * sizeof(size_t));
    labelDims[0] = 1;
    labelDims[1] = 3;
    size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 2, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){2.f, 4.f, 6.f}, 3);

    float capturedLoss = mseLossForward(output, label, REDUCTION_MEAN);

    freeTensor(label);
    freeTensor(output);

    /* MEAN: ((1-2)² + (2-4)² + (3-6)²) / 3 = (1+4+9)/3 = 14/3 ≈ 4.667 */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 14.0f / 3.0f, capturedLoss);
}

void testMSEForward_SumReturnsRawSum() {
    /* Output [1, 3]: B = 1, F = 3. */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(output, (float[]){1.f, 2.f, 3.f}, 3);

    size_t *labelDims = reserveMemory(2 * sizeof(size_t));
    labelDims[0] = 1;
    labelDims[1] = 3;
    size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 2, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){2.f, 4.f, 6.f}, 3);

    float capturedLoss = mseLossForward(output, label, REDUCTION_SUM);

    freeTensor(label);
    freeTensor(output);

    /* SUM: 1 + 4 + 9 = 14 (no division). */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 14.0f, capturedLoss);
}

void testMSELossBackward_FloatWritesRawPerElementGrad() {
    size_t numberOfElements = 3;
    size_t dims[] = {numberOfElements};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .orderOfDimensions = orderOfDims, .numberOfDimensions = 1};

    tensor_t modelOutput;
    quantization_t modelOutputQ;
    initFloat32Quantization(&modelOutputQ);
    float modelOutputData[] = {1.f, 2.f, -3.f};
    setTensorValues(&modelOutput, (uint8_t *)modelOutputData, &shape, &modelOutputQ, NULL);

    tensor_t label;
    quantization_t labelQ;
    initFloat32Quantization(&labelQ);
    float labelData[] = {-5.f, -4.f, 2.f};
    setTensorValues(&label, (uint8_t *)labelData, &shape, &labelQ, NULL);

    tensor_t result;
    quantization_t resultQ;
    initFloat32Quantization(&resultQ);
    float resultData[3];
    setTensorValues(&result, (uint8_t *)resultData, &shape, &resultQ, NULL);

    /* Raw per-element gradient: 2*(o-l). No /F division — that lives in
     * computeMeanScaleMSE applied at the optimizer step. */
    mseLossBackwardFloat(&modelOutput, &label, &result);

    /* delta = [6, 6, -5]; raw grad = 2*delta = [12, 12, -10] */
    float expected[] = {12.f, 12.f, -10.f};
    float *actual = (float *)result.data;
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected[i], actual[i]);
    }
}

void testMSELossBackward_SymInt32WritesRawPerElementGrad() {
    size_t numberOfElements = 3;
    size_t dims[] = {1, numberOfElements};
    size_t orderOfDims[] = {0, 1};
    shape_t shape = {.dimensions = dims, .orderOfDimensions = orderOfDims, .numberOfDimensions = 2};

    tensor_t modelOutput;
    quantization_t modelOutputQ;
    initFloat32Quantization(&modelOutputQ);
    float modelOutputData[] = {1.f, 2.f, -3.f};
    setTensorValues(&modelOutput, (uint8_t *)modelOutputData, &shape, &modelOutputQ, NULL);

    tensor_t modelOutputSymInt32;
    symInt32QConfig_t modelOutputSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &modelOutputSymInt32QC);
    quantization_t modelOutputSymInt32Q;
    initSymInt32Quantization(&modelOutputSymInt32QC, &modelOutputSymInt32Q);
    uint8_t modelOutputSymInt32Data[numberOfElements * sizeof(int32_t)];
    setTensorValuesForConversion(modelOutputSymInt32Data, &modelOutputSymInt32Q, &modelOutput,
                                 &modelOutputSymInt32);
    convertTensor(&modelOutput, &modelOutputSymInt32);

    tensor_t label;
    quantization_t labelQ;
    initFloat32Quantization(&labelQ);
    float labelData[] = {-5.f, -4.f, 2.f};
    setTensorValues(&label, (uint8_t *)labelData, &shape, &labelQ, NULL);

    tensor_t labelSymInt32;
    symInt32QConfig_t labelSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &labelSymInt32QC);
    quantization_t labelSymInt32Q;
    initSymInt32Quantization(&labelSymInt32QC, &labelSymInt32Q);
    uint8_t labelSymInt32Data[numberOfElements * sizeof(int32_t)];
    setTensorValuesForConversion(labelSymInt32Data, &labelSymInt32Q, &label, &labelSymInt32);
    convertTensor(&label, &labelSymInt32);

    tensor_t result;
    quantization_t resultQ;
    initFloat32Quantization(&resultQ);
    float resultData[numberOfElements];
    setTensorValues(&result, (uint8_t *)resultData, &shape, &resultQ, NULL);

    tensor_t resultSymInt32;
    symInt32QConfig_t resultSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &resultSymInt32QC);
    quantization_t resultSymInt32Q;
    initSymInt32Quantization(&resultSymInt32QC, &resultSymInt32Q);
    uint8_t resultSymInt32Data[numberOfElements * sizeof(int32_t)];
    memset(resultSymInt32Data, 0, numberOfElements * sizeof(int32_t));
    setTensorValuesForConversion(resultSymInt32Data, &resultSymInt32Q, &result, &resultSymInt32);

    mseLossBackward(&modelOutputSymInt32, &labelSymInt32, &resultSymInt32);
    convertTensor(&resultSymInt32, &result);

    /* Raw per-element gradient over the [1, 3] wire; wider tolerance for fixed-point. */
    float expected[] = {12.f, 12.f, -10.f};
    float *actual = (float *)result.data;
    for (size_t i = 0; i < numberOfElements; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.5f, expected[i], actual[i]);
    }
}

/* ---- BFP fake-quant arms (BFP epic PR4, R-P6) ---- */

/* BFP epic PR4 (R-P6): a BFP wire with EXACT codes and a per-tensor exponent.
 * The fixture is CANONICAL — requantizing its exact dequant reproduces these
 * codes and this exponent — so the fake-quant arm's dequant is lossless and
 * the test can assert EQUALITY against the FLOAT32 arm (R-P7b). */
static tensor_t *buildBfpTensor1DWithCodes(size_t n, uint8_t mantissaBits, uint8_t exponentBits,
                                           int32_t *codes, uint8_t storedExponent) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = n;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    quantization_t *q = quantizationInitBfp(mantissaBits, exponentBits, HALF_AWAY);
    tensor_t *t = initTensor(shape, q, NULL);
    if (codes != NULL) {
        byteConversion((uint8_t *)codes, 32, t->data, mantissaBits, n);
    }
    ((bfpQConfig_t *)q->qConfig)->exponents[0] = storedExponent;
    return t;
}

/* [1, n] FLOAT32 twin of the builder above — same rank, so both operands reach
 * the float core with identical shape pointers. */
static tensor_t *buildFloatTensor1D(size_t n, const float *values) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = n;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)values, n);
    return t;
}

/* Heap FLOAT32 tensor of any rank >= 1 with the given dims, filled from values. */
static tensor_t *buildFloatTensorShaped(const size_t *dims, size_t rank, float *values) {
    size_t *d = reserveMemory(rank * sizeof(size_t));
    size_t n = 1;
    for (size_t k = 0; k < rank; k++) {
        d[k] = dims[k];
        n *= dims[k];
    }
    size_t *order = reserveMemory(rank * sizeof(size_t));
    setOrderOfDimsForNewTensor(rank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, d, rank, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, values, n);
    return t;
}

/* R-P6: the BFP arm routes to the SAME dtype-generic fake-quant helper the
 * SYM arm uses — convertTensor already owns conversionMatrix[BFP][FLOAT32].
 * Codes [8, -4, 12, 16] at stored exponent 127 (= bias, scale 1.0) dequantize
 * EXACTLY to [8, -4, 12, 16], so the BFP loss must equal the FLOAT32 loss
 * bit-for-bit, and the hand-derived value is
 * ((8-7)^2 + 0 + (12-10)^2 + 0) / 4 = 5/4 = 1.25. */
void testMseLossForwardBfpEqualsFloat32OnExactGrid(void) {
    int32_t codes[4] = {8, -4, 12, 16};
    tensor_t *bfpOut = buildBfpTensor1DWithCodes(4, 6, 8, codes, 127);
    float exactValues[4] = {8.0f, -4.0f, 12.0f, 16.0f};
    float labelValues[4] = {7.0f, -4.0f, 10.0f, 16.0f};
    tensor_t *floatOut = buildFloatTensor1D(4, exactValues);
    tensor_t *label = buildFloatTensor1D(4, labelValues);

    float bfpLoss = mseLossForward(bfpOut, label, REDUCTION_MEAN);
    float floatLoss = mseLossForward(floatOut, label, REDUCTION_MEAN);

    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(floatLoss, bfpLoss,
                                    "an exact-grid BFP output must give the FLOAT32 loss");
    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(1.25f, bfpLoss, "hand-derived MEAN MSE");

    freeTensor(label);
    freeTensor(floatOut);
    freeTensor(bfpOut);
}

/* The backward's final convertTensor requantizes the raw 2*(p - y) into the
 * RESULT's own BFP grid with FRESH per-group exponents (the BFP analog of the
 * SYM arm's fresh absmax scale). Hand-derived: 2*(p-y) = [2, 0, 4, 0];
 * absmax 4, qMax = 2^5 - 1 = 31, 4/31 = 0.129032 = 0.516129 * 2^-2, so the
 * smallest E with absmax/2^E <= qMax is -2 -> stored 127 - 2 = 125,
 * scale 2^-2 = 0.25, codes = [8, 0, 16, 0]. */
void testMseLossBackwardBfpRequantizesIntoFreshGrid(void) {
    int32_t codes[4] = {8, -4, 12, 16};
    int32_t sentinel[4] = {-9, -9, -9, -9};
    tensor_t *bfpOut = buildBfpTensor1DWithCodes(4, 6, 8, codes, 127);
    float labelValues[4] = {7.0f, -4.0f, 10.0f, 16.0f};
    tensor_t *label = buildFloatTensor1D(4, labelValues);
    tensor_t *result = buildBfpTensor1DWithCodes(4, 6, 8, sentinel, 127);

    mseLossBackward(bfpOut, label, result);

    int32_t got[4];
    unpackSignExtend(result->data, 6, 0, got, 4);
    int32_t expected[4] = {8, 0, 16, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, 4);
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(125,
                                    ((bfpQConfig_t *)result->quantization->qConfig)->exponents[0],
                                    "the produced grad must get a FRESH exponent (absmax 4)");

    freeTensor(result);
    freeTensor(label);
    freeTensor(bfpOut);
}

/* PR4 adversarial gate (F1/F2): every scratch buffer in the fake-quant bodies
 * is a VLA sized from the MODEL OUTPUT's element count, but each convertTensor
 * walks its OWN source's count and the label scratch even borrows the LABEL's
 * shape — so a label longer than the output writes past the end of a stack
 * array. The mismatch is deliberately ONE element: large enough to overrun,
 * small enough that the unguarded child returns normally instead of crashing,
 * which is what makes the missing-guard RED a clean "exit code 0" rather than
 * a signal. */
void testMseLossForwardBfpRejectsLabelCountMismatch(void) {
    int32_t codes[4] = {8, -4, 12, 16};
    tensor_t *bfpOut = buildBfpTensor1DWithCodes(4, 6, 8, codes, 127);
    float labelValues[5] = {7.0f, -4.0f, 10.0f, 16.0f, 1.0f};
    tensor_t *label = buildFloatTensor1D(5, labelValues);

    ASSERT_EXITS_WITH_FAILURE((void)mseLossForward(bfpOut, label, REDUCTION_MEAN));

    freeTensor(label);
    freeTensor(bfpOut);
}

void testMseLossBackwardBfpRejectsLabelCountMismatch(void) {
    int32_t codes[4] = {8, -4, 12, 16};
    int32_t sentinel[4] = {-9, -9, -9, -9};
    tensor_t *bfpOut = buildBfpTensor1DWithCodes(4, 6, 8, codes, 127);
    float labelValues[5] = {7.0f, -4.0f, 10.0f, 16.0f, 1.0f};
    tensor_t *label = buildFloatTensor1D(5, labelValues);
    tensor_t *result = buildBfpTensor1DWithCodes(4, 6, 8, sentinel, 127);

    ASSERT_EXITS_WITH_FAILURE(mseLossBackward(bfpOut, label, result));

    freeTensor(result);
    freeTensor(label);
    freeTensor(bfpOut);
}

/* PR4 adversarial gate (F1/F2 follow-through): the RESULT wire is the third
 * operand with the same hole, and it needs its own case because a CORRECT label
 * is what carries the call past the label guard. The result scratch is a VLA
 * sized from the model output's count but borrows the RESULT's shape_t, so the
 * closing convertTensor(&resultFloat, result) walks the RESULT's count and
 * reads past the end of that stack array; a shorter result would instead emit a
 * silently truncated grad. */
void testMseLossBackwardBfpRejectsResultCountMismatch(void) {
    int32_t codes[4] = {8, -4, 12, 16};
    int32_t sentinel[5] = {-9, -9, -9, -9, -9};
    tensor_t *bfpOut = buildBfpTensor1DWithCodes(4, 6, 8, codes, 127);
    float labelValues[4] = {7.0f, -4.0f, 10.0f, 16.0f};
    tensor_t *label = buildFloatTensor1D(4, labelValues);
    tensor_t *result = buildBfpTensor1DWithCodes(5, 6, 8, sentinel, 127);

    ASSERT_EXITS_WITH_FAILURE(mseLossBackward(bfpOut, label, result));

    freeTensor(result);
    freeTensor(label);
    freeTensor(bfpOut);
}

/* PR4 adversarial delta (D0), kept by #153: the operand shape guard must sit
 * at the PUBLIC entry, not inside the fake-quant helper — otherwise the
 * FLOAT32 arm keeps the whole hole.
 * That arm is not the milder half either: mseLossBackwardFloat writes
 * resultArray[i] for i < the MODEL OUTPUT's count straight into the caller's
 * buffer, so a SHORTER result is an out-of-bounds heap WRITE. These two cases
 * use a LONGER operand on purpose — the float arm ignores the surplus, so the
 * unguarded child returns normally and the missing-guard RED is a clean "exit
 * code 0" rather than a signal — while the guard they pin rejects both
 * directions. They are also what pins the HOIST itself: a guard pushed back
 * down into the fake-quant helper would leave these two red while every BFP
 * death test stayed green. */
void testMseLossForwardFloat32RejectsLabelCountMismatch(void) {
    float outValues[4] = {8.0f, -4.0f, 12.0f, 16.0f};
    float labelValues[5] = {7.0f, -4.0f, 10.0f, 16.0f, 1.0f};
    tensor_t *output = buildFloatTensor1D(4, outValues);
    tensor_t *label = buildFloatTensor1D(5, labelValues);

    ASSERT_EXITS_WITH_FAILURE((void)mseLossForward(output, label, REDUCTION_MEAN));

    freeTensor(label);
    freeTensor(output);
}

void testMseLossBackwardFloat32RejectsResultCountMismatch(void) {
    float outValues[4] = {8.0f, -4.0f, 12.0f, 16.0f};
    float labelValues[4] = {7.0f, -4.0f, 10.0f, 16.0f};
    float resultValues[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *modelOutput = buildFloatTensor1D(4, outValues);
    tensor_t *label = buildFloatTensor1D(4, labelValues);
    tensor_t *result = buildFloatTensor1D(5, resultValues);

    ASSERT_EXITS_WITH_FAILURE(mseLossBackward(modelOutput, label, result));

    freeTensor(result);
    freeTensor(label);
    freeTensor(modelOutput);
}

/* #153: every loss operand must have the model output's exact shape, and the
 * output must be [B, ...] with a feature axis and at least one element.
 * Each fixture below (except the leading-axis one) passes the old
 * count-only check, so only the shape check can reject it. */
void testMseLossForwardRejectsRank1Output(void) {
    size_t dims[] = {3};
    tensor_t *output = buildFloatTensorShaped(dims, 1, (float[]){1.f, 2.f, 3.f});
    tensor_t *label = buildFloatTensorShaped(dims, 1, (float[]){2.f, 4.f, 6.f});
    ASSERT_EXITS_WITH_FAILURE((void)mseLossForward(output, label, REDUCTION_MEAN));
    freeTensor(label);
    freeTensor(output);
}

/* Empty tensors are stack-built: a heap N = 0 allocation is
 * implementation-defined (#160). */
static void expectMseForwardDiesOnEmpty(size_t d0, size_t d1) {
    size_t dims[] = {d0, d1};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t q;
    initFloat32Quantization(&q);
    float dummy[1] = {0.f};
    tensor_t output;
    tensor_t label;
    setTensorValues(&output, (uint8_t *)dummy, &shape, &q, NULL);
    setTensorValues(&label, (uint8_t *)dummy, &shape, &q, NULL);
    ASSERT_EXITS_WITH_FAILURE((void)mseLossForward(&output, &label, REDUCTION_MEAN));
}

void testMseLossForwardRejectsEmptyBatch(void) {
    expectMseForwardDiesOnEmpty(0, 3);
}

void testMseLossForwardRejectsEmptyFeatureAxis(void) {
    expectMseForwardDiesOnEmpty(1, 0);
}

void testMseLossForwardRejectsLabelRankMismatch(void) {
    size_t outDims[] = {1, 6};
    size_t labelDims[] = {1, 6, 1};
    float v[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *output = buildFloatTensorShaped(outDims, 2, v);
    tensor_t *label = buildFloatTensorShaped(labelDims, 3, v);
    ASSERT_EXITS_WITH_FAILURE((void)mseLossForward(output, label, REDUCTION_MEAN));
    freeTensor(label);
    freeTensor(output);
}

/* Only dims[0] differs. Equal counts cannot isolate axis 0, so the label is
 * LONGER: the FLOAT32 arm reads just the output's 3 elements and returns
 * normally if the axis-0 compare is missing (clean exit-0 RED, no OOB). */
void testMseLossForwardRejectsLeadingAxisMismatch(void) {
    size_t outDims[] = {1, 3};
    size_t labelDims[] = {2, 3};
    float v[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *output = buildFloatTensorShaped(outDims, 2, v);
    tensor_t *label = buildFloatTensorShaped(labelDims, 2, v);
    ASSERT_EXITS_WITH_FAILURE((void)mseLossForward(output, label, REDUCTION_MEAN));
    freeTensor(label);
    freeTensor(output);
}

/* Same rank, same dims[0], same count: only a full-shape compare catches it
 * (a [B, C, L] label against a [B, L, C] output). */
void testMseLossForwardRejectsTransposedFeatureLayout(void) {
    size_t outDims[] = {1, 2, 3};
    size_t labelDims[] = {1, 3, 2};
    float v[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *output = buildFloatTensorShaped(outDims, 3, v);
    tensor_t *label = buildFloatTensorShaped(labelDims, 3, v);
    ASSERT_EXITS_WITH_FAILURE((void)mseLossForward(output, label, REDUCTION_MEAN));
    freeTensor(label);
    freeTensor(output);
}

/* Over-rejection guard: a rank-3 output with a real leading-1 feature axis
 * (ECG's [1, 1, 140] shape) must still be scored. */
void testMseLossForwardAcceptsMatchingRank3Shapes(void) {
    size_t dims[] = {1, 1, 4};
    tensor_t *output = buildFloatTensorShaped(dims, 3, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *label = buildFloatTensorShaped(dims, 3, (float[]){2.f, 2.f, 3.f, 6.f});
    float loss = mseLossForward(output, label, REDUCTION_SUM);
    freeTensor(label);
    freeTensor(output);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, loss); /* 1 + 0 + 0 + 4 */
}

/* #153: computeMeanScaleMSE divides the element count by dims[0] as a size_t.
 * dims[0] = 0 is a SIGFPE on x86 and a silent 0 (then +inf) on AArch64 /
 * Armv7-M; a rank-1 view has no feature axis. Its only production caller
 * passes the batchViewOf label view, so both mean a caller bug. */
void testComputeMeanScaleMseRejectsEmptyBatch(void) {
    size_t dims[] = {0, 10};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t q;
    initFloat32Quantization(&q);
    float dummy[1] = {0.f};
    tensor_t labelView;
    setTensorValues(&labelView, (uint8_t *)dummy, &shape, &q, NULL);
    ASSERT_EXITS_WITH_FAILURE((void)computeMeanScaleMSE(32, &labelView));
}

void testComputeMeanScaleMseRejectsRank1View(void) {
    size_t dims[] = {5};
    tensor_t *labelView = buildFloatTensorShaped(dims, 1, (float[]){1.f, 2.f, 3.f, 4.f, 5.f});
    ASSERT_EXITS_WITH_FAILURE((void)computeMeanScaleMSE(32, labelView));
    freeTensor(labelView);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(testMSEForward_MeanReturnsPerSampleMean);
    RUN_TEST(testMSEForward_SumReturnsRawSum);

    RUN_TEST(testMSELossBackward_FloatWritesRawPerElementGrad);
    RUN_TEST(testMSELossBackward_SymInt32WritesRawPerElementGrad);

    RUN_TEST(testMseLossForwardBfpEqualsFloat32OnExactGrid);
    RUN_TEST(testMseLossBackwardBfpRequantizesIntoFreshGrid);
    RUN_TEST(testMseLossForwardBfpRejectsLabelCountMismatch);
    RUN_TEST(testMseLossBackwardBfpRejectsLabelCountMismatch);
    RUN_TEST(testMseLossBackwardBfpRejectsResultCountMismatch);
    RUN_TEST(testMseLossForwardFloat32RejectsLabelCountMismatch);
    RUN_TEST(testMseLossBackwardFloat32RejectsResultCountMismatch);

    RUN_TEST(testMseLossForwardRejectsRank1Output);
    RUN_TEST(testMseLossForwardRejectsEmptyBatch);
    RUN_TEST(testMseLossForwardRejectsEmptyFeatureAxis);
    RUN_TEST(testMseLossForwardRejectsLabelRankMismatch);
    RUN_TEST(testMseLossForwardRejectsLeadingAxisMismatch);
    RUN_TEST(testMseLossForwardRejectsTransposedFeatureLayout);
    RUN_TEST(testMseLossForwardAcceptsMatchingRank3Shapes);

    RUN_TEST(testComputeMeanScaleMseRejectsEmptyBatch);
    RUN_TEST(testComputeMeanScaleMseRejectsRank1View);

    return UNITY_END();
}
