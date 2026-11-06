#define SOURCE_FILE "UnitTestCrossEntropy"

#include "BfpRawFixture.h"
#include "BfpScatterRef.h"
#include "CrossEntropy.h"
#include "DeathTest.h"
#include "LayerExec.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_cross_entropy_sym.h"
#include "unity.h"
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

void unitTestCrossEntropySoftmaxBackward() {
    size_t inputSize = 3;

    tensor_t logits;
    float logitsData[] = {2.f, 1.f, 0.1f};
    size_t logitsDims[] = {1, inputSize};
    size_t logitsNumberOfDims = 2;
    size_t logitsOrder[] = {0, 1};
    shape_t logitsShape;
    setShape(&logitsShape, logitsDims, logitsNumberOfDims, logitsOrder);
    quantization_t logitsQ;
    initFloat32Quantization(&logitsQ);
    setTensorValues(&logits, (uint8_t *)logitsData, &logitsShape, &logitsQ, NULL);

    tensor_t softmaxOutput;
    float softmaxOutputData[inputSize];
    size_t softmaxOutputDims[] = {1, inputSize};
    size_t softmaxOutputNumberOfDims = 2;
    size_t softmaxOutputOrder[] = {0, 1};
    shape_t softmaxOutputShape;
    setShape(&softmaxOutputShape, softmaxOutputDims, softmaxOutputNumberOfDims, softmaxOutputOrder);
    quantization_t softmaxOutputQ;
    initFloat32Quantization(&softmaxOutputQ);
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxOutputData, &softmaxOutputShape,
                    &softmaxOutputQ, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.forward(softmaxLayer, &logits, &softmaxOutput, NULL);

    tensor_t distribution;
    float distData[] = {0.9f, 0.1f, 0.0f};
    size_t distDims[] = {1, inputSize};
    size_t distNumberOfDims = 2;
    size_t distOrderOfDims[] = {0, 1};
    shape_t distShape;
    setShape(&distShape, distDims, distNumberOfDims, distOrderOfDims);
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &distShape, &distQ, NULL);

    tensor_t propLoss;
    float propLossData[] = {2.f, 1.f, 0.1f};
    size_t propLossDims[] = {1, inputSize};
    size_t propLossNumberOfDims = 2;
    size_t propLossOrder[] = {0, 1};
    shape_t propLossShape;
    setShape(&propLossShape, propLossDims, propLossNumberOfDims, propLossOrder);
    quantization_t propLossQ;
    initFloat32Quantization(&propLossQ);
    setTensorValues(&propLoss, (uint8_t *)propLossData, &propLossShape, &propLossQ, NULL);

    crossEntropySoftmaxBackward(&softmaxOutput, &distribution, &propLoss, NULL);

    /* CAPTURE. propLoss.data is stack memory (propLossData[]); copy into a
     * dedicated capture array so the assertions read from a buffer that
     * doesn't depend on freeing softmaxLayer/floatQ first. */
    float capturedActual[3];
    for (size_t i = 0; i < inputSize; i++) {
        capturedActual[i] = ((float *)propLoss.data)[i];
    }

    /* FREE. */
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);

    /* ASSERT: raw per-element gradient (p-y), no batch divisor. */
    float expected[] = {-0.2410f, 0.1424f, 0.0986f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, expected[i], capturedActual[i]);
    }
}

void testCrossEntropyBackward_WritesRawPerElementGrad(void) {
    size_t inputSize = 3;

    tensor_t softmaxOutput;
    float softmaxData[] = {0.7f, 0.2f, 0.1f};
    size_t dims[] = {1, inputSize};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t softmaxQ;
    initFloat32Quantization(&softmaxQ);
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxData, &shape, &softmaxQ, NULL);

    tensor_t distribution;
    float distData[] = {1.f, 0.f, 0.f};
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &shape, &distQ, NULL);

    tensor_t lossGrad;
    float lossGradData[3] = {0};
    quantization_t lossGradQ;
    initFloat32Quantization(&lossGradQ);
    setTensorValues(&lossGrad, (uint8_t *)lossGradData, &shape, &lossGradQ, NULL);

    /* Raw per-element gradient: (p-y). No batch divisor — scaleOptimizerGradients
     * applies macro-scale at the optimizer step. */
    crossEntropySoftmaxBackward(&softmaxOutput, &distribution, &lossGrad, NULL);

    /* Expected = p - y = [0.7-1, 0.2-0, 0.1-0] = [-0.3, 0.2, 0.1] */
    float expected[3] = {0.7f - 1.f, 0.2f - 0.f, 0.1f - 0.f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expected[i], ((float *)lossGrad.data)[i]);
    }
}

void testCrossEntropyForward_SumReturnsRawSum() {
    tensor_t logits;
    float logitData[] = {2.f, 1.f, 0.1f};
    size_t logitDims[] = {1, 3};
    size_t logitOrder[] = {0, 1};
    shape_t logitShape;
    setShape(&logitShape, logitDims, 2, logitOrder);
    quantization_t logitQ;
    initFloat32Quantization(&logitQ);
    setTensorValues(&logits, (uint8_t *)logitData, &logitShape, &logitQ, NULL);

    tensor_t softmaxOutput;
    float outputData[3];
    size_t outputDims[] = {1, 3};
    size_t outputOrder[] = {0, 1};
    shape_t outputShape;
    setShape(&outputShape, outputDims, 2, outputOrder);
    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    setTensorValues(&softmaxOutput, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.forward(softmaxLayer, &logits, &softmaxOutput, NULL);

    tensor_t distribution;
    float distData[] = {0.9f, 0.1f, 0.0f};
    size_t distDims[] = {1, 3};
    size_t distOrder[] = {0, 1};
    shape_t distShape;
    setShape(&distShape, distDims, 2, distOrder);
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &distShape, &distQ, NULL);

    float capturedActual = crossEntropyForwardFloat(&softmaxOutput, &distribution, REDUCTION_SUM);

    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);

    /* SUM: same as the pre-existing forward value (raw -log probability sum). */
    float expected = 0.5170299410820007f;
    TEST_ASSERT_EQUAL_FLOAT(expected, capturedActual);
}

void testCrossEntropyForward_MeanDividesByMicrobatch() {
    /* Synthetic distribution with B=2 (dimensions[0] = 2). The B=2 case
     * forces MEAN and SUM to diverge numerically — without B>1, MEAN and
     * SUM would coincide and a missing branch in the impl would pass
     * the test accidentally. */
    tensor_t softmaxOutput;
    float softmaxData[] = {0.6f, 0.4f, 0.7f, 0.3f};
    size_t dims[] = {2, 2};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t softmaxQ;
    initFloat32Quantization(&softmaxQ);
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxData, &shape, &softmaxQ, NULL);

    tensor_t distribution;
    float distData[] = {1.f, 0.f, 1.f, 0.f};
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &shape, &distQ, NULL);

    /* SUM = -log(0.6) + -log(0.7); MEAN = SUM / 2 (microbatch dim B=2). */
    float capturedSum = crossEntropyForwardFloat(&softmaxOutput, &distribution, REDUCTION_SUM);
    float capturedMean = crossEntropyForwardFloat(&softmaxOutput, &distribution, REDUCTION_MEAN);

    float expectedSum = -logf(0.6f) + -logf(0.7f);
    float expectedMean = expectedSum / 2.0f;

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedSum, capturedSum);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedMean, capturedMean);
}

void testCrossEntropyForward_Mean1DTensorReturnsSum() {
    /* 1D tensor (numberOfDimensions=1) carries no microbatch dim — MEAN
     * passes through as the raw -log probability sum (PyTorch parity:
     * a 1D logit vector is treated as a single-sample batch with no
     * divisor). Without the numberOfDimensions>=2 guard the impl would
     * divide by dimensions[0]=3 and yield -log(0.6)/3, breaking parity.
     * Arm-only: crossEntropyForward rejects a rank-1 output (#153); this
     * calls crossEntropyForwardFloat directly. */
    tensor_t softmaxOutput;
    float softmaxData[] = {0.6f, 0.3f, 0.1f};
    size_t dims[] = {3};
    size_t order[] = {0};
    shape_t shape;
    setShape(&shape, dims, 1, order);
    quantization_t softmaxQ;
    initFloat32Quantization(&softmaxQ);
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxData, &shape, &softmaxQ, NULL);

    tensor_t distribution;
    float distData[] = {1.f, 0.f, 0.f};
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &shape, &distQ, NULL);

    float capturedSum = crossEntropyForwardFloat(&softmaxOutput, &distribution, REDUCTION_SUM);
    float capturedMean = crossEntropyForwardFloat(&softmaxOutput, &distribution, REDUCTION_MEAN);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -logf(0.6f), capturedSum);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, capturedSum, capturedMean);
}

void testCrossEntropyForward_DispatcherFloat32MatchesFloatImpl() {
    /* The dispatcher's FLOAT32 arm must be behavior-identical to the direct
     * float implementation. Synthetic softmax output with B=2 so the MEAN
     * branch of the float impl is exercised through the dispatcher too. */
    tensor_t softmaxOutput;
    float softmaxData[] = {0.6f, 0.4f, 0.7f, 0.3f};
    size_t dims[] = {2, 2};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t softmaxQ;
    initFloat32Quantization(&softmaxQ);
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxData, &shape, &softmaxQ, NULL);

    tensor_t distribution;
    float distData[] = {1.f, 0.f, 1.f, 0.f};
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &shape, &distQ, NULL);

    float viaDispatcher = crossEntropyForward(&softmaxOutput, &distribution, REDUCTION_MEAN, NULL);
    float viaFloatImpl = crossEntropyForwardFloat(&softmaxOutput, &distribution, REDUCTION_MEAN);

    TEST_ASSERT_EQUAL_FLOAT(viaFloatImpl, viaDispatcher);
    /* Anchor against both-wrong drift: (-log 0.6 - log 0.7) / 2 (B=2 MEAN). */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, (-logf(0.6f) - logf(0.7f)) / 2.0f, viaDispatcher);
}

/* ---- SYM_INT32 fake-quant arms (#206, M5) ---- */

typedef struct symStackTensor {
    tensor_t tensor;
    quantization_t q;
    symInt32QConfig_t qc;
} symStackTensor_t;

/* Quantize a FLOAT32 stack tensor into caller-provided SYM_INT32 storage
 * (int12 default width, HALF_AWAY) — the MSE SYM test pattern. The gold
 * fixtures are dequant-round-trip-stable, so this lands on exactly the
 * generator's mantissas + scale. */
static void makeSymFromFloat(symStackTensor_t *st, uint8_t *storage, tensor_t *floatSrc) {
    initSymInt32QConfig(HALF_AWAY, &st->qc);
    initSymInt32Quantization(&st->qc, &st->q);
    setTensorValuesForConversion(storage, &st->q, floatSrc, &st->tensor);
    convertTensor(floatSrc, &st->tensor, NULL);
}

static void makeFloatStack(tensor_t *t, float *data, shape_t *shape, quantization_t *q) {
    initFloat32Quantization(q);
    setTensorValues(t, (uint8_t *)data, shape, q, NULL);
}

void testCrossEntropyForwardSymInt32MatchesGold(void) {
    size_t dims[] = {1, 3};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);

    float pData[3];
    memcpy(pData, softmaxOutput_ceSym_symBasic, sizeof(pData));
    tensor_t pFloat;
    quantization_t pFloatQ;
    makeFloatStack(&pFloat, pData, &shape, &pFloatQ);

    symStackTensor_t pSym;
    uint8_t pSymData[3 * sizeof(int32_t)];
    makeSymFromFloat(&pSym, pSymData, &pFloat);

    float yData[3];
    memcpy(yData, label_ceSym_symBasic, sizeof(yData));
    tensor_t label;
    quantization_t labelQ;
    makeFloatStack(&label, yData, &shape, &labelQ);

    float lossSum = crossEntropyForward(&pSym.tensor, &label, REDUCTION_SUM, NULL);

    TEST_ASSERT_FLOAT_WITHIN(lossTol_ceSym_symBasic, expectedLossSum_ceSym_symBasic, lossSum);
}

void testCrossEntropyForwardSymMicrobatchMeanDivides(void) {
    size_t dims[] = {2, 3};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);

    float pData[6];
    memcpy(pData, softmaxOutput_ceSym_symMicrobatch, sizeof(pData));
    tensor_t pFloat;
    quantization_t pFloatQ;
    makeFloatStack(&pFloat, pData, &shape, &pFloatQ);

    symStackTensor_t pSym;
    uint8_t pSymData[6 * sizeof(int32_t)];
    makeSymFromFloat(&pSym, pSymData, &pFloat);

    float yData[6];
    memcpy(yData, label_ceSym_symMicrobatch, sizeof(yData));
    tensor_t label;
    quantization_t labelQ;
    makeFloatStack(&label, yData, &shape, &labelQ);

    float lossMean = crossEntropyForward(&pSym.tensor, &label, REDUCTION_MEAN, NULL);

    TEST_ASSERT_FLOAT_WITHIN(lossTol_ceSym_symMicrobatch, expectedLossMean_ceSym_symMicrobatch,
                             lossMean);
}

void testCrossEntropySoftmaxBackwardSymWritesRequantizedGrad(void) {
    size_t dims[] = {2, 3};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);

    float pData[6];
    memcpy(pData, softmaxOutput_ceSym_symMicrobatch, sizeof(pData));
    tensor_t pFloat;
    quantization_t pFloatQ;
    makeFloatStack(&pFloat, pData, &shape, &pFloatQ);

    symStackTensor_t pSym;
    uint8_t pSymData[6 * sizeof(int32_t)];
    makeSymFromFloat(&pSym, pSymData, &pFloat);

    float yData[6];
    memcpy(yData, label_ceSym_symMicrobatch, sizeof(yData));
    tensor_t label;
    quantization_t labelQ;
    makeFloatStack(&label, yData, &shape, &labelQ);

    /* SYM result wire (the loss-grad seed inherits the model output's dtype,
     * the remat wire table's inherited seed header) — its own int12
     * HALF_AWAY config; the backward requantizes (p-y) into it. */
    symStackTensor_t grad;
    int32_t gradData[6] = {0};
    initSymInt32QConfig(HALF_AWAY, &grad.qc);
    initSymInt32Quantization(&grad.qc, &grad.q);
    setTensorValues(&grad.tensor, (uint8_t *)gradData, &shape, &grad.q, NULL);

    crossEntropySoftmaxBackward(&pSym.tensor, &label, &grad.tensor, NULL);

    float scale = grad.qc.scale;
    TEST_ASSERT_FLOAT_WITHIN(expectedGradScale_ceSym_symMicrobatch * scaleTol_ceSym_symMicrobatch,
                             expectedGradScale_ceSym_symMicrobatch, scale);
    for (size_t i = 0; i < expectedGradMantissas_ceSym_symMicrobatch_len; i++) {
        TEST_ASSERT_INT_WITHIN(mantissaTol_ceSym_symMicrobatch,
                               expectedGradMantissas_ceSym_symMicrobatch[i], gradData[i]);
        TEST_ASSERT_FLOAT_WITHIN(gradDequantTol_ceSym_symMicrobatch,
                                 expectedGradDequant_ceSym_symMicrobatch[i],
                                 (float)gradData[i] * scale);
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

/* [1, n] FLOAT32 twin of the builder above. */
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

/* R-P6: BFP joins the SYM_INT32 fake-quant arm. p = [0.5, 0.25, 0.125, 0.125]
 * is grid-exact at m=6: absmax 0.5, 0.5/31 = 0.0161290 = 0.516129 * 2^-5, so
 * the stored exponent is 127 - 5 = 122 (scale 2^-5 = 0.03125) and the codes
 * are [16, 8, 4, 4]. The dequant is therefore lossless and the BFP loss must
 * equal the FLOAT32 loss bit-for-bit (no logf gold needed). */
void testCrossEntropyForwardBfpEqualsFloat32OnExactGrid(void) {
    int32_t codes[4] = {16, 8, 4, 4};
    tensor_t *bfpP = buildBfpTensor1DWithCodes(4, 6, 8, codes, 122);
    float exactP[4] = {0.5f, 0.25f, 0.125f, 0.125f};
    float y[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *floatP = buildFloatTensor1D(4, exactP);
    tensor_t *label = buildFloatTensor1D(4, y);

    float bfpLoss = crossEntropyForward(bfpP, label, REDUCTION_SUM, NULL);
    float floatLoss = crossEntropyForward(floatP, label, REDUCTION_SUM, NULL);

    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(floatLoss, bfpLoss,
                                    "an exact-grid BFP softmax output must give the FLOAT32 CE");
    freeTensor(label);
    freeTensor(floatP);
    freeTensor(bfpP);
}

/* Fused (p - y) = [-0.5, 0.25, 0.125, 0.125]; absmax 0.5 -> the SAME grid as
 * the input (stored 122, scale 0.03125), so the produced codes are
 * [-16, 8, 4, 4]. */
void testCrossEntropySoftmaxBackwardBfpRequantizesIntoFreshGrid(void) {
    int32_t codes[4] = {16, 8, 4, 4};
    int32_t sentinel[4] = {-9, -9, -9, -9};
    tensor_t *bfpP = buildBfpTensor1DWithCodes(4, 6, 8, codes, 122);
    float y[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *label = buildFloatTensor1D(4, y);
    tensor_t *loss = buildBfpTensor1DWithCodes(4, 6, 8, sentinel, 127);

    crossEntropySoftmaxBackward(bfpP, label, loss, NULL);

    int32_t got[4];
    unpackSignExtend(loss->data, 6, 0, got, 4);
    int32_t expected[4] = {-16, 8, 4, 4};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, 4);
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(122,
                                    ((bfpQConfig_t *)loss->quantization->qConfig)->exponents[0],
                                    "the produced grad must get a FRESH exponent (absmax 0.5)");
    freeTensor(loss);
    freeTensor(label);
    freeTensor(bfpP);
}

/* The loss boundary (arithmetic-bfp.md section 14): CE's float island reads
 * every operand through its own dtype, so the FLOAT32 arm (raw float casts)
 * runs only when ALL its operands are FLOAT32. A FLOAT32 softmax output with a
 * BFP grad wire is the island too: p - y in float, then ONE float -> BFP
 * conversion into the wire -- never floats written into a packed BFP payload. */
void testCrossEntropySoftmaxBackwardFloatOutputIntoABfpGradWireIsTheLossBoundary(void) {
    float exactP[4] = {0.5f, 0.25f, 0.125f, 0.125f};
    float y[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    int32_t sentinel[4] = {-9, -9, -9, -9};
    tensor_t *floatP = buildFloatTensor1D(4, exactP);
    tensor_t *label = buildFloatTensor1D(4, y);
    tensor_t *loss = buildBfpTensor1DWithCodes(4, 6, 8, sentinel, 127);

    crossEntropySoftmaxBackward(floatP, label, loss, NULL);

    int32_t got[4];
    unpackSignExtend(loss->data, 6, 0, got, 4);
    int32_t expected[4] = {-16, 8, 4, 4};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, 4);
    TEST_ASSERT_EQUAL_UINT8(122, ((bfpQConfig_t *)loss->quantization->qConfig)->exponents[0]);
    freeTensor(loss);
    freeTensor(label);
    freeTensor(floatP);
}

/* A BFP label next to a FLOAT32 output is dequantized, not read as floats:
 * y = [1, 0, 0, 0] is grid-exact at m 6 (code 16 on 2^-4, stored 123). */
void testCrossEntropyFloatOutputWithABfpLabelDequantizesTheLabel(void) {
    float exactP[4] = {0.5f, 0.25f, 0.125f, 0.125f};
    float y[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    int32_t yCodes[4] = {16, 0, 0, 0};
    tensor_t *floatP = buildFloatTensor1D(4, exactP);
    tensor_t *floatLabel = buildFloatTensor1D(4, y);
    tensor_t *bfpLabel = buildBfpTensor1DWithCodes(4, 6, 8, yCodes, 123);
    float viaFloat = crossEntropyForward(floatP, floatLabel, REDUCTION_SUM, NULL);
    float viaBfp = crossEntropyForward(floatP, bfpLabel, REDUCTION_SUM, NULL);
    TEST_ASSERT_EQUAL_FLOAT(viaFloat, viaBfp);

    tensor_t *gFloat = buildFloatTensor1D(4, (float[]){9, 9, 9, 9});
    crossEntropySoftmaxBackward(floatP, bfpLabel, gFloat, NULL);
    const float expected[4] = {-0.5f, 0.25f, 0.125f, 0.125f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)gFloat->data, 4);
    freeTensor(gFloat);
    freeTensor(bfpLabel);
    freeTensor(floatLabel);
    freeTensor(floatP);
}

/* BFP epic PR4: every scratch buffer in the fake-quant bodies
 * is a VLA sized from the MODEL OUTPUT's element count, but each convertTensor
 * walks its OWN source's count and the distribution scratch even borrows the
 * DISTRIBUTION's shape — so a label longer than the softmax output writes past
 * the end of a stack array. The mismatch is deliberately ONE element: large
 * enough to overrun, small enough that the unguarded child returns normally
 * instead of crashing, which is what makes the missing-guard RED a clean "exit
 * code 0" rather than a signal. */
void testCrossEntropyForwardBfpRejectsDistributionCountMismatch(void) {
    int32_t codes[4] = {16, 8, 4, 4};
    tensor_t *bfpP = buildBfpTensor1DWithCodes(4, 6, 8, codes, 122);
    float y[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *label = buildFloatTensor1D(5, y);

    ASSERT_EXITS_WITH_FAILURE((void)crossEntropyForward(bfpP, label, REDUCTION_SUM, NULL));

    freeTensor(label);
    freeTensor(bfpP);
}

void testCrossEntropySoftmaxBackwardBfpRejectsDistributionCountMismatch(void) {
    int32_t codes[4] = {16, 8, 4, 4};
    int32_t sentinel[4] = {-9, -9, -9, -9};
    tensor_t *bfpP = buildBfpTensor1DWithCodes(4, 6, 8, codes, 122);
    float y[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *label = buildFloatTensor1D(5, y);
    tensor_t *loss = buildBfpTensor1DWithCodes(4, 6, 8, sentinel, 127);

    ASSERT_EXITS_WITH_FAILURE(crossEntropySoftmaxBackward(bfpP, label, loss, NULL));

    freeTensor(loss);
    freeTensor(label);
    freeTensor(bfpP);
}

/* BFP epic PR4: the produced LOSS wire is the
 * third operand with the same hole, and it needs its own case because a CORRECT
 * distribution is what carries the call past the distribution guard. The loss
 * scratch is a VLA sized from the model output's count but borrows the LOSS's
 * shape_t, so the closing convertTensor(&lossFloat, loss) walks the LOSS's count
 * and reads past the end of that stack array; a shorter loss wire would instead
 * emit a silently truncated gradient. */
void testCrossEntropySoftmaxBackwardBfpRejectsLossWireCountMismatch(void) {
    int32_t codes[4] = {16, 8, 4, 4};
    int32_t sentinel[5] = {-9, -9, -9, -9, -9};
    tensor_t *bfpP = buildBfpTensor1DWithCodes(4, 6, 8, codes, 122);
    float y[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *label = buildFloatTensor1D(4, y);
    tensor_t *loss = buildBfpTensor1DWithCodes(5, 6, 8, sentinel, 127);

    ASSERT_EXITS_WITH_FAILURE(crossEntropySoftmaxBackward(bfpP, label, loss, NULL));

    freeTensor(loss);
    freeTensor(label);
    freeTensor(bfpP);
}

/* BFP epic PR4, kept by #153: the operand shape guard must sit
 * at the PUBLIC entry, not inside the fake-quant helper — otherwise the
 * FLOAT32 arm keeps the whole hole (crossEntropySoftmaxBackwardFloat writes
 * lossFloat[i] for i < the model output's count straight into the caller's
 * buffer, so a SHORTER loss wire is an out-of-bounds heap WRITE). These two
 * cases use a LONGER operand on purpose — the float arm ignores the surplus,
 * so the unguarded child returns normally and the missing-guard RED is a
 * clean "exit code 0" rather than a signal — while the guard they pin rejects
 * both directions. They are also what pins the HOIST itself: a guard pushed
 * back down into the fake-quant helper would leave these two red while every
 * BFP death test stayed green. */
void testCrossEntropyForwardFloat32RejectsDistributionCountMismatch(void) {
    float p[4] = {0.5f, 0.25f, 0.125f, 0.125f};
    float y[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *floatP = buildFloatTensor1D(4, p);
    tensor_t *label = buildFloatTensor1D(5, y);

    ASSERT_EXITS_WITH_FAILURE((void)crossEntropyForward(floatP, label, REDUCTION_SUM, NULL));

    freeTensor(label);
    freeTensor(floatP);
}

void testCrossEntropySoftmaxBackwardFloat32RejectsLossWireCountMismatch(void) {
    float p[4] = {0.5f, 0.25f, 0.125f, 0.125f};
    float y[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float lossValues[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *floatP = buildFloatTensor1D(4, p);
    tensor_t *label = buildFloatTensor1D(4, y);
    tensor_t *loss = buildFloatTensor1D(5, lossValues);

    ASSERT_EXITS_WITH_FAILURE(crossEntropySoftmaxBackward(floatP, label, loss, NULL));

    freeTensor(loss);
    freeTensor(label);
    freeTensor(floatP);
}

/* #153: the CE dispatchers share MSE's shape check. A rank-1 softmax output
 * would leave the batch axis to be guessed (crossEntropyForwardFloat treats
 * it as one row); equal-count operands with another shape are rejected. */
void testCrossEntropyForwardRejectsRank1Output(void) {
    size_t dims[] = {3};
    tensor_t *softmaxOutput = buildFloatTensorShaped(dims, 1, (float[]){0.6f, 0.3f, 0.1f});
    tensor_t *distribution = buildFloatTensorShaped(dims, 1, (float[]){1.f, 0.f, 0.f});
    ASSERT_EXITS_WITH_FAILURE(
        (void)crossEntropyForward(softmaxOutput, distribution, REDUCTION_MEAN, NULL));
    freeTensor(distribution);
    freeTensor(softmaxOutput);
}

void testCrossEntropyForwardRejectsLeadingAxisMismatch(void) {
    size_t outDims[] = {2, 3};
    size_t distDims[] = {3, 2};
    tensor_t *softmaxOutput =
        buildFloatTensorShaped(outDims, 2, (float[]){0.6f, 0.3f, 0.1f, 0.2f, 0.2f, 0.6f});
    tensor_t *distribution =
        buildFloatTensorShaped(distDims, 2, (float[]){1.f, 0.f, 0.f, 0.f, 0.f, 1.f});
    ASSERT_EXITS_WITH_FAILURE(
        (void)crossEntropyForward(softmaxOutput, distribution, REDUCTION_MEAN, NULL));
    freeTensor(distribution);
    freeTensor(softmaxOutput);
}

void testCrossEntropySoftmaxBackwardRejectsDistributionRankMismatch(void) {
    size_t outDims[] = {1, 3};
    size_t distDims[] = {1, 3, 1};
    tensor_t *softmaxOutput = buildFloatTensorShaped(outDims, 2, (float[]){0.6f, 0.3f, 0.1f});
    tensor_t *distribution = buildFloatTensorShaped(distDims, 3, (float[]){1.f, 0.f, 0.f});
    tensor_t *lossWire = buildFloatTensorShaped(outDims, 2, (float[]){0.f, 0.f, 0.f});
    ASSERT_EXITS_WITH_FAILURE(
        crossEntropySoftmaxBackward(softmaxOutput, distribution, lossWire, NULL));
    freeTensor(lossWire);
    freeTensor(distribution);
    freeTensor(softmaxOutput);
}

void testCrossEntropyForward_NonFiniteSoftmaxReturnsNanInsteadOfAborting(void) {
    size_t inputSize = 3;
    float softmaxData[] = {0.5f, NAN, 0.5f};
    size_t dims[] = {1, inputSize};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t q;
    initFloat32Quantization(&q);
    tensor_t softmaxOutput;
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxData, &shape, &q, NULL);

    float distData[] = {0.f, 1.f, 0.f};
    tensor_t distribution;
    setTensorValues(&distribution, (uint8_t *)distData, &shape, &q, NULL);

    float loss = crossEntropyForward(&softmaxOutput, &distribution, REDUCTION_MEAN, NULL);

    TEST_ASSERT_TRUE(isnan(loss));
}

/* Sibling of the NAN pin above: an INFINITY softmax element must also yield
 * NaN rather than abort (#446) -- NAN and INFINITY are both "non-finite" but
 * distinct bit patterns, so the NAN case alone would not catch an isfinite()
 * check narrowed to isnan(). */
void testCrossEntropyForward_InfiniteSoftmaxReturnsNanInsteadOfAborting(void) {
    size_t inputSize = 3;
    float softmaxData[] = {0.5f, INFINITY, 0.5f};
    size_t dims[] = {1, inputSize};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t q;
    initFloat32Quantization(&q);
    tensor_t softmaxOutput;
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxData, &shape, &q, NULL);

    float distData[] = {0.f, 1.f, 0.f};
    tensor_t distribution;
    setTensorValues(&distribution, (uint8_t *)distData, &shape, &q, NULL);

    float loss = crossEntropyForward(&softmaxOutput, &distribution, REDUCTION_MEAN, NULL);

    TEST_ASSERT_TRUE(isnan(loss));
}

void setUp() {}
void tearDown() {}

/* ---- the loss gradient rounds with the loss's BACKWARD stream ---- */

/* Python, tags as numbers (BACKWARD 5, LOSS 17):
 *   uv run python -c "from examples._shared.keyed_rng import (fold, key_from_seed as kfs,
 *   open_stream as o); print(hex(o(fold(fold(kfs(0x1a7e), 5, 0), 17, 0))))" */
static conversionFunction_t ceSavedCell;
static size_t ceCellCalls;
static bool ceCellHasStream;
static uint32_t ceCellState;

static void ceRecordingCell(tensor_t *in, tensor_t *out, rngStream_t *rs) {
    ceCellCalls++;
    ceCellHasStream = rs != NULL;
    ceCellState = (rs != NULL) ? rs->state : 0u;
    ceSavedCell(in, out, rs);
}

static tensor_t *ceBuildSym1x3(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
}

/* A SYM_INT32 softmax output takes the fake-quant backward: the raw (p - y)
 * is requantized into the SYM_INT32 grad with (BACKWARD, LOSS) of the context
 * the loss receives; the operand dequantizations go into FLOAT32 (another
 * cell) and draw nothing. */
void testCrossEntropyBackwardRequantizesTheGradWithItsLossStream(void) {
    layerExec_t exec = {.key = rngKeyFromSeed(0x1a7eu)};
    rngKey_t phase;
    tensor_t *output = ceBuildSym1x3();
    tensor_t *label = ceBuildSym1x3();
    tensor_t *result = ceBuildSym1x3();
    ceSavedCell = conversionMatrix[FLOAT32][SYM_INT32];
    conversionMatrix[FLOAT32][SYM_INT32] = ceRecordingCell;
    ceCellCalls = 0;

    crossEntropySoftmaxBackward(output, label, result, &exec);

    conversionMatrix[FLOAT32][SYM_INT32] = ceSavedCell;
    freeTensor(result);
    freeTensor(label);
    freeTensor(output);
    TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)ceCellCalls);
    TEST_ASSERT_TRUE(ceCellHasStream);
    rngKey_t lossKey =
        rngFold(*layerExecPhaseKey(&exec, RNG_TAG_BACKWARD, &phase), RNG_TAG_LOSS, 0u);
    TEST_ASSERT_EQUAL_HEX32(rngOpen(lossKey).state, ceCellState);
    TEST_ASSERT_EQUAL_HEX32(0x7114be1fu, ceCellState);
}

/* ---- head bfp: the integer combined gradient p - y (ruling R2) ----------
 * p is the integer Softmax's BFP output, y a FLOAT32 one-hot label (read by
 * bit pattern: 1.0f / +-0.0f, no float compare). Each element is an int32
 * term on its own grid: a non-label element is its code on its p group's
 * exponent E_g; the label element subtracts 1.0 = 2^-E on the grid E =
 * clamp(E_g, -30, 0), exact for E_g in [-30, 0] -- also when 1.0 needs code
 * 2^(m-1), one beyond the m-bit maximum. The raw of every target group then
 * takes the data rule (E_t = max(E + bitlen|term|) - 31), and the funnel
 * re-blocks it onto the loss-grad wire by the wire's mode. */

/* [rows, cols] BFP tensor (m, e = 8), grouped (groupSize > 0) or per tensor,
 * from codes and BIASED stored exponents. */
static tensor_t *ceBfp2d(size_t rows, size_t cols, uint8_t m, size_t numGroups, size_t groupSize,
                         const int32_t *codes, const uint8_t *stored, roundingMode_t mode) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    quantization_t *q = groupSize == 0
                            ? quantizationInitBfp(m, 8, mode)
                            : quantizationInitBfpGrouped(m, 8, mode, numGroups, groupSize);
    tensor_t *t = initTensor(shape, q, NULL);
    size_t n = rows * cols;
    if (codes != NULL) {
        int32_t tmp[64];
        TEST_ASSERT_TRUE(n <= 64);
        memcpy(tmp, codes, n * sizeof(int32_t));
        byteConversion((uint8_t *)tmp, 32, t->data, m, n);
    }
    bfpQConfig_t *qc = q->qConfig;
    for (size_t g = 0; g < qc->numGroups; g++) {
        qc->exponents[g] = stored != NULL ? stored[g] : 127;
    }
    return t;
}

static tensor_t *ceFloat2d(size_t rows, size_t cols, const float *values) {
    size_t dims[2] = {rows, cols};
    return buildFloatTensorShaped(dims, 2, (float *)values);
}

typedef struct ceIntCase {
    size_t rows, cols;
    uint8_t m;
    size_t pNumGroups, pGroupSize; /* p's grouping (0 = per tensor) */
    const int32_t *codes;
    const uint8_t *stored; /* biased, e = 8 */
    const size_t *label;   /* one class per row */
    size_t rawNumGroups, rawGroupSize;
} ceIntCase_t;

/* The int64 reference, from the definition: the exact value of p_i - y_i on
 * the grid e_i = min(E_g, 0) (exact for every E_g >= -62), then the data-rule
 * re-grid of BfpScatterRef.h. Exact wherever the kernel accepts p (a label
 * below E_g = -30 must sit on the 2^-30 grid). */
static void ceIntReference(const ceIntCase_t *c, int32_t *mant, int16_t *rawE) {
    size_t n = c->rows * c->cols;
    int64_t sum[64];
    int eAcc[64];
    for (size_t i = 0; i < n; i++) {
        size_t g = c->pGroupSize == 0 ? 0 : i / c->pGroupSize;
        int eg = (int)c->stored[g] - 127;
        int e = eg < 0 ? eg : 0;
        bool one = (i % c->cols) == c->label[i / c->cols];
        sum[i] = (int64_t)c->codes[i] * ((int64_t)1 << (eg - e)) - (one ? (int64_t)1 << (-e) : 0);
        eAcc[i] = e;
    }
    bfpRefStats_t st = {0};
    bfpRefRawFromRows(sum, eAcc, n, c->rawGroupSize, mant, rawE, &st);
}

static void ceOneHot(const ceIntCase_t *c, float *y) {
    for (size_t i = 0; i < c->rows * c->cols; i++) {
        y[i] = (i % c->cols) == c->label[i / c->cols] ? 1.0f : 0.0f;
    }
}

static void ceCheckIntegerRaw(const ceIntCase_t *c) {
    size_t n = c->rows * c->cols;
    size_t dims[2] = {c->rows, c->cols};
    bfpScratchFixture_t p;
    bfpScratchInit(&p, 2, dims, c->codes, c->stored, c->pGroupSize == 0 ? 1 : c->pNumGroups,
                   c->pGroupSize, c->m, 8);
    float y[64];
    ceOneHot(c, y);
    tensor_t *label = ceFloat2d(c->rows, c->cols, y);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 2, dims, c->rawNumGroups, c->rawGroupSize);
    tensor_t *ops[1] = {&p.tensor};

    crossEntropySoftmaxBackwardKernelBfp(ops, 1, &raw.tensor, NULL, label, NULL);

    int32_t mant[64];
    int16_t rawE[16];
    ceIntReference(c, mant, rawE);
    assertRawEquals(&raw, mant, n, rawE, c->rawNumGroups);
    freeTensor(label);
}

/* p_max near 1 (code 127 on 2^-7 at m 8: 0.9921875) with the label at the
 * max (p - 1 = -1/128) and at a non-max class (the 1.0 is code 128 = 2^(m-1)
 * on that grid, one beyond the 8-bit maximum: exact in int32); a label whose
 * p is 0; grouped p with groups 6 binades apart into a per-tensor raw and
 * into raw groups that straddle the p groups; two rows; m 16. */
void testCeIntegerHeadRawMatchesTheInt64Reference(void) {
    int32_t near1[6] = {127, 1, 0, 0, 0, 0};
    uint8_t e7[1] = {120};
    size_t atMax[1] = {0}, atOther[1] = {3}, atZero[1] = {5};
    ceIntCase_t a = {1, 6, 8, 1, 0, near1, e7, atMax, 1, 0};
    ceCheckIntegerRaw(&a);
    a.label = atOther;
    ceCheckIntegerRaw(&a);
    a.label = atZero;
    ceCheckIntegerRaw(&a);

    int32_t grouped[6] = {90, -3, 17, 64, 5, 127};
    uint8_t ge[3] = {120, 114, 126};
    size_t l2[1] = {2};
    ceIntCase_t b = {1, 6, 8, 3, 2, grouped, ge, l2, 1, 0};
    ceCheckIntegerRaw(&b);
    b.rawNumGroups = 2;
    b.rawGroupSize = 3; /* raw groups straddle p groups 1 */
    ceCheckIntegerRaw(&b);
    b.label = atOther;
    ceCheckIntegerRaw(&b);

    int32_t two[6] = {100, 20, 7, 3, 120, 5};
    uint8_t te[2] = {120, 121};
    size_t lab2[2] = {1, 1};
    ceIntCase_t r2 = {2, 3, 8, 2, 3, two, te, lab2, 2, 3};
    ceCheckIntegerRaw(&r2);

    int32_t wide[6] = {32767, 12, -1, 0, 4000, 9};
    uint8_t w15[1] = {112}; /* E = -15: 1.0 = 2^15 codes, one beyond the m16 maximum */
    size_t lw[1] = {4};
    ceIntCase_t c16 = {1, 6, 16, 1, 0, wide, w15, lw, 1, 0};
    ceCheckIntegerRaw(&c16);
}

/* 1.0 exactly beyond the m-bit maximum: p = 0 at the label on 2^-7, so the
 * term is -128 = -2^(m-1) codes -- the raw holds exactly -1.0. An all-zero p
 * group in the zero state (E = 0) holding the label: the term is -1 on 2^0. */
void testCeIntegerHeadSubtractsOneExactly(void) {
    int32_t codes[4] = {127, 0, 0, 0};
    uint8_t e7[1] = {120};
    size_t lab[1] = {1};
    ceIntCase_t a = {1, 4, 8, 1, 0, codes, e7, lab, 1, 0};
    ceCheckIntegerRaw(&a);
    size_t dims[2] = {1, 4};
    bfpScratchFixture_t p;
    bfpScratchInit(&p, 2, dims, codes, e7, 1, 0, 8, 8);
    tensor_t *label = ceFloat2d(1, 4, (float[]){0.f, 1.f, 0.f, 0.f});
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 2, dims, 1, 0);
    tensor_t *ops[1] = {&p.tensor};
    crossEntropySoftmaxBackwardKernelBfp(ops, 1, &raw.tensor, NULL, label, NULL);
    TEST_ASSERT_TRUE(bfpValueExactlyEqual(raw.mant[1], raw.exps[0], -1, 0));
    TEST_ASSERT_TRUE(bfpValueExactlyEqual(raw.mant[0], raw.exps[0], 127, -7));

    int32_t zeroGroup[4] = {120, 3, 0, 0};
    uint8_t zg[2] = {120, 127};
    size_t lz[1] = {3};
    ceIntCase_t z = {1, 4, 8, 2, 2, zeroGroup, zg, lz, 1, 0};
    ceCheckIntegerRaw(&z);
    z.rawNumGroups = 2;
    z.rawGroupSize = 2;
    ceCheckIntegerRaw(&z);
    freeTensor(label);
}

/* A label whose p group sits below 2^-30 (E_g < -30): 1.0 cannot share that
 * grid in int32, so the term is placed on 2^-30. That is exact only for a p
 * ON the 2^-30 grid (the code's low -30 - E_g bits are zero) -- every p the
 * integer Softmax produces (Q30, re-blocked onto a finer group by exact left
 * shifts). A finer fraction would be lost before the SR re-block, so it
 * fails fast. */
static void ceLabelOnFineGrid(uint8_t m, int32_t labelCode, int eG) {
    int32_t codes[4] = {127, 64, labelCode, 3};
    uint8_t st[2] = {120, (uint8_t)(eG + 127)};
    size_t lab[1] = {2};
    ceIntCase_t c = {1, 4, m, 2, 2, codes, st, lab, 2, 2};
    ceCheckIntegerRaw(&c);
}
static void ceDieLabelCode64AtMinus38(void) { /* 2^-32: the review's case */
    ceLabelOnFineGrid(8, 64, -38);
}
static void ceDieLabelCode101AtMinus40(void) {
    ceLabelOnFineGrid(8, 101, -40);
}
static void ceDieLabelCode127AtMinus37(void) { /* 0.99 of 2^-30 */
    ceLabelOnFineGrid(8, 127, -37);
}
static void ceDieLabelCode3AtMinus31(void) { /* one stray 2^-31 */
    ceLabelOnFineGrid(8, 3, -31);
}
static void ceDieLabelCode1AtMinus70(void) { /* 40 bits below: past any int32 shift */
    ceLabelOnFineGrid(8, 1, -70);
}

void testCeIntegerHeadRejectsALabelFinerThanTwoToMinus30(void) {
    const char *msg = "is not on the 2^-30 grid";
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieLabelCode64AtMinus38());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieLabelCode101AtMinus40());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieLabelCode127AtMinus37());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieLabelCode3AtMinus31());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieLabelCode1AtMinus70());
}

/* On the 2^-30 grid below E_g = -30 the term is exact (the int64 reference
 * holds p - 1 on 2^E_g): 126 * 2^-31 = 63 * 2^-30, a zero p at 2^-40, and
 * m 16's 1024 * 2^-40 = 2^-30. E_g = -30 itself is the plain exact case. */
void testCeIntegerHeadPlacesALabelOnTheTwoToMinus30GridExactly(void) {
    ceLabelOnFineGrid(8, 126, -31);
    ceLabelOnFineGrid(8, 0, -40);
    ceLabelOnFineGrid(16, 1024, -40);
    ceLabelOnFineGrid(8, 3, -30);
}

/* Layer level: the raw re-blocked onto the loss-grad wire by the WIRE's
 * mode; under SR one draw per element in element order from the op's OUT
 * stream: rngFold(rngFold(rngFold(key, BACKWARD, 0), OP, 0), OUT, 0). */
static void ceIntoBfpWire(roundingMode_t mode, uint64_t seed, size_t gradGroups, size_t gradSize) {
    int32_t codes[6] = {90, -3, 17, 64, 5, 127};
    uint8_t ge[3] = {120, 114, 126};
    size_t lab[1] = {2};
    ceIntCase_t c = {1, 6, 8, 3, 2, codes, ge, lab, gradSize == 0 ? 1 : gradGroups, gradSize};
    tensor_t *p = ceBfp2d(1, 6, 8, 3, 2, codes, ge, HALF_AWAY);
    float y[6];
    ceOneHot(&c, y);
    tensor_t *label = ceFloat2d(1, 6, y);
    tensor_t *grad = ceBfp2d(1, 6, 8, gradGroups, gradSize, NULL, NULL, mode);
    layerExec_t exec = {.key = rngKeyFromSeed(seed)};

    crossEntropySoftmaxBackward(p, label, grad, mode == SR_HALF_AWAY ? &exec : NULL);

    int32_t mant[64];
    int16_t rawE[16];
    ceIntReference(&c, mant, rawE);
    rngStream_t out = rngOpen(
        rngFold(rngFold(rngFold(exec.key, RNG_TAG_BACKWARD, 0u), RNG_TAG_OP, 0u), RNG_TAG_OUT, 0u));
    int32_t refCodes[6];
    uint8_t refStored[3];
    bfpRefReblockRaw(mant, rawE, 6, gradSize, 8, 8, mode == SR_HALF_AWAY ? &out : NULL, refCodes,
                     refStored);
    int32_t got[6];
    unpackSignExtend(grad->data, 8, 0, got, 6);
    TEST_ASSERT_EQUAL_INT32_ARRAY(refCodes, got, 6);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refStored,
                                  ((bfpQConfig_t *)grad->quantization->qConfig)->exponents,
                                  gradSize == 0 ? 1 : gradGroups);
    freeTensor(grad);
    freeTensor(label);
    freeTensor(p);
}

void testCeIntegerHeadReblocksOntoTheLossGradWireByItsMode(void) {
    ceIntoBfpWire(HALF_AWAY, 0u, 1, 0);
    ceIntoBfpWire(SR_HALF_AWAY, 41u, 1, 0);
    ceIntoBfpWire(HALF_AWAY, 0u, 2, 3);
    ceIntoBfpWire(SR_HALF_AWAY, 42u, 2, 3);
}

/* SR statistics of the re-block: p = [101, 13, 9, 5] / 128, label 0, into an
 * m 4 grad wire (code step 2^-5: every value between two codes), many keys: every output
 * code is one of its two neighbours and the mean is the exact value within
 * 5 sigma (per element, Bernoulli on the dropped fraction). */
void testCeIntegerHeadSrReblockIsUnbiased(void) {
    int32_t codes[4] = {101, 13, 9, 5};
    uint8_t e7[1] = {120};
    tensor_t *p = ceBfp2d(1, 4, 8, 1, 0, codes, e7, HALF_AWAY);
    tensor_t *label = ceFloat2d(1, 4, (float[]){1.f, 0.f, 0.f, 0.f});
    const double exact[4] = {101.0 / 128 - 1.0, 13.0 / 128, 9.0 / 128, 5.0 / 128};
    const size_t trials = 4000;
    double sum[4] = {0}, sumSq[4] = {0};
    for (size_t t = 0; t < trials; t++) {
        tensor_t *grad = ceBfp2d(1, 4, 4, 1, 0, NULL, NULL, SR_HALF_AWAY);
        layerExec_t exec = {.key = rngKeyFromSeed(1000u + t)};
        crossEntropySoftmaxBackward(p, label, grad, &exec);
        int32_t got[4];
        unpackSignExtend(grad->data, 4, 0, got, 4);
        int e = (int)((bfpQConfig_t *)grad->quantization->qConfig)->exponents[0] - 127;
        TEST_ASSERT_EQUAL_INT(-5, e); /* |p - y| max 27/128 = 6.75 * 2^-5 <= 7 * 2^-5 */
        for (size_t i = 0; i < 4; i++) {
            double v = ldexp((double)got[i], e);
            double lo = floor(exact[i] / ldexp(1.0, e)) * ldexp(1.0, e);
            TEST_ASSERT_TRUE(v == lo || v == lo + ldexp(1.0, e));
            sum[i] += v;
            sumSq[i] += v * v;
        }
        freeTensor(grad);
    }
    for (size_t i = 0; i < 4; i++) {
        double mean = sum[i] / (double)trials;
        double var = sumSq[i] / (double)trials - mean * mean;
        double sigma = sqrt((var > 0 ? var : 0) / (double)trials) + 1e-12;
        TEST_ASSERT_TRUE_MESSAGE(fabs(mean - exact[i]) <= 5.0 * sigma + ldexp(1.0, -20),
                                 "SR re-block is unbiased");
    }
    freeTensor(label);
    freeTensor(p);
}

/* Half-specified heads fail fast: a BFP p needs a BFP loss-grad wire, a
 * FLOAT32 one-hot label (0 or 1 by bit pattern), and a probability <= 1 at
 * the label. */
static void ceDieBfpPFloatGrad(void) {
    int32_t codes[4] = {64, 32, 16, 16};
    uint8_t e7[1] = {120};
    tensor_t *p = ceBfp2d(1, 4, 8, 1, 0, codes, e7, HALF_AWAY);
    tensor_t *label = ceFloat2d(1, 4, (float[]){1.f, 0.f, 0.f, 0.f});
    tensor_t *grad = ceFloat2d(1, 4, (float[]){0.f, 0.f, 0.f, 0.f});
    crossEntropySoftmaxBackward(p, label, grad, NULL);
}
static void ceDieSymLabel(void) {
    int32_t codes[3] = {64, 32, 32};
    uint8_t e7[1] = {120};
    tensor_t *p = ceBfp2d(1, 3, 8, 1, 0, codes, e7, HALF_AWAY);
    tensor_t *label = ceBuildSym1x3();
    tensor_t *grad = ceBfp2d(1, 3, 8, 1, 0, NULL, NULL, HALF_AWAY);
    crossEntropySoftmaxBackward(p, label, grad, NULL);
}
static void ceDieSoftLabel(void) {
    int32_t codes[4] = {64, 32, 16, 16};
    uint8_t e7[1] = {120};
    tensor_t *p = ceBfp2d(1, 4, 8, 1, 0, codes, e7, HALF_AWAY);
    tensor_t *label = ceFloat2d(1, 4, (float[]){0.5f, 0.5f, 0.f, 0.f});
    tensor_t *grad = ceBfp2d(1, 4, 8, 1, 0, NULL, NULL, HALF_AWAY);
    crossEntropySoftmaxBackward(p, label, grad, NULL);
}
static void ceDieProbabilityAboveOne(void) {
    int32_t codes[4] = {3, 0, 0, 0};
    uint8_t e1[1] = {128}; /* E = +1: p = 6 */
    tensor_t *p = ceBfp2d(1, 4, 8, 1, 0, codes, e1, HALF_AWAY);
    tensor_t *label = ceFloat2d(1, 4, (float[]){1.f, 0.f, 0.f, 0.f});
    tensor_t *grad = ceBfp2d(1, 4, 8, 1, 0, NULL, NULL, HALF_AWAY);
    crossEntropySoftmaxBackward(p, label, grad, NULL);
}

void testCeIntegerHeadRejectsAHalfSpecifiedBoundary(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "the integer head needs a BFP loss-grad wire",
                             ceDieBfpPFloatGrad());
    ASSERT_EXITS_WITH_OUTPUT(1, "the integer head needs a FLOAT32 or BFP one-hot label",
                             ceDieSymLabel());
    ASSERT_EXITS_WITH_OUTPUT(1, "is not one-hot (0 or 1)", ceDieSoftLabel());
    ASSERT_EXITS_WITH_OUTPUT(1, "a probability above 1", ceDieProbabilityAboveOne());
}

/* ---- BFP one-hot labels: the integer head reads them in integers ----
 * A dataset converted offline stores its one-hot labels as BFP (0 and 1.0 are
 * exact). The label rides as the op's second operand: unpacked codes under
 * its own group exponents. Element i is the class iff code * 2^E == 1
 * exactly (E <= 0, code == 2^-E), not the class iff code == 0; anything else
 * fails fast. The raw must equal the FLOAT32-label raw on every case. */
static void ceCheckIntegerRawBfpLabel(const ceIntCase_t *c, const int32_t *yCodes,
                                      const uint8_t *yStored, size_t yNumGroups, size_t yGroupSize,
                                      uint8_t yM) {
    size_t n = c->rows * c->cols;
    size_t dims[2] = {c->rows, c->cols};
    bfpScratchFixture_t p;
    bfpScratchInit(&p, 2, dims, c->codes, c->stored, c->pGroupSize == 0 ? 1 : c->pNumGroups,
                   c->pGroupSize, c->m, 8);
    bfpScratchFixture_t y;
    bfpScratchInit(&y, 2, dims, yCodes, yStored, yNumGroups, yGroupSize, yM, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 2, dims, c->rawNumGroups, c->rawGroupSize);
    tensor_t *ops[2] = {&p.tensor, &y.tensor};

    crossEntropySoftmaxBackwardKernelBfp(ops, 2, &raw.tensor, NULL, NULL, NULL);

    int32_t mant[64];
    int16_t rawE[16];
    ceIntReference(c, mant, rawE);
    assertRawEquals(&raw, mant, n, rawE, c->rawNumGroups);
}

void testCeIntegerHeadReadsABfpOneHotLabelExactly(void) {
    /* m 8 per tensor: the offline conversion's 1.0 is code 64 on 2^-6 */
    int32_t near1[6] = {127, 1, 0, 0, 0, 0};
    uint8_t e7[1] = {120};
    size_t atMax[1] = {0}, atOther[1] = {3};
    ceIntCase_t a = {1, 6, 8, 1, 0, near1, e7, atMax, 1, 0};
    ceCheckIntegerRawBfpLabel(&a, (int32_t[]){64, 0, 0, 0, 0, 0}, (uint8_t[]){121}, 1, 0, 8);
    a.label = atOther;
    ceCheckIntegerRawBfpLabel(&a, (int32_t[]){0, 0, 0, 64, 0, 0}, (uint8_t[]){121}, 1, 0, 8);
    /* 1.0 as code 1 on 2^0, and as code 2^14 on 2^-14 at m 16 */
    ceCheckIntegerRawBfpLabel(&a, (int32_t[]){0, 0, 0, 1, 0, 0}, (uint8_t[]){127}, 1, 0, 8);
    ceCheckIntegerRawBfpLabel(&a, (int32_t[]){0, 0, 0, 16384, 0, 0}, (uint8_t[]){113}, 1, 0, 16);

    /* grouped p and a grouped label of ANOTHER geometry (3 groups of 2 vs 2
     * of 3): the class group on 2^-6, a zero group under an odd exponent */
    int32_t grouped[6] = {90, -3, 17, 64, 5, 127};
    uint8_t ge[3] = {120, 114, 126};
    size_t l4[1] = {4};
    ceIntCase_t b = {1, 6, 8, 3, 2, grouped, ge, l4, 2, 3};
    ceCheckIntegerRawBfpLabel(&b, (int32_t[]){0, 0, 0, 0, 64, 0}, (uint8_t[]){3, 121}, 2, 3, 8);

    /* two rows, each its own class; the label grouped per row */
    int32_t two[6] = {100, 20, 7, 3, 120, 5};
    uint8_t te[2] = {120, 121};
    size_t lab2[2] = {0, 2};
    ceIntCase_t r2 = {2, 3, 8, 2, 3, two, te, lab2, 2, 3};
    ceCheckIntegerRawBfpLabel(&r2, (int32_t[]){64, 0, 0, 0, 0, 64}, (uint8_t[]){121, 121}, 2, 3, 8);
}

/* Layer level: a BFP label gives the same loss-grad wire as the FLOAT32
 * one-hot, under HALF_AWAY and under SR (same op, same OUT stream). */
static void ceLayerBfpVsFloatLabel(roundingMode_t mode, uint64_t seed) {
    int32_t codes[6] = {90, -3, 17, 64, 5, 127};
    uint8_t ge[3] = {120, 114, 126};
    tensor_t *p = ceBfp2d(1, 6, 8, 3, 2, codes, ge, HALF_AWAY);
    tensor_t *yFloat = ceFloat2d(1, 6, (float[]){0.f, 0.f, 1.f, 0.f, 0.f, 0.f});
    tensor_t *yBfp =
        ceBfp2d(1, 6, 8, 1, 0, (int32_t[]){0, 0, 64, 0, 0, 0}, (uint8_t[]){121}, HALF_AWAY);
    tensor_t *gFloat = ceBfp2d(1, 6, 8, 2, 3, NULL, NULL, mode);
    tensor_t *gBfp = ceBfp2d(1, 6, 8, 2, 3, NULL, NULL, mode);
    layerExec_t exec = {.key = rngKeyFromSeed(seed)};

    crossEntropySoftmaxBackward(p, yFloat, gFloat, &exec);
    crossEntropySoftmaxBackward(p, yBfp, gBfp, &exec);

    TEST_ASSERT_EQUAL_UINT8_ARRAY(gFloat->data, gBfp->data, 6);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(((bfpQConfig_t *)gFloat->quantization->qConfig)->exponents,
                                  ((bfpQConfig_t *)gBfp->quantization->qConfig)->exponents, 2);
    freeTensor(gBfp);
    freeTensor(gFloat);
    freeTensor(yBfp);
    freeTensor(yFloat);
    freeTensor(p);
}

void testCeIntegerHeadGivesTheSameGradForABfpAndAFloatOneHot(void) {
    ceLayerBfpVsFloatLabel(HALF_AWAY, 5u);
    ceLayerBfpVsFloatLabel(SR_HALF_AWAY, 43u);
    ceLayerBfpVsFloatLabel(SR_HALF_AWAY, 44u);
}

static void ceBfpLabelDies(int32_t code, uint8_t stored) {
    int32_t codes[4] = {64, 32, 16, 16};
    uint8_t e7[1] = {120};
    tensor_t *p = ceBfp2d(1, 4, 8, 1, 0, codes, e7, HALF_AWAY);
    tensor_t *label = ceBfp2d(1, 4, 8, 1, 0, (int32_t[]){0, code, 0, 0}, &stored, HALF_AWAY);
    tensor_t *grad = ceBfp2d(1, 4, 8, 1, 0, NULL, NULL, HALF_AWAY);
    crossEntropySoftmaxBackward(p, label, grad, NULL);
}
static void ceDieBfpLabel65(void) { /* 65/64 */
    ceBfpLabelDies(65, 121);
}
static void ceDieBfpLabelMinusOne(void) {
    ceBfpLabelDies(-64, 121);
}
static void ceDieBfpLabelHalf(void) {
    ceBfpLabelDies(32, 121);
}
static void ceDieBfpLabelTwo(void) { /* code 1 on 2^1 */
    ceBfpLabelDies(1, 128);
}

void testCeIntegerHeadRejectsABfpLabelThatIsNotOneHot(void) {
    const char *msg = "is not one-hot (0 or 1)";
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieBfpLabel65());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieBfpLabelMinusOne());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieBfpLabelHalf());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieBfpLabelTwo());
}

/* Review E #4: the label is a ONE-HOT per class row, not per element: a row
 * with two exact ones (p - y would be the gradient of a different loss) or
 * none (no class) fails fast, for a BFP and a FLOAT32 label alike; several
 * rows, each with its own one, stay legal (testCeIntegerHeadReadsABfp...). */
static void ceLabelRowDies(const int32_t *yCodes, const float *yFloat) {
    int32_t codes[4] = {64, 32, 16, 16};
    uint8_t e7[1] = {120};
    tensor_t *p = ceBfp2d(1, 4, 8, 1, 0, codes, e7, HALF_AWAY);
    tensor_t *label = yCodes != NULL ? ceBfp2d(1, 4, 8, 1, 0, yCodes, (uint8_t[]){121}, HALF_AWAY)
                                     : ceFloat2d(1, 4, yFloat);
    tensor_t *grad = ceBfp2d(1, 4, 8, 1, 0, NULL, NULL, HALF_AWAY);
    crossEntropySoftmaxBackward(p, label, grad, NULL);
}
static void ceDieBfpTwoOnes(void) {
    ceLabelRowDies((int32_t[]){64, 64, 0, 0}, NULL);
}
static void ceDieBfpNoOne(void) {
    ceLabelRowDies((int32_t[]){0, 0, 0, 0}, NULL);
}
static void ceDieFloatTwoOnes(void) {
    ceLabelRowDies(NULL, (float[]){1.f, 0.f, 1.f, 0.f});
}
static void ceDieFloatNoOne(void) {
    ceLabelRowDies(NULL, (float[]){0.f, 0.f, 0.f, 0.f});
}
/* two rows: the first one-hot, the second with two ones */
static void ceDieSecondRowTwoOnes(void) {
    int32_t codes[6] = {100, 20, 7, 3, 120, 5};
    uint8_t te[2] = {120, 121};
    tensor_t *p = ceBfp2d(2, 3, 8, 2, 3, codes, te, HALF_AWAY);
    tensor_t *label =
        ceBfp2d(2, 3, 8, 1, 0, (int32_t[]){64, 0, 0, 64, 0, 64}, (uint8_t[]){121}, HALF_AWAY);
    tensor_t *grad = ceBfp2d(2, 3, 8, 1, 0, NULL, NULL, HALF_AWAY);
    crossEntropySoftmaxBackward(p, label, grad, NULL);
}

void testCeIntegerHeadRequiresExactlyOneOnePerClassRow(void) {
    const char *msg = "needs exactly one 1.0";
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieBfpTwoOnes());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieBfpNoOne());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieFloatTwoOnes());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, ceDieFloatNoOne());
    ASSERT_EXITS_WITH_OUTPUT(1, "label row 1", ceDieSecondRowTwoOnes());
}

/* Re-review E #4: a class row is Softmax's row (#152): axis 0 is the row
 * axis, a row spans every trailing dim (rank 1 = one row), so the label of a
 * [1, 2, 2] prediction is ONE row of 4 elements -- not two rows of 2. The
 * kernel is called directly with a BFP label as the second operand and must
 * give the FLOAT32-label raw where the row is one-hot. */
static void ceRowGeometryCase(size_t ndim, const size_t *dims, const int32_t *yCodes,
                              const float *yFloat) {
    int32_t codes[4] = {64, 32, 16, 16};
    uint8_t e7[1] = {120};
    bfpScratchFixture_t p;
    bfpScratchInit(&p, ndim, dims, codes, e7, 1, 0, 8, 8);
    bfpScratchFixture_t y;
    bfpScratchInit(&y, ndim, dims, yCodes, (uint8_t[]){121}, 1, 0, 8, 8);
    bfpRawFixture_t rawB, rawF;
    bfpRawInit(&rawB, ndim, dims, 1, 0);
    bfpRawInit(&rawF, ndim, dims, 1, 0);
    tensor_t *opsB[2] = {&p.tensor, &y.tensor};
    crossEntropySoftmaxBackwardKernelBfp(opsB, 2, &rawB.tensor, NULL, NULL, NULL);
    tensor_t *label = buildFloatTensorShaped(dims, ndim, (float *)yFloat);
    tensor_t *opsF[1] = {&p.tensor};
    crossEntropySoftmaxBackwardKernelBfp(opsF, 1, &rawF.tensor, NULL, label, NULL);
    TEST_ASSERT_EQUAL_INT32_ARRAY(rawF.mant, rawB.mant, 4);
    TEST_ASSERT_EQUAL_INT16(rawF.exps[0], rawB.exps[0]);
    freeTensor(label);
}

static const size_t CE_DIMS_R3[3] = {1, 2, 2};
static const size_t CE_DIMS_R1[1] = {4};

void testCeOneHotRowsFollowSoftmaxRowGeometry(void) {
    ceRowGeometryCase(3, CE_DIMS_R3, (int32_t[]){64, 0, 0, 0}, (float[]){1.f, 0.f, 0.f, 0.f});
    ceRowGeometryCase(1, CE_DIMS_R1, (int32_t[]){0, 0, 64, 0}, (float[]){0.f, 0.f, 1.f, 0.f});
}

static void ceDieRank3TwoHot(void) {
    ceRowGeometryCase(3, CE_DIMS_R3, (int32_t[]){64, 0, 64, 0}, (float[]){1.f, 0.f, 1.f, 0.f});
}
static void ceDieRank1TwoHot(void) {
    ceRowGeometryCase(1, CE_DIMS_R1, (int32_t[]){64, 64, 0, 0}, (float[]){1.f, 1.f, 0.f, 0.f});
}

void testCeOneHotRowsRejectATwoHotSoftmaxRow(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "needs exactly one 1.0", ceDieRank3TwoHot());
    ASSERT_EXITS_WITH_OUTPUT(1, "needs exactly one 1.0", ceDieRank1TwoHot());
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testCeIntegerHeadReadsABfpOneHotLabelExactly);
    RUN_TEST(testCeIntegerHeadGivesTheSameGradForABfpAndAFloatOneHot);
    RUN_TEST(testCeIntegerHeadRejectsABfpLabelThatIsNotOneHot);
    RUN_TEST(testCeIntegerHeadRequiresExactlyOneOnePerClassRow);
    RUN_TEST(testCeOneHotRowsFollowSoftmaxRowGeometry);
    RUN_TEST(testCeOneHotRowsRejectATwoHotSoftmaxRow);
    RUN_TEST(unitTestCrossEntropySoftmaxBackward);
    RUN_TEST(testCrossEntropyBackward_WritesRawPerElementGrad);
    RUN_TEST(testCrossEntropyForward_SumReturnsRawSum);
    RUN_TEST(testCrossEntropyForward_MeanDividesByMicrobatch);
    RUN_TEST(testCrossEntropyForward_Mean1DTensorReturnsSum);
    RUN_TEST(testCrossEntropyForward_DispatcherFloat32MatchesFloatImpl);
    RUN_TEST(testCrossEntropyForwardSymInt32MatchesGold);
    RUN_TEST(testCrossEntropyForwardSymMicrobatchMeanDivides);
    RUN_TEST(testCrossEntropySoftmaxBackwardSymWritesRequantizedGrad);
    RUN_TEST(testCrossEntropyForwardBfpEqualsFloat32OnExactGrid);
    RUN_TEST(testCrossEntropySoftmaxBackwardBfpRequantizesIntoFreshGrid);
    RUN_TEST(testCrossEntropySoftmaxBackwardFloatOutputIntoABfpGradWireIsTheLossBoundary);
    RUN_TEST(testCrossEntropyFloatOutputWithABfpLabelDequantizesTheLabel);
    RUN_TEST(testCrossEntropyForwardBfpRejectsDistributionCountMismatch);
    RUN_TEST(testCrossEntropySoftmaxBackwardBfpRejectsDistributionCountMismatch);
    RUN_TEST(testCrossEntropySoftmaxBackwardBfpRejectsLossWireCountMismatch);
    RUN_TEST(testCrossEntropyForwardFloat32RejectsDistributionCountMismatch);
    RUN_TEST(testCrossEntropySoftmaxBackwardFloat32RejectsLossWireCountMismatch);
    RUN_TEST(testCrossEntropyForwardRejectsRank1Output);
    RUN_TEST(testCrossEntropyForwardRejectsLeadingAxisMismatch);
    RUN_TEST(testCrossEntropySoftmaxBackwardRejectsDistributionRankMismatch);
    RUN_TEST(testCrossEntropyForward_NonFiniteSoftmaxReturnsNanInsteadOfAborting);
    RUN_TEST(testCrossEntropyForward_InfiniteSoftmaxReturnsNanInsteadOfAborting);
    RUN_TEST(testCrossEntropyBackwardRequantizesTheGradWithItsLossStream);
    RUN_TEST(testCeIntegerHeadRawMatchesTheInt64Reference);
    RUN_TEST(testCeIntegerHeadSubtractsOneExactly);
    RUN_TEST(testCeIntegerHeadRejectsALabelFinerThanTwoToMinus30);
    RUN_TEST(testCeIntegerHeadPlacesALabelOnTheTwoToMinus30GridExactly);
    RUN_TEST(testCeIntegerHeadReblocksOntoTheLossGradWireByItsMode);
    RUN_TEST(testCeIntegerHeadSrReblockIsUnbiased);
    RUN_TEST(testCeIntegerHeadRejectsAHalfSpecifiedBoundary);
    return UNITY_END();
}
