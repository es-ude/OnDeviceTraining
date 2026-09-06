#define SOURCE_FILE "UnitTestCrossEntropy"

#include "CrossEntropy.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_cross_entropy_sym.h"
#include "unity.h"
#include <math.h>
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
    softmaxFns.forward(softmaxLayer, &logits, &softmaxOutput);

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

    crossEntropySoftmaxBackward(&softmaxOutput, &distribution, &propLoss);

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
    crossEntropySoftmaxBackward(&softmaxOutput, &distribution, &lossGrad);

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
    softmaxFns.forward(softmaxLayer, &logits, &softmaxOutput);

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
     * divide by dimensions[0]=3 and yield -log(0.6)/3, breaking parity. */
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

    float viaDispatcher = crossEntropyForward(&softmaxOutput, &distribution, REDUCTION_MEAN);
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
    convertTensor(floatSrc, &st->tensor);
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

    float lossSum = crossEntropyForward(&pSym.tensor, &label, REDUCTION_SUM);

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

    float lossMean = crossEntropyForward(&pSym.tensor, &label, REDUCTION_MEAN);

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
     * CalculateGradsSequential initGradTensor(..., NULL)) — its own int12
     * HALF_AWAY config; the backward requantizes (p-y) into it. */
    symStackTensor_t grad;
    int32_t gradData[6] = {0};
    initSymInt32QConfig(HALF_AWAY, &grad.qc);
    initSymInt32Quantization(&grad.qc, &grad.q);
    setTensorValues(&grad.tensor, (uint8_t *)gradData, &shape, &grad.q, NULL);

    crossEntropySoftmaxBackward(&pSym.tensor, &label, &grad.tensor);

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

    float bfpLoss = crossEntropyForward(bfpP, label, REDUCTION_SUM);
    float floatLoss = crossEntropyForward(floatP, label, REDUCTION_SUM);

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

    crossEntropySoftmaxBackward(bfpP, label, loss);

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

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();
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
    return UNITY_END();
}
