#include "CrossEntropy.h"
#include "QuantizationApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "TensorApi.h"
#include "unity.h"
#include <math.h>

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
    layer_t *softmaxLayer = softmaxLayerInitLegacy(floatQ, floatQ);
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
    freeSoftmaxLayerLegacy(softmaxLayer);
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
    layer_t *softmaxLayer = softmaxLayerInitLegacy(floatQ, floatQ);
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

    freeSoftmaxLayerLegacy(softmaxLayer);
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
    return UNITY_END();
}
