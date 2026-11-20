#define SOURCE_FILE "UNIT_TEST_LAYER"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "BatchNorm1d.h"
#include "Conv1d.h"
#include "DeathTest.h"
#include "Conv1dTransposed.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "Tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Stack-built fixtures (the UnitTestSgd.c idiom): both helpers read only the
 * type, `frozen`, BatchNorm1d's `affine` and the weight/bias (gamma/beta)
 * pointers, never the parameters themselves, so each parameter_t's address is
 * the identity under test. Param-free layers keep config == NULL, as
 * flattenLayerInit does. */

static void assertParams(layer_t *layer, parameter_t *expectedWeight, parameter_t *expectedBias,
                         const char *row) {
    parameter_t *weight = NULL;
    parameter_t *bias = NULL;
    TEST_ASSERT_TRUE_MESSAGE(layerParameters(layer, &weight, &bias), row);
    TEST_ASSERT_EQUAL_PTR_MESSAGE(expectedWeight, weight, row);
    TEST_ASSERT_EQUAL_PTR_MESSAGE(expectedBias, bias, row);
}

void testLayerParametersReturnsEachParamLayersOwnPair(void) {
    parameter_t p[13];
    linearConfig_t linear = {.weights = &p[0], .bias = &p[1]};
    conv1dConfig_t conv = {.weights = &p[2], .bias = &p[3]};
    conv1dConfig_t convNoBias = {.weights = &p[4], .bias = NULL};
    conv1dTransposedConfig_t convT = {.weights = &p[5], .bias = &p[6]};
    layerNormConfig_t layerNorm = {.gamma = &p[7], .beta = &p[8]};
    groupNormConfig_t groupNorm = {.gamma = &p[9], .beta = &p[10]};
    batchNorm1dConfig_t batchNorm = {.gamma = &p[11], .beta = &p[12], .affine = true};

    assertParams(&(layer_t){LINEAR, &(layerConfig_t){.linear = &linear}}, &p[0], &p[1], "LINEAR");
    assertParams(&(layer_t){CONV1D, &(layerConfig_t){.conv1d = &conv}}, &p[2], &p[3], "CONV1D");
    assertParams(&(layer_t){CONV1D, &(layerConfig_t){.conv1d = &convNoBias}}, &p[4], NULL,
                 "CONV1D without bias");
    assertParams(&(layer_t){CONV1D_TRANSPOSED, &(layerConfig_t){.conv1dTransposed = &convT}}, &p[5],
                 &p[6], "CONV1D_TRANSPOSED");
    assertParams(&(layer_t){LAYERNORM, &(layerConfig_t){.layerNorm = &layerNorm}}, &p[7], &p[8],
                 "LAYERNORM");
    assertParams(&(layer_t){GROUPNORM, &(layerConfig_t){.groupNorm = &groupNorm}}, &p[9], &p[10],
                 "GROUPNORM");
    assertParams(&(layer_t){BATCHNORM1D, &(layerConfig_t){.batchNorm1d = &batchNorm}}, &p[11],
                 &p[12], "affine BATCHNORM1D");
}

void testLayerParametersFalseForParamFreeLayers(void) {
    const layerType_t paramFree[] = {RELU,    MAXPOOL1D,    AVGPOOL1D,          SOFTMAX,
                                     FLATTEN, QUANTIZATION, ADAPTIVE_AVGPOOL1D, DROPOUT};
    parameter_t sentinel;
    for (size_t i = 0; i < sizeof(paramFree) / sizeof(paramFree[0]); i++) {
        layer_t layer = {.type = paramFree[i], .config = NULL};
        parameter_t *weight = &sentinel;
        parameter_t *bias = &sentinel;
        char row[32];
        snprintf(row, sizeof(row), "param-free type %d", (int)paramFree[i]);
        TEST_ASSERT_FALSE_MESSAGE(layerParameters(&layer, &weight, &bias), row);
        TEST_ASSERT_EQUAL_PTR_MESSAGE(&sentinel, weight, row);
        TEST_ASSERT_EQUAL_PTR_MESSAGE(&sentinel, bias, row);
    }

    /* A non-affine BatchNorm1d is config-bearing (unlike the array above) but
     * still param-free: no gamma/beta, so never trainable. */
    batchNorm1dConfig_t nonAffine = {.gamma = NULL, .beta = NULL, .affine = false};
    layer_t nonAffineBn = {.type = BATCHNORM1D,
                           .config = &(layerConfig_t){.batchNorm1d = &nonAffine}};
    parameter_t *bnWeight = &sentinel;
    parameter_t *bnBias = &sentinel;
    TEST_ASSERT_FALSE_MESSAGE(layerParameters(&nonAffineBn, &bnWeight, &bnBias),
                              "non-affine BATCHNORM1D");
    TEST_ASSERT_EQUAL_PTR_MESSAGE(&sentinel, bnWeight, "non-affine BATCHNORM1D");
    TEST_ASSERT_EQUAL_PTR_MESSAGE(&sentinel, bnBias, "non-affine BATCHNORM1D");
}

/* A frozen layer still owns its parameters: traceModelParams (and through it
 * the HAR finetune example's resident-param byte count) reports them. */
void testLayerParametersIgnoresFrozen(void) {
    parameter_t p[4];
    linearConfig_t linear = {.weights = &p[0], .bias = &p[1], .frozen = true};
    layerNormConfig_t layerNorm = {.gamma = &p[2], .beta = &p[3], .frozen = true};

    assertParams(&(layer_t){LINEAR, &(layerConfig_t){.linear = &linear}}, &p[0], &p[1],
                 "frozen LINEAR");
    assertParams(&(layer_t){LAYERNORM, &(layerConfig_t){.layerNorm = &layerNorm}}, &p[2], &p[3],
                 "frozen LAYERNORM");
}

void testAddReachingAChainDriverExits(void) {
    layer_t add = {.type = ADD, .config = NULL};
    tensor_t x = {0}, y = {0};
    ASSERT_EXITS_WITH_OUTPUT(1, "merge node reached a chain driver",
                             layerFunctions[ADD].forward(&add, &x, &y, NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, "merge node reached a chain driver",
                             layerFunctions[ADD].backward(&add, &x, &y, &y, NULL));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerParametersReturnsEachParamLayersOwnPair);
    RUN_TEST(testLayerParametersFalseForParamFreeLayers);
    RUN_TEST(testLayerParametersIgnoresFrozen);
    RUN_TEST(testAddReachingAChainDriverExits);
    return UNITY_END();
}
