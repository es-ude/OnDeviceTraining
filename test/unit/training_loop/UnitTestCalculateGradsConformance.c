#define SOURCE_FILE "UNIT_TEST_CALCULATE_GRADS_CONFORMANCE"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LegacyCalculateGrads.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingLoopApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---- the NULL scheduler against the Legacy oracle (spec §12.1, §12.2 item 4) ---- */

static quantization_t g_float = {.type = FLOAT32, .qConfig = NULL};

/* A tensor with real, deterministic bytes: the driver's checker requires the
 * input to be resident, and remat P1 needs values that exercise every layer. */
static tensor_t *makeFloatTensor(const size_t *dims, size_t rank, float seed) {
    size_t *d = reserveMemory(rank * sizeof(size_t));
    size_t *order = reserveMemory(rank * sizeof(size_t));
    size_t elements = 1;
    for (size_t k = 0; k < rank; k++) {
        d[k] = dims[k];
        elements *= dims[k];
    }
    setOrderOfDimsForNewTensor(rank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, d, rank, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    float *v = (float *)t->data;
    for (size_t i = 0; i < elements; i++) {
        v[i] = seed * (float)((int)((i * 37u + 11u) % 17u) - 8) / 8.0f;
    }
    return t;
}

#define ZOO_MAX_LAYERS 12
typedef struct zooFixture {
    layer_t *model[ZOO_MAX_LAYERS];
    size_t n;
    lossConfig_t loss;
    tensor_t *x;
    tensor_t *y;
} zooFixture_t;

static void freeZoo(zooFixture_t *f) {
    for (size_t i = 0; i < f->n; i++) {
        switch (f->model[i]->type) {
        case LINEAR:
            freeLinearLayer(f->model[i]);
            break;
        case RELU:
            freeReluLayer(f->model[i]);
            break;
        default:
            TEST_FAIL_MESSAGE("freeZoo: extend the switch for this layer type");
        }
    }
    freeTensor(f->x);
    freeTensor(f->y);
}

/* Linear 4 -> 3 -> ReLU -> Linear 3 -> 2 under MSE: every allocator arm the
 * plain FLOAT32 path has (forward wires, the seed, one dx wire, grads-only
 * at deepest). */
static void buildMlp(zooFixture_t *f) {
    rngSetSeed(7u);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_float);
    f->model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = 3}, &lq);
    f->model[1] = reluLayerInit(&lq);
    f->model[2] = linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2}, &lq);
    f->n = 3;
    f->loss = defaultLossConfig(MSE);
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    f->y = makeFloatTensor((size_t[]){1, 2}, 2, 0.5f);
}

/* Everything P1 compares: the loss, the output snapshot and every parameter
 * grad, each tensor with its dynamic quantization state (SYM scale, BFP
 * exponents). Static, never reserved, so the capture cannot move the
 * memory-profile counters it sits beside. */
#define CAPTURE_MAX_BYTES 65536u
typedef struct runCapture {
    float loss;
    size_t used;
    uint8_t bytes[CAPTURE_MAX_BYTES];
} runCapture_t;
static runCapture_t g_legacy;
static runCapture_t g_driver;

static void appendBytes(runCapture_t *cap, const void *src, size_t n) {
    TEST_ASSERT_TRUE_MESSAGE(cap->used + n <= CAPTURE_MAX_BYTES, "raise CAPTURE_MAX_BYTES");
    memcpy(cap->bytes + cap->used, src, n);
    cap->used += n;
}

static void captureTensor(runCapture_t *cap, tensor_t *t) {
    size_t elements = calcNumberOfElementsByTensor(t);
    int type = (int)t->quantization->type;
    appendBytes(cap, &type, sizeof type);
    appendBytes(cap, &elements, sizeof elements);
    appendBytes(cap, t->data, calcNumberOfBytesForData(t->quantization, elements));
    if (t->quantization->type == SYM_INT32) {
        const symInt32QConfig_t *qc = t->quantization->qConfig;
        appendBytes(cap, &qc->scale, sizeof qc->scale);
    } else if (t->quantization->type == BFP) {
        const bfpQConfig_t *qc = t->quantization->qConfig;
        appendBytes(cap, &qc->numGroups, sizeof qc->numGroups);
        appendBytes(cap, qc->exponents, qc->numGroups);
    }
}

static void forEachGrad(layer_t **model, size_t n, runCapture_t *cap) {
    for (size_t i = 0; i < n; i++) {
        parameter_t *w = NULL;
        parameter_t *b = NULL;
        if (!layerParameters(model[i], &w, &b)) {
            continue;
        }
        parameter_t *params[2] = {w, b};
        for (size_t p = 0; p < 2; p++) {
            tensor_t *g = (params[p] != NULL) ? getGradFromParameter(params[p]) : NULL;
            if (g == NULL) {
                continue; /* no bias, or a frozen layer (#380: no grad tensor) */
            }
            if (cap != NULL) {
                captureTensor(cap, g);
            } else {
                memset(g->data, 0,
                       calcNumberOfBytesForData(g->quantization, calcNumberOfElementsByTensor(g)));
            }
        }
    }
}

typedef enum { RUN_LEGACY, RUN_DRIVER } runner_t;

/* Grads accumulate (OUT_ACC), so both runs start from zeroed grads. */
static void captureRun(runner_t runner, zooFixture_t *f, runCapture_t *cap) {
    forEachGrad(f->model, f->n, NULL);
    cap->used = 0;
    trainingStats_t *stats =
        (runner == RUN_LEGACY)
            ? legacyCalculateGrads(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y, NULL, NULL)
            : calculateGradsSequential(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y);
    cap->loss = stats->loss;
    captureTensor(cap, stats->output);
    freeTrainingStats(stats);
    forEachGrad(f->model, f->n, cap);
}

static void assertSameP1(void) {
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&g_legacy.loss, &g_driver.loss, sizeof g_legacy.loss,
                                     "P1: loss");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.used, g_driver.used, "P1: captured byte count");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_legacy.bytes, g_driver.bytes, g_legacy.used,
                                     "P1: output snapshot and parameter grads");
}

void testNullPathMatchesLegacyOnTheMlp(void) {
    zooFixture_t f;
    buildMlp(&f);
    captureRun(RUN_LEGACY, &f, &g_legacy);
    captureRun(RUN_DRIVER, &f, &g_driver);
    TEST_ASSERT_TRUE(g_legacy.used > 0);
    assertSameP1();
    freeZoo(&f);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testNullPathMatchesLegacyOnTheMlp);
    return UNITY_END();
}
