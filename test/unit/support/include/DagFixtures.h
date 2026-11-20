#ifndef ODT_TEST_DAG_FIXTURES_H
#define ODT_TEST_DAG_FIXTURES_H

/*
 * The shared DAG models (test-only), small enough that every forward value
 * and gradient is derived by hand. Buffer 0 is the input, buffer j + 1 the
 * output of node j; every layer is FLOAT32 and every Linear is 2 -> 2 with a
 * hand-set diagonal weight and a zero bias.
 *
 *   F3  n0 = Linear I (b0); n1 = Linear D (b1); n2 = ADD(b2, b1); n3 = ADD(b3, b1)
 *   F3r as F3, but n3 = ADD(b3, b2)
 *   Y   n0 = Linear I (b0); n1 = ADD(b1, b1)
 *   K   n0 = Linear D (b0); n1 = ADD(b1, b0)
 *   S   n0 = Linear I (b0); n1 = Linear D (b0), frozen; n2 = ADD(b1, b2)
 *   P2  n0 = Linear I (b0); n1 = ReLU(b1); n2 = Linear D (b1); n3 = ADD(b2, b3);
 *       n4 = ADD(b4, b1)
 *
 * with I = identity and D = diag(2, 3). The model of a dagFixture_t points
 * into the fixture's own arrays: build it in place and never copy it.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "AddApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "Model.h"
#include "ModelGraphApi.h"
#include "Quantization.h"
#include "RNG.h"
#include "ReluApi.h"
#include "Tensor.h"

typedef struct dagFixture {
    layer_t *layers[5];
    uint16_t inputStart[6];
    uint16_t inputIds[10];
    graphBuilder_t g;
    model_t m;
} dagFixture_t;

/* A [1, 2] FLOAT32 tensor whose shape, quantization and data all live in the
 * struct, so it is valid exactly as long as the struct. */
typedef struct dagTensor {
    tensor_t t;
    shape_t shape;
    size_t dims[2];
    size_t order[2];
    float data[2];
    quantization_t q;
} dagTensor_t;

static inline tensor_t *dagInput(dagTensor_t *s, float a, float b) {
    s->dims[0] = 1;
    s->dims[1] = 2;
    s->order[0] = 0;
    s->order[1] = 1;
    s->data[0] = a;
    s->data[1] = b;
    s->q = (quantization_t){.type = FLOAT32, .qConfig = NULL};
    s->shape =
        (shape_t){.numberOfDimensions = 2, .dimensions = s->dims, .orderOfDimensions = s->order};
    s->t = (tensor_t){
        .data = (uint8_t *)s->data, .shape = &s->shape, .quantization = &s->q, .sparsity = NULL};
    return &s->t;
}

/* The factories borrow their wire templates, so the template outlives every fixture. */
static inline quantization_t *dagFloatQ(void) {
    static quantization_t q = {.type = FLOAT32, .qConfig = NULL};
    return &q;
}

/* The drawn weights and bias are overwritten below, so position only has to give a valid init
 * key. A diagonal weight is its own
 * transpose, so the [out, in] layout cannot flip a hand-derived value. */
static inline layer_t *dagLinear(size_t position, float d0, float d1, bool frozen) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, dagFloatQ());
    const rngKey_t key = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, (uint32_t)position);
    layer_t *layer =
        linearLayerInit(&(linearInit_t){.inFeatures = 2,
                                        .outFeatures = 2,
                                        .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT,
                                        .weightInit.key = &key},
                        &lq);
    float *w = (float *)layer->config->linear->weights->param->data;
    w[0] = d0;
    w[1] = 0.0f;
    w[2] = 0.0f;
    w[3] = d1;
    float *b = (float *)layer->config->linear->bias->param->data;
    b[0] = 0.0f;
    b[1] = 0.0f;
    return layer;
}

static inline layer_t *dagRelu(void) {
    return reluLayerInit(&(layerQuant_t){.outputQ = dagFloatQ(), .propLossQ = dagFloatQ()});
}

static inline void dagBegin(dagFixture_t *f) {
    graphBegin(&f->g, f->layers, f->inputStart, f->inputIds, 5, 10);
}

static inline void dagBuildF3(dagFixture_t *f, bool freezeN0) {
    dagBegin(f);
    uint16_t b1 = graphNode(&f->g, dagLinear(0, 1.0f, 1.0f, freezeN0), GRAPH_INPUT);
    uint16_t b2 = graphNode(&f->g, dagLinear(1, 2.0f, 3.0f, false), b1);
    uint16_t b3 = graphMerge(&f->g, addLayerInit(NULL), b2, b1);
    (void)graphMerge(&f->g, addLayerInit(NULL), b3, b1);
    f->m = graphFinish(&f->g);
}

static inline void dagBuildF3r(dagFixture_t *f) {
    dagBegin(f);
    uint16_t b1 = graphNode(&f->g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphNode(&f->g, dagLinear(1, 2.0f, 3.0f, false), b1);
    uint16_t b3 = graphMerge(&f->g, addLayerInit(NULL), b2, b1);
    (void)graphMerge(&f->g, addLayerInit(NULL), b3, b2);
    f->m = graphFinish(&f->g);
}

static inline void dagBuildY(dagFixture_t *f) {
    dagBegin(f);
    uint16_t b1 = graphNode(&f->g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    (void)graphMerge(&f->g, addLayerInit(NULL), b1, b1);
    f->m = graphFinish(&f->g);
}

static inline void dagBuildK(dagFixture_t *f) {
    dagBegin(f);
    uint16_t b1 = graphNode(&f->g, dagLinear(0, 2.0f, 3.0f, false), GRAPH_INPUT);
    (void)graphMerge(&f->g, addLayerInit(NULL), b1, GRAPH_INPUT);
    f->m = graphFinish(&f->g);
}

static inline void dagBuildS(dagFixture_t *f) {
    dagBegin(f);
    uint16_t b1 = graphNode(&f->g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphNode(&f->g, dagLinear(1, 2.0f, 3.0f, true), GRAPH_INPUT);
    (void)graphMerge(&f->g, addLayerInit(NULL), b1, b2);
    f->m = graphFinish(&f->g);
}

static inline void dagBuildP2(dagFixture_t *f) {
    dagBegin(f);
    uint16_t b1 = graphNode(&f->g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphNode(&f->g, dagRelu(), b1);
    uint16_t b3 = graphNode(&f->g, dagLinear(2, 2.0f, 3.0f, false), b1);
    uint16_t b4 = graphMerge(&f->g, addLayerInit(NULL), b2, b3);
    (void)graphMerge(&f->g, addLayerInit(NULL), b4, b1);
    f->m = graphFinish(&f->g);
}

/* Frees every layer of a built fixture; the fixture storage itself is the caller's. */
static inline void dagFree(dagFixture_t *f) {
    for (size_t j = 0; j < f->m.numLayers; j++) {
        layer_t *layer = f->layers[j];
        switch (layer->type) {
        case LINEAR:
            freeLinearLayer(layer);
            break;
        case RELU:
            freeReluLayer(layer);
            break;
        default:
            freeAddLayer(layer);
            break;
        }
    }
}

#endif /* ODT_TEST_DAG_FIXTURES_H */
