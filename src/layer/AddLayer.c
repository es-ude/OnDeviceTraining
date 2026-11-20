#define SOURCE_FILE "ADD_LAYER"

#include <stdio.h>
#include <stdlib.h>

#include "AddLayer.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "LayerExec.h"

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    ADD_OP_FORWARD = 0,
    ADD_OP_DX = 1,
};

static void addFloatKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                           const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    (void)rs;
    const float *a = (const float *)ops[0]->data;
    const float *b = (const float *)ops[1]->data;
    float *raw = (float *)rawOut->data;
    size_t count = calcNumberOfElementsByTensor(rawOut);
    for (size_t i = 0; i < count; i++) {
        raw[i] = a[i] + b[i];
    }
}

void addForward(layer_t *layer, tensor_t *const *in, uint8_t k, tensor_t *out,
                const layerExec_t *exec) {
    (void)k;
    addConfig_t *cfg = layer->config->add;
    tensor_t *ops[2] = {in[0], in[1]};
    rngKey_t opKey;
    executeOp(
        &(opSpec_t){
            .kernel = addFloatKernel,
            .inputs = ops,
            .nInputs = 2,
            .arithmetic = cfg->forwardMath,
            .key = layerExecOpKey(exec, RNG_TAG_FORWARD, ADD_OP_FORWARD, &opKey),
            .mode = OUT_WRITE,
        },
        out);
}

/* Truncates silently: the text only feeds an error message. */
static void formatDims(const shape_t *s, char *buf, size_t cap) {
    size_t used = 0;
    buf[0] = '\0';
    for (size_t i = 0; i < s->numberOfDimensions; i++) {
        int w = snprintf(buf + used, cap - used, i == 0 ? "%zu" : ", %zu", s->dimensions[i]);
        if (w < 0 || (size_t)w >= cap - used) {
            return;
        }
        used += (size_t)w;
    }
}

static bool sameShape(const shape_t *a, const shape_t *b) {
    if (a->numberOfDimensions != b->numberOfDimensions) {
        return false;
    }
    for (size_t i = 0; i < a->numberOfDimensions; i++) {
        if (a->dimensions[i] != b->dimensions[i] ||
            a->orderOfDimensions[i] != b->orderOfDimensions[i]) {
            return false;
        }
    }
    return true;
}

void addShape(layer_t *layer, const shape_t *const *in, uint8_t k, shape_t *out) {
    (void)layer;
    (void)k;
    if (!sameShape(in[0], in[1])) {
        char a[64], b[64];
        formatDims(in[0], a, sizeof(a));
        formatDims(in[1], b, sizeof(b));
        PRINT_ERROR("addShape: input shapes differ: [%s] vs [%s]; ADD does not broadcast", a, b);
        exit(1);
    }
    out->numberOfDimensions = in[0]->numberOfDimensions;
    for (size_t i = 0; i < out->numberOfDimensions; i++) {
        out->dimensions[i] = in[0]->dimensions[i];
        out->orderOfDimensions[i] = in[0]->orderOfDimensions[i];
    }
}
