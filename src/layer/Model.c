#define SOURCE_FILE "MODEL"

#include <stdlib.h>

#include "Common.h"
#include "Model.h"
#include "OpTable.h"
#include "StorageApi.h"

uint8_t modelNodeInputs(const model_t *m, size_t j, uint16_t ids[MODEL_MAX_ARITY]) {
    if (m->inputStart == NULL) {
        ids[0] = (uint16_t)j;
        return 1;
    }
    uint16_t begin = m->inputStart[j];
    uint8_t count = (uint8_t)(m->inputStart[j + 1] - begin);
    for (uint8_t k = 0; k < count; k++) {
        ids[k] = m->inputIds[begin + k];
    }
    return count;
}

bool modelIsChain(const model_t *m) {
    if (m->inputStart == NULL) {
        return true;
    }
    for (size_t j = 0; j <= m->numLayers; j++) {
        if (m->inputStart[j] != j) {
            return false;
        }
    }
    for (size_t j = 0; j < m->numLayers; j++) {
        if (m->inputIds[j] != j) {
            return false;
        }
    }
    return true;
}

void modelShapePass(const model_t *m, const shape_t *input, shape_t *shapes) {
    shapes[0].numberOfDimensions = input->numberOfDimensions;
    for (size_t d = 0; d < input->numberOfDimensions; d++) {
        shapes[0].dimensions[d] = input->dimensions[d];
        shapes[0].orderOfDimensions[d] = input->orderOfDimensions[d];
    }
    for (size_t j = 0; j < m->numLayers; j++) {
        uint16_t ids[MODEL_MAX_ARITY];
        uint8_t k = modelNodeInputs(m, j, ids);
        const shape_t *in[MODEL_MAX_ARITY];
        for (uint8_t i = 0; i < k; i++) {
            in[i] = &shapes[ids[i]];
        }
        layer_t *layer = m->layers[j];
        opFunctions[layer->type].shape(layer, in, k, &shapes[j + 1]);
    }
}

void modelGradFlags(const model_t *m, lossFuncType_t lt, uint8_t *needsGrad,
                    uint8_t *runsBackward) {
    size_t n = m->numLayers;
    needsGrad[0] = 0;
    for (size_t j = 0; j < n; j++) {
        parameter_t *w = NULL;
        parameter_t *b = NULL;
        layer_t *layer = m->layers[j];
        bool needs = layerParameters(layer, &w, &b) && !layerIsFrozen(layer);
        uint16_t ids[MODEL_MAX_ARITY];
        uint8_t k = modelNodeInputs(m, j, ids);
        for (uint8_t i = 0; i < k; i++) {
            needs = needs || needsGrad[ids[i]];
        }
        needsGrad[j + 1] = needs;
        runsBackward[j] = needs;
    }
    if (lt == CROSS_ENTROPY) {
        runsBackward[n - 1] = 0;
    }
}

typedef struct nodeLayer {
    uintptr_t layer;
    size_t node;
} nodeLayer_t;

static int compareNodeLayers(const void *a, const void *b) {
    const nodeLayer_t *x = a;
    const nodeLayer_t *y = b;
    if (x->layer != y->layer) {
        return x->layer < y->layer ? -1 : 1;
    }
    return x->node < y->node ? -1 : (x->node > y->node);
}

/* For non-chain models only: a layer's state (parameter grads, saved masks or indices) lives on
 * the object, so two nodes sharing one would share it. Chains keep accepting a shared stateless
 * layer, as they always have. Sorting finds a repeat in O(n log n). */
static void requireOneLayerPerNode(const model_t *m, const char *caller) {
    size_t n = m->numLayers;
    nodeLayer_t *pairs = reserveMemory(n * sizeof(nodeLayer_t));
    if (pairs == NULL) {
        PRINT_ERROR("%s: wiring: no memory for the %zu-byte layer map", caller,
                    n * sizeof(nodeLayer_t));
        exit(1);
    }
    for (size_t j = 0; j < n; j++) {
        pairs[j] = (nodeLayer_t){(uintptr_t)m->layers[j], j};
    }
    qsort(pairs, n, sizeof(nodeLayer_t), compareNodeLayers);
    for (size_t i = 0; i + 1 < n; i++) {
        if (pairs[i].layer == pairs[i + 1].layer) {
            size_t later = pairs[i + 1].node;
            size_t earlier = pairs[i].node;
            freeReservedMemory(pairs);
            PRINT_ERROR("%s: wiring: node %zu reuses the layer object of node %zu (a non-chain "
                        "model needs one layer_t per node)",
                        caller, later, earlier);
            exit(1);
        }
    }
    freeReservedMemory(pairs);
}

void modelRequireValid(const model_t *m, const char *caller) {
    size_t n = m->numLayers;
    if (n == 0) {
        PRINT_ERROR("%s: modelSize == 0: nothing to schedule", caller);
        exit(1);
    }
    /* RematPlan's REMAT_NONE sentinel, which this library cannot include. */
    if (n >= 0xFFFF) {
        PRINT_ERROR("%s: modelSize %zu reaches 0xFFFF: buffer ids are uint16_t", caller, n);
        exit(1);
    }
    if (m->inputStart == NULL) {
        for (size_t j = 0; j < n; j++) {
            layerType_t type = m->layers[j]->type;
            if (opFunctions[type].arity != 1) {
                PRINT_ERROR("%s: wiring: node %zu has layer type %d of arity %u; a chain encoding "
                            "takes arity 1 only",
                            caller, j, (int)type, (unsigned)opFunctions[type].arity);
                exit(1);
            }
        }
        return;
    }

    const uint16_t *start = m->inputStart;
    if (start[0] != 0) {
        PRINT_ERROR("%s: wiring: inputStart[0] is %u, not 0", caller, (unsigned)start[0]);
        exit(1);
    }
    for (size_t j = 0; j < n; j++) {
        if (start[j + 1] < start[j]) {
            PRINT_ERROR("%s: wiring: inputStart[%zu] = %u is below inputStart[%zu] = %u", caller,
                        j + 1, (unsigned)start[j + 1], j, (unsigned)start[j]);
            exit(1);
        }
        unsigned count = (unsigned)(start[j + 1] - start[j]);
        layerType_t type = m->layers[j]->type;
        if (count != opFunctions[type].arity) {
            PRINT_ERROR("%s: wiring: node %zu has %u inputs, layer type %d takes %u", caller, j,
                        count, (int)type, (unsigned)opFunctions[type].arity);
            exit(1);
        }
    }
    for (size_t j = 0; j < n; j++) {
        for (uint16_t e = start[j]; e < start[j + 1]; e++) {
            if (m->inputIds[e] > j) {
                PRINT_ERROR("%s: wiring: node %zu reads buffer %u, which is not produced before it",
                            caller, j, (unsigned)m->inputIds[e]);
                exit(1);
            }
        }
    }

    if (modelIsChain(m)) {
        return;
    }
    uint8_t *consumed = reserveMemory(n);
    if (consumed == NULL) {
        PRINT_ERROR("%s: wiring: no memory for the %zu-byte consumer map", caller, n);
        exit(1);
    }
    for (uint16_t e = 0; e < start[n]; e++) {
        consumed[m->inputIds[e]] = 1;
    }
    for (size_t b = 0; b < n; b++) {
        if (!consumed[b]) {
            freeReservedMemory(consumed);
            PRINT_ERROR("%s: wiring: buffer %zu has no consumer", caller, b);
            exit(1);
        }
    }
    freeReservedMemory(consumed);

    requireOneLayerPerNode(m, caller);
}
