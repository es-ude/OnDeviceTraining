#ifndef ODT_TEST_BORROWED_LAYER_H
#define ODT_TEST_BORROWED_LAYER_H

/*
 * Borrowed-parameter layer builders (test-only).
 *
 * The public factories (linearLayerInit, conv1dLayerInit, ...) always allocate
 * and randomly initialize their own FLOAT32 parameters (requireFloat32 in
 * LayerCommon.c). That gate is by design (#270): user code reaches
 * SYM_INT32-native params through FLOAT32 init + requantizeTensorInPlace()
 * (see examples/mixed_width_mlp) or through the LayerNorm/GroupNorm
 * constant-fill factories — not through a factory knob. Tests, however, need
 * what no public path should offer: exact hand-seeded parameter values,
 * SYM_INT32-native (or sub-byte SYM grad) storage built by the fixture, and
 * shapes decoupled from any real inFeatures/outFeatures pairing. These
 * builders wire the layer config directly, borrowing fixture-owned
 * parameter_t's — replicating the deleted
 * *LayerInitLegacy(weights, bias, q, q, q, q) uniform-Q shape: the single q
 * (caller-owned, ownsQuantizations=false) covers forward + all backward math
 * and the produced outputQ/propLossQ wires.
 */

#include "ArithmeticType.h"
#include "Conv1d.h"
#include "Layer.h"
#include "Linear.h"
#include "StorageApi.h"

static inline layer_t *buildBorrowedLinearLayer(parameter_t *weights, parameter_t *bias,
                                                quantization_t *q) {
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    cfg->weights = weights;
    cfg->bias = bias;
    cfg->forwardMath = arithmeticFromQuantization(q);
    cfg->weightGradMath = arithmeticFromQuantization(q);
    cfg->biasGradMath = arithmeticFromQuantization(q);
    cfg->propLossMath = arithmeticFromQuantization(q);
    cfg->outputQ = q;
    cfg->propLossQ = q;
    /* PR3 spec D1: today's per-callsite hardcodes (linearBackward); hand-wired
     * here since this helper builds the config directly instead of going
     * through linearInitConfig/a layerQuant_t factory. */
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    cfg->ownsQuantizations = false;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    layerCfg->linear = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LINEAR, layerCfg);
    return layer;
}

/*! Frees only the layer_t + layerConfig_t + linearConfig_t shells — NOT the
 *  weight/bias parameters (caller-owned, either freed explicitly or via
 *  freeOptim's cascade). Needed after freeOptim, which already frees every
 *  parameter it registered (freeLinearLayer would double-free them). */
static inline void freeLinearLayerShellOnly(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/*! Conv1d analogue at an explicit `groups`; goes straight through
 *  initConv1dConfigWithWeightsAndBias. The borrowed kernel_t must outlive the
 *  layer. Weight shape must be [Cout, Cin/groups, K] — the factory's own
 *  channel-consistency guards still apply. */
static inline layer_t *buildBorrowedConv1dLayerGrouped(parameter_t *weights, parameter_t *bias,
                                                       kernel_t *kernel, size_t groups,
                                                       quantization_t *q) {
    conv1dConfig_t *cfg = reserveMemory(sizeof(conv1dConfig_t));
    initConv1dConfigWithWeightsAndBias(cfg, kernel, weights, bias, groups, q, q, q, q);
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    layerCfg->conv1d = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, CONV1D, layerCfg);
    return layer;
}

/*! groups==1 shorthand (the overwhelmingly common fixture shape). */
static inline layer_t *buildBorrowedConv1dLayer(parameter_t *weights, parameter_t *bias,
                                                kernel_t *kernel, quantization_t *q) {
    return buildBorrowedConv1dLayerGrouped(weights, bias, kernel, 1u, q);
}

#endif /* ODT_TEST_BORROWED_LAYER_H */
