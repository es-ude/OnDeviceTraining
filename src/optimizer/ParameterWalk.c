#define SOURCE_FILE "PARAMETER_WALK"

#include <stdlib.h>

#include "BatchNorm1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "ParameterWalk.h"

static void visitParameter(size_t layerIndex, const layer_t *layer, paramRole_t role,
                           parameter_t *parameter, paramVisitFn_t visit, void *ctx) {
    const paramRef_t ref = {
        .layerIndex = layerIndex, .layer = layer, .role = role, .parameter = parameter};
    visit(&ref, ctx);
}

/* Weight, then the bias if the layer has one (BIAS_FALSE leaves it NULL, header-sanctioned). */
static void visitWeightAndBias(size_t i, const layer_t *layer, parameter_t *weights,
                               parameter_t *bias, paramVisitFn_t visit, void *ctx) {
    visitParameter(i, layer, PARAM_ROLE_WEIGHT, weights, visit, ctx);
    if (bias != NULL) {
        visitParameter(i, layer, PARAM_ROLE_BIAS, bias, visit, ctx);
    }
}

static void visitScaleAndShift(size_t i, const layer_t *layer, parameter_t *gamma,
                               parameter_t *beta, paramVisitFn_t visit, void *ctx) {
    visitParameter(i, layer, PARAM_ROLE_NORM_SCALE, gamma, visit, ctx);
    visitParameter(i, layer, PARAM_ROLE_NORM_SHIFT, beta, visit, ctx);
}

void forEachTrainableParameter(const model_t *m, paramVisitFn_t visit, void *ctx) {
    for (size_t i = 0; i < m->numLayers; i++) {
        const layer_t *layer = m->layers[i];
        if (layerIsFrozen(layer)) {
            continue;
        }
        const layerConfig_t *cfg = layer->config;
        switch (layer->type) {
        case LINEAR:
            visitWeightAndBias(i, layer, cfg->linear->weights, cfg->linear->bias, visit, ctx);
            break;
        case CONV1D:
            visitWeightAndBias(i, layer, cfg->conv1d->weights, cfg->conv1d->bias, visit, ctx);
            break;
        case CONV1D_TRANSPOSED:
            visitWeightAndBias(i, layer, cfg->conv1dTransposed->weights,
                               cfg->conv1dTransposed->bias, visit, ctx);
            break;
        case LAYERNORM:
            visitScaleAndShift(i, layer, cfg->layerNorm->gamma, cfg->layerNorm->beta, visit, ctx);
            break;
        case GROUPNORM:
            visitScaleAndShift(i, layer, cfg->groupNorm->gamma, cfg->groupNorm->beta, visit, ctx);
            break;
        case BATCHNORM1D:
            if (cfg->batchNorm1d->affine) {
                visitScaleAndShift(i, layer, cfg->batchNorm1d->gamma, cfg->batchNorm1d->beta, visit,
                                   ctx);
            }
            break;
        case RELU:
        case SOFTMAX:
        case FLATTEN:
        case MAXPOOL1D:
        case AVGPOOL1D:
        case ADAPTIVE_AVGPOOL1D:
        case DROPOUT:
        case QUANTIZATION:
        case ADD:
            break;
        default:
            PRINT_ERROR("Unknown Layer Type!");
            exit(1);
        }
    }
}
