#define SOURCE_FILE "SOFTMAX_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "LayerQuant.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

static layer_t *buildSoftmaxLayer(layerQuant_t *lq, bool owning, const char *who) {
    if (lq == NULL) {
        PRINT_ERROR("%s: lq pointer is NULL", who);
        exit(1);
    }
    if (lq->outputQ == NULL || lq->propLossQ == NULL) {
        PRINT_ERROR("%s: layerQuant.outputQ and layerQuant.propLossQ must be set", who);
        exit(1);
    }
    softmaxConfig_t probe = {.forwardMath = lq->forwardMath,
                             .propLossMath = lq->propLossMath,
                             .outputQ = lq->outputQ,
                             .propLossQ = lq->propLossQ};
    softmaxValidateConfig(&probe, who);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = SOFTMAX;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    softmaxConfig_t *cfg = reserveMemory(sizeof(softmaxConfig_t));
    layerCfg->softmax = cfg;
    layer->config = layerCfg;

    *cfg = probe;
    if (owning) {
        cfg->outputQ = deepCopyQuantization(lq->outputQ);
        cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    }
    cfg->ownsQuantizations = owning;
    cfg->bfpExpShiftRounding = BFP_SHIFT_TRUNC;
    return layer;
}

layer_t *softmaxLayerInit(layerQuant_t *lq) {
    return buildSoftmaxLayer(lq, false, "softmaxLayerInit");
}

layer_t *softmaxLayerInitOwning(layerQuant_t *lq) {
    return buildSoftmaxLayer(lq, true, "softmaxLayerInitOwning");
}

void softmaxSetBfpExpShiftRounding(layer_t *softmaxLayer, bfpShiftRounding_t mode) {
    if (mode != BFP_SHIFT_TRUNC && mode != BFP_SHIFT_HALF_AWAY && mode != BFP_SHIFT_SR) {
        PRINT_ERROR("softmaxSetBfpExpShiftRounding: unknown mode %d", (int)mode);
        exit(1);
    }
    softmaxLayer->config->softmax->bfpExpShiftRounding = mode;
}

void freeSoftmaxLayer(layer_t *softmaxLayer) {
    if (softmaxLayer == NULL) {
        return;
    }
    softmaxConfig_t *cfg = softmaxLayer->config->softmax;
    if (cfg->ownsQuantizations) {
        freeQuantization(cfg->outputQ);
        freeQuantization(cfg->propLossQ);
    }
    freeReservedMemory(cfg);
    freeReservedMemory(softmaxLayer->config);
    freeReservedMemory(softmaxLayer);
}
