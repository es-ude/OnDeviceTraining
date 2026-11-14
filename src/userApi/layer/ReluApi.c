#define SOURCE_FILE "RELU_API"

#include <stdbool.h>
#include <stdlib.h> /* exit */

#include "ArithmeticType.h"
#include "Common.h" /* PRINT_ERROR */
#include "LayerQuant.h"
#include "Relu.h"
#include "ReluApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

static layer_t *buildReluLayer(layerQuant_t *lq, bool owning, const char *who) {
    if (lq == NULL) {
        PRINT_ERROR("%s: lq pointer is NULL", who);
        exit(1);
    }
    if (lq->outputQ == NULL || lq->propLossQ == NULL) {
        PRINT_ERROR("%s: layerQuant.outputQ and layerQuant.propLossQ must be set", who);
        exit(1);
    }
    reluConfig_t probe = {.forwardMath = lq->forwardMath,
                          .propLossMath = lq->propLossMath,
                          .outputQ = lq->outputQ,
                          .propLossQ = lq->propLossQ};
    reluValidateConfig(&probe, who);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = RELU;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    reluConfig_t *cfg = reserveMemory(sizeof(reluConfig_t));
    layerCfg->relu = cfg;
    layer->config = layerCfg;

    *cfg = probe;
    if (owning) {
        cfg->outputQ = deepCopyQuantization(lq->outputQ);
        cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    }
    cfg->ownsQuantizations = owning;
    return layer;
}

layer_t *reluLayerInit(layerQuant_t *lq) {
    return buildReluLayer(lq, false, "reluLayerInit");
}

layer_t *reluLayerInitOwning(layerQuant_t *lq) {
    return buildReluLayer(lq, true, "reluLayerInitOwning");
}

void freeReluLayer(layer_t *reluLayer) {
    if (reluLayer == NULL) {
        return;
    }
    reluConfig_t *cfg = reluLayer->config->relu;
    if (cfg->ownsQuantizations) {
        freeQuantization(cfg->outputQ);
        freeQuantization(cfg->propLossQ);
    }
    freeReservedMemory(cfg);
    freeReservedMemory(reluLayer->config);
    freeReservedMemory(reluLayer);
}
