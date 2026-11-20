#define SOURCE_FILE "ADD_API"

#include <stdbool.h>
#include <stdlib.h> /* exit */

#include "AddApi.h"
#include "AddLayer.h"
#include "ArithmeticType.h"
#include "Common.h" /* PRINT_ERROR */
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

static void requireFloat32Math(arithmetic_t a, const char *field) {
    if (a.type != ARITH_FLOAT32) {
        PRINT_ERROR("addLayerInit: layerQuant.%s is not FLOAT32", field);
        exit(1);
    }
}

static void requireFloat32Wire(const quantization_t *q, const char *field) {
    if (q == NULL) {
        PRINT_ERROR("addLayerInit: layerQuant.%s must be set", field);
        exit(1);
    }
    if (q->type != FLOAT32) {
        PRINT_ERROR("addLayerInit: layerQuant.%s is not FLOAT32", field);
        exit(1);
    }
}

layer_t *addLayerInit(layerQuant_t *lq) {
    if (lq != NULL) {
        requireFloat32Math(lq->forwardMath, "forwardMath");
        requireFloat32Math(lq->propLossMath, "propLossMath");
        requireFloat32Wire(lq->outputQ, "outputQ");
        requireFloat32Wire(lq->propLossQ, "propLossQ");
    }

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = ADD;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    addConfig_t *cfg = reserveMemory(sizeof(addConfig_t));
    layerCfg->add = cfg;
    layer->config = layerCfg;

    if (lq == NULL) {
        cfg->forwardMath = (arithmetic_t){ARITH_FLOAT32, HALF_AWAY};
        cfg->propLossMath = (arithmetic_t){ARITH_FLOAT32, HALF_AWAY};
        cfg->outputQ = quantizationInitFloat();
        cfg->propLossQ = quantizationInitFloat();
        cfg->ownsQuantizations = true;
    } else {
        cfg->forwardMath = lq->forwardMath;
        cfg->propLossMath = lq->propLossMath;
        cfg->outputQ = lq->outputQ;
        cfg->propLossQ = lq->propLossQ;
        cfg->ownsQuantizations = false;
    }
    return layer;
}

void freeAddLayer(layer_t *addLayer) {
    addConfig_t *cfg = addLayer->config->add;
    if (cfg->ownsQuantizations) {
        freeQuantization(cfg->outputQ);
        freeQuantization(cfg->propLossQ);
    }
    freeReservedMemory(cfg);
    freeReservedMemory(addLayer->config);
    freeReservedMemory(addLayer);
}
