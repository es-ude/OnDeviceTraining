#define SOURCE_FILE "LAYER_QUANT"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "LayerQuant.h"
#include "StorageApi.h"

void layerQuantInitUniform(layerQuant_t *lq, quantization_t *q) {
    arithmetic_t a = arithmeticFromQuantization(q);
    lq->forwardMath = a;
    lq->weightGradMath = a;
    lq->biasGradMath = a;
    lq->propLossMath = a;

    lq->outputQ = q;
    lq->propLossQ = q;
    lq->weightStorage = q;
    lq->biasStorage = q;
    lq->weightGradStorage = NULL;
    lq->biasGradStorage = NULL;

    /* Both DYNAMIC: the only accumulate mode valid for every layer's grad
     * intermediate (FLOAT32 or SYM_INT32), so this convenience default can
     * never abort a layer -- e.g. LayerNorm's FLOAT32 beta-grad intermediate,
     * which OUT_ACC_FIXED_SCALE rejects. Opt a layer into a fixed-scale-integer
     * bias-grad scheme (Linear/Conv) via biasGradAccMode explicitly. */
    lq->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    lq->biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
}

quantization_t *deepCopyQuantization(quantization_t *src) {
    if (src == NULL) {
        return NULL;
    }

    quantization_t *dst = reserveMemory(sizeof(quantization_t));
    dst->type = src->type;

    size_t cfgSize = 0;
    switch (src->type) {
    case FLOAT32:
        cfgSize = 0;
        break;
    case INT32:
        cfgSize = 0;
        break;
    case BOOL:
        cfgSize = 0;
        break;
    case SYM_INT32:
        cfgSize = sizeof(symInt32QConfig_t);
        break;
    case SYM:
        cfgSize = sizeof(symQConfig_t);
        break;
    case ASYM:
        cfgSize = sizeof(asymQConfig_t);
        break;
    default:
        PRINT_ERROR("deepCopyQuantization: unknown quantization type %d", (int)src->type);
        exit(1);
    }

    if (cfgSize == 0) {
        dst->qConfig = NULL;
    } else {
        dst->qConfig = reserveMemory(cfgSize);
        memcpy(dst->qConfig, src->qConfig, cfgSize);
        if (src->type == SYM) {
            /* Group-quant PR1: symQConfig_t carries a heap `scales` pointer --
             * the memcpy above just copied that POINTER, so dst and src now
             * alias the same array (double-free / shared-mutation hazard).
             * Replace it with a genuine deep copy: a fresh array, sized from
             * src's own numGroups (PR1: always 1), values copied. */
            symQConfig_t *srcQC = src->qConfig;
            symQConfig_t *dstQC = dst->qConfig;
            float *scales = reserveMemory(srcQC->numGroups * sizeof(float));
            memcpy(scales, srcQC->scales, srcQC->numGroups * sizeof(float));
            dstQC->scales = scales;
        }
    }
    return dst;
}
