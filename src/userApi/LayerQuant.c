#define SOURCE_FILE "LAYER_QUANT"

#include <string.h>

#include "LayerQuant.h"
#include "ScaleCanon.h"
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
    case BFP:
        cfgSize = sizeof(bfpQConfig_t);
        break;
    case BFP_RAW:
        PRINT_ERROR("deepCopyQuantization: BFP_RAW is executeOp scratch, never copied");
        exit(1);
    }

    if (cfgSize == 0) {
        dst->qConfig = NULL;
    } else {
        dst->qConfig = reserveMemory(cfgSize);
        memcpy(dst->qConfig, src->qConfig, cfgSize);
        if (src->type == SYM) {
            symQConfig_t *srcQC = src->qConfig;
            symQConfig_t *dstQC = dst->qConfig;
            odtScale_t *scales = reserveMemory(srcQC->numGroups * sizeof(*scales));
            copyCanonScales(scales, srcQC->scales, srcQC->numGroups);
            dstQC->scales = scales;
        }
        if (src->type == ASYM) {
            asymQConfig_t *srcQC = src->qConfig;
            asymQConfig_t *dstQC = dst->qConfig;
            odtScale_t *scales = reserveMemory(srcQC->numGroups * sizeof(*scales));
            copyCanonScales(scales, srcQC->scales, srcQC->numGroups);
            dstQC->scales = scales;
            uint16_t *zeroPoints = reserveMemory(srcQC->numGroups * sizeof(uint16_t));
            memcpy(zeroPoints, srcQC->zeroPoints, srcQC->numGroups * sizeof(uint16_t));
            dstQC->zeroPoints = zeroPoints;
        }
        if (src->type == BFP) {
            bfpQConfig_t *srcQC = src->qConfig;
            bfpQConfig_t *dstQC = dst->qConfig;
            uint8_t *exponents = reserveMemory(srcQC->numGroups * sizeof(uint8_t));
            memcpy(exponents, srcQC->exponents, srcQC->numGroups * sizeof(uint8_t));
            dstQC->exponents = exponents;
        }
    }
    return dst;
}
