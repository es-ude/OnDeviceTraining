#define SOURCE_FILE "param_gate"

#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "Quantization.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "param_gate.h"

/* HAR weight tensors (N = element count, outCh = dim-0 size, pc = N/outCh =
 * elements per output channel; Conv weight = [outCh, inCh/groups, K], Linear
 * = [outFeatures, inFeatures], so outCh is always dim 0):
 *   layer   shape         N      outCh  pc   G64                G32
 *   conv1   [16, 9, 7]    1008   16     63   FALLBACK->pc=63    FALLBACK->pc=63
 *   conv2   [32,16, 5]    2560   32     80   ->40 groups        ->80 groups
 *   conv3   [64,32, 3]    6144   64     96   ->96 groups        ->192 groups
 *   linear  [6, 64]       384    6      64   ->6 groups (==pc)  ->12 groups
 * (conv1's N=1008=2^4*3^2*7: neither 64 nor 32 divide it, so both G64 and G32
 * fall back to per-channel (pc=63=3^2*7) for that layer only.) */
groupShape_t resolveGroupShape(size_t N, size_t outCh, groupModeSweep_t mode, int groupSizeEnv) {
    if (mode == GROUP_MODE_TENSOR) {
        return (groupShape_t){.numGroups = 1, .groupSize = 0};
    }
    size_t groupSize;
    if (mode == GROUP_MODE_CHANNEL) {
        groupSize = N / outCh;
    } else { /* GROUP_MODE_SIZE */
        size_t requested = (size_t)groupSizeEnv;
        groupSize = (requested > 0 && N % requested == 0) ? requested : N / outCh;
    }
    size_t numGroups = N / groupSize;
    if (numGroups <= 1) {
        return (groupShape_t){.numGroups = 1, .groupSize = 0};
    }
    return (groupShape_t){.numGroups = numGroups, .groupSize = groupSize};
}

/* symQConfig_t / asymQConfig_t / bfpQConfig_t share the field NAMES read
 * here but differ in layout, so the qtype decides which struct qConfig
 * points to. Explicit switch, fail-fast default: an "else is SYM" arm would
 * read a float qConfig as a symQConfig_t. */
qShapeView_t viewQShape(const quantization_t *q) {
    switch (q->type) {
    case SYM: {
        const symQConfig_t *sc = q->qConfig;
        return (qShapeView_t){.qBits = sc->qBits,
                              .exponentBits = 0,
                              .numGroups = sc->numGroups,
                              .groupSize = sc->groupSize};
    }
    case ASYM: {
        const asymQConfig_t *ac = q->qConfig;
        return (qShapeView_t){.qBits = ac->qBits,
                              .exponentBits = 0,
                              .numGroups = ac->numGroups,
                              .groupSize = ac->groupSize};
    }
    case BFP: {
        const bfpQConfig_t *bc = q->qConfig;
        return (qShapeView_t){.qBits = bc->mantissaBits,
                              .exponentBits = bc->exponentBits,
                              .numGroups = bc->numGroups,
                              .groupSize = bc->groupSize};
    }
    default:
        PRINT_ERROR("viewQShape: %s carries no group shape", quantTypeToString(q->type));
        exit(1);
    }
}

bool paramGateCheck(const tensor_t *tensor, const paramGateExpect_t *expect, char *msg,
                    size_t msgLen) {
    qtype_t actualType = tensor->quantization->type;
    if (actualType != expect->type) {
        snprintf(msg, msgLen, "expected %s, got %s", quantTypeToString(expect->type),
                 quantTypeToString(actualType));
        return false;
    }
    switch (expect->type) {
    case FLOAT32:
        return true;
    case SYM:
    case ASYM:
    case BFP: {
        qShapeView_t actual = viewQShape(tensor->quantization);
        const char *bitsName = (expect->type == BFP) ? "mantissaBits" : "qBits";
        if (actual.qBits != expect->bits) {
            snprintf(msg, msgLen, "expected %s %u, got %u", bitsName, (unsigned)expect->bits,
                     (unsigned)actual.qBits);
            return false;
        }
        if (expect->type == BFP && actual.exponentBits != expect->exponentBits) {
            snprintf(msg, msgLen, "expected exponentBits %u, got %u",
                     (unsigned)expect->exponentBits, (unsigned)actual.exponentBits);
            return false;
        }
        if (actual.numGroups != expect->shape.numGroups ||
            actual.groupSize != expect->shape.groupSize) {
            snprintf(msg, msgLen, "expected group shape {%zu,%zu}, got {%zu,%zu}",
                     expect->shape.numGroups, expect->shape.groupSize, actual.numGroups,
                     actual.groupSize);
            return false;
        }
        return true;
    }
    default:
        PRINT_ERROR("paramGateCheck: no gate arm for expectation dtype %s",
                    quantTypeToString(expect->type));
        exit(1);
    }
}
