#define SOURCE_FILE "param_gate"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "BfpKernelSupport.h"
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
    if (N == 0 || outCh == 0 || N % outCh != 0) {
        PRINT_ERROR("resolveGroupShape: N=%zu outCh=%zu is not the element count and dim 0 of "
                    "a non-empty tensor (need N > 0, outCh > 0, N %% outCh == 0)",
                    N, outCh);
        exit(1);
    }
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

groupShape_t resolveWireShape(size_t N, wireBlockSweep_t mode, int size) {
    if (N == 0) {
        PRINT_ERROR("resolveWireShape: a wire has no elements (N == 0)");
        exit(1);
    }
    if (mode != WIRE_BLOCK_SIZE) {
        return (groupShape_t){.numGroups = 1, .groupSize = 0};
    }
    if (size <= 0) {
        PRINT_ERROR("resolveWireShape: BFP_WIRE_BLOCK must be a positive block size, got %d", size);
        exit(1);
    }
    size_t g = (size_t)size;
    if (N % g != 0 || N / g <= 1) {
        return (groupShape_t){.numGroups = 1, .groupSize = 0}; /* g == N or fallback */
    }
    return (groupShape_t){.numGroups = N / g, .groupSize = g};
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
    /* Validate the EXPECTATION dtype before anything else -- including before
     * the actual-vs-expected mismatch check below. A caller passing an
     * unsupported expect->type (INT32/SYM_INT32/BOOL) must fail fast
     * regardless of the tensor's actual type: that mismatch is the common
     * case for exactly the programmer error this guard exists to catch, so
     * gating the check on actualType == expect->type would let most such
     * mistakes slip through as a benign "expected X, got Y" false result. */
    switch (expect->type) {
    case FLOAT32:
    case SYM:
    case ASYM:
    case BFP:
        break;
    default:
        PRINT_ERROR("paramGateCheck: no gate arm for expectation dtype %s",
                    quantTypeToString(expect->type));
        exit(1);
    }
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
        /* Unreachable: expect->type was validated above. Kept explicit (not
         * folded away) per the repo's explicit-switch/fail-fast-default
         * convention for dtype dispatch. */
        PRINT_ERROR("paramGateCheck: no gate arm for expectation dtype %s",
                    quantTypeToString(expect->type));
        exit(1);
    }
}

size_t packedPayloadBytes(qtype_t type, uint8_t bits, size_t N) {
    switch (type) {
    case FLOAT32:
        return N * sizeof(float);
    case SYM:
    case ASYM:
    case BFP:
        return ((size_t)bits * N + 7) / 8;
    default:
        PRINT_ERROR("packedPayloadBytes: no sweep accounting for %s", quantTypeToString(type));
        exit(1);
    }
}

size_t packedMetadataBytes(qtype_t type, size_t numGroups) {
    switch (type) {
    case FLOAT32:
        return 0;
    case BFP:
        return numGroups * sizeof(uint8_t);
    case SYM:
        return numGroups * sizeof(float);
    case ASYM:
        return numGroups * (sizeof(float) + sizeof(uint16_t));
    default:
        PRINT_ERROR("packedMetadataBytes: no sweep accounting for %s", quantTypeToString(type));
        exit(1);
    }
}

/* strtol with full-consumption + range check; false on any junk. Rejects
 * anything strtol would accept but the name grammar would not: leading
 * whitespace and a leading '+' are only valid because strtol skips/permits
 * them, not because the knob's value grammar does. */
static bool parseIntStrict(const char *s, long lo, long hi, long *out) {
    if (s == NULL || s[0] == '\0') {
        return false;
    }
    if (!isdigit((unsigned char)s[0])) {
        return false;
    }
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || *end != '\0' || v < lo || v > hi) {
        return false;
    }
    *out = v;
    return true;
}

static const char *envOrEmpty(const char *name) {
    const char *v = getenv(name);
    return (v != NULL) ? v : "";
}

const char *bfpSweepConfigFromEnv(bfpSweepConfig_t *out) {
    *out = (bfpSweepConfig_t){.mantissaBits = 8,
                              .exponentBits = 8,
                              .weightMode = GROUP_MODE_TENSOR,
                              .weightSize = 0,
                              .wireMode = WIRE_BLOCK_FLOAT,
                              .wireSize = 0,
                              .math = BFP_MATH_NATIVE,
                              .bfpGrads = false,
                              .bfpState = false,
                              .rounding = BFP_ROUNDING_SR,
                              .ignoredLegacyKnobs = 0u};
    strncpy(out->weightBlockStr, "tensor", sizeof(out->weightBlockStr));
    strncpy(out->wireBlockStr, "float", sizeof(out->wireBlockStr));

    const char *s;
    long v;
    s = envOrEmpty("BFP_MANTISSA_BITS");
    if (s[0] != '\0') {
        if (!parseIntStrict(s, 2, 16, &v)) {
            return "BFP_MANTISSA_BITS must be an integer in [2, 16]";
        }
        out->mantissaBits = (uint8_t)v;
    }
    s = envOrEmpty("BFP_EXPONENT_BITS");
    if (s[0] != '\0') {
        if (!parseIntStrict(s, 2, 8, &v)) {
            return "BFP_EXPONENT_BITS must be an integer in [2, 8]";
        }
        out->exponentBits = (uint8_t)v;
    }
    s = envOrEmpty("BFP_WEIGHT_BLOCK");
    if (s[0] != '\0') {
        if (strcmp(s, "tensor") == 0) {
            out->weightMode = GROUP_MODE_TENSOR;
        } else if (strcmp(s, "channel") == 0) {
            out->weightMode = GROUP_MODE_CHANNEL;
        } else if (parseIntStrict(s, 1, INT_MAX, &v)) {
            out->weightMode = GROUP_MODE_SIZE;
            out->weightSize = (int)v;
        } else {
            return "BFP_WEIGHT_BLOCK must be tensor, channel, or a positive integer";
        }
        strncpy(out->weightBlockStr, s, sizeof(out->weightBlockStr) - 1);
        out->weightBlockStr[sizeof(out->weightBlockStr) - 1] = '\0';
    }
    s = envOrEmpty("BFP_WIRE_BLOCK");
    if (s[0] != '\0') {
        if (strcmp(s, "float") == 0) {
            out->wireMode = WIRE_BLOCK_FLOAT;
        } else if (strcmp(s, "tensor") == 0) {
            out->wireMode = WIRE_BLOCK_TENSOR;
        } else if (parseIntStrict(s, 1, INT_MAX, &v)) {
            out->wireMode = WIRE_BLOCK_SIZE;
            out->wireSize = (int)v;
        } else {
            return "BFP_WIRE_BLOCK must be float, tensor, or a positive integer";
        }
        strncpy(out->wireBlockStr, s, sizeof(out->wireBlockStr) - 1);
        out->wireBlockStr[sizeof(out->wireBlockStr) - 1] = '\0';
    }
    s = envOrEmpty("BFP_MATH");
    if (s[0] != '\0') {
        if (strcmp(s, "native") == 0) {
            out->math = BFP_MATH_NATIVE;
        } else if (strcmp(s, "fq") == 0) {
            out->math = BFP_MATH_FQ;
        } else {
            return "BFP_MATH must be native or fq";
        }
    }
    s = envOrEmpty("BFP_GRADS");
    if (s[0] != '\0') {
        if (!parseIntStrict(s, 0, 1, &v)) {
            return "BFP_GRADS must be 0 or 1";
        }
        out->bfpGrads = (v == 1);
    }
    s = envOrEmpty("BFP_STATE");
    if (s[0] != '\0') {
        if (!parseIntStrict(s, 0, 1, &v)) {
            return "BFP_STATE must be 0 or 1";
        }
        out->bfpState = (v == 1);
    }
    if (out->bfpState && !out->bfpGrads) {
        return "BFP_STATE=1 requires BFP_GRADS=1 (the storage ladder is weights -> +grads -> "
               "+state)";
    }
    s = envOrEmpty("BFP_ROUNDING");
    if (s[0] != '\0') {
        if (strcmp(s, "sr") == 0) {
            out->rounding = BFP_ROUNDING_SR;
        } else if (strcmp(s, "det") == 0) {
            out->rounding = BFP_ROUNDING_DET;
        } else {
            return "BFP_ROUNDING must be sr or det";
        }
    }

    static const struct {
        const char *name;
        unsigned bit;
    } legacy[] = {
        {"SYM_BITS", LEGACY_KNOB_SYM_BITS},
        {"SYM_WIRES", LEGACY_KNOB_SYM_WIRES},
        {"WEIGHT_DTYPE", LEGACY_KNOB_WEIGHT_DTYPE},
        {"GROUP_MODE", LEGACY_KNOB_GROUP_MODE},
        {"GROUP_SIZE", LEGACY_KNOB_GROUP_SIZE},
        {"SYM_ROUNDING", LEGACY_KNOB_SYM_ROUNDING},
        {"ODTS_ROUNDTRIP", LEGACY_KNOB_ODTS_ROUNDTRIP},
    };
    for (size_t i = 0; i < sizeof(legacy) / sizeof(legacy[0]); i++) {
        if (envOrEmpty(legacy[i].name)[0] != '\0') {
            out->ignoredLegacyKnobs |= legacy[i].bit;
        }
    }
    return NULL;
}

bool bfpBlockHeadroomFits(uint8_t ma, uint8_t mb, size_t runA, size_t runB, size_t reductionLen) {
    size_t a = (runA == 0) ? reductionLen : runA;
    size_t b = (runB == 0) ? reductionLen : runB;
    size_t maxSeg = a < b ? a : b;
    if (maxSeg > reductionLen) {
        maxSeg = reductionLen;
    }
    return maxSeg <= bfpSegmentLimit(ma, mb);
}

bool bfpSumHeadroomFits(uint8_t m, size_t run, size_t reductionLen) {
    size_t maxSeg = (run == 0) ? reductionLen : run;
    if (maxSeg > reductionLen) {
        maxSeg = reductionLen;
    }
    return maxSeg <= bfpSumSegmentLimit(m);
}
