#ifndef SYM_GROUPED_WEIGHT_H
#define SYM_GROUPED_WEIGHT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Quantization.h"
#include "SymAccumGuard.h"
#include "Tensor.h"

/* The grouped weight the SYM GEMM-family kernels consume
 * (matmulSymInt32TensorsGroupedWeight, conv1dKernelSymInt32Grouped,
 * convTranspose1dKernelSymInt32Grouped): the group config (per-group scales,
 * group shape, stored qBits) plus the CODE DOMAIN of the unpacked int32
 * mantissas the funnel staged, which the config's qBits alone cannot name. A
 * stored grouped SYM weight unpacks to two's-complement qBits codes; a stored
 * grouped ASYM weight unpacks to the image code - zp[g] (ExecuteOp.c) -- the
 * same mantissa the kernels then scale per group, but spanning
 * +-(2^qBits - 1), one code wider on each side than qBits reads. The kernels
 * take the weight's operand range from this tag (symGroupedWeightRange) for
 * the accumulator guard's proof AND the declared output width
 * (SymAccumGuard.h), so a true SYM weight never pays a bit it cannot use and
 * the ASYM image is covered exactly. The tag is the dtype's STATIC range, not
 * a per-group zero point: a declared width must not drift with the data a
 * requant writes. No valid tag is 0: a zeroed or partially designated ctx
 * (`.domain` left out) must hit symGroupedWeightRange's fail-fast, never read
 * as a SYM weight, because the SYM range is the UNSAFE one for an ASYM image
 * (the guard would prove a site that can overflow). Header-only static
 * inline, like SymAccumGuard.h. */
typedef enum symCodeDomain {
    SYM_CODES_TWOS_COMPLEMENT = 1, /* [-2^(q-1), 2^(q-1) - 1]: a stored grouped SYM weight */
    SYM_CODES_ASYM_IMAGE = 2,      /* code - zp[g] in [-(2^q - 1), 2^q - 1]: a stored grouped
                                      ASYM weight */
} symCodeDomain_t;

typedef struct symGroupedWeight {
    const symQConfig_t *groups; /* per-group scales, group shape, stored qBits */
    symCodeDomain_t domain;
} symGroupedWeight_t;

/* The signed range of the weight's unpacked codes, from its code domain and
 * stored qBits (SYM qBits clamp to [1, 32], an ASYM image to int32). */
static inline void symGroupedWeightRange(const symGroupedWeight_t *w, int32_t *min, int32_t *max) {
    switch (w->domain) {
    case SYM_CODES_TWOS_COMPLEMENT:
        symAccumWidthRange(w->groups->qBits, min, max);
        return;
    case SYM_CODES_ASYM_IMAGE: {
        unsigned q = symAccumClampedWidth(w->groups->qBits);
        uint64_t reach = ((uint64_t)1 << q) - 1u;
        if (reach > (uint64_t)INT32_MAX) {
            reach = (uint64_t)INT32_MAX;
        }
        *min = -(int32_t)reach;
        *max = (int32_t)reach;
        return;
    }
    }
    PRINT_ERROR("symGroupedWeightRange: unknown code domain tag %d (0 = the ctx's .domain was "
                "never set; a grouped weight is SYM_CODES_TWOS_COMPLEMENT or "
                "SYM_CODES_ASYM_IMAGE)",
                (int)w->domain);
    exit(1);
}

/* Grouped-weight detection across BOTH grouped carrier dtypes (SYM and ASYM
 * share the {numGroups, groupSize} shape grammar, group-quant D6). Fills *ctx
 * and returns it iff the stored weight is grouped (numGroups > 1), else NULL.
 * Grouped SYM: ctx->groups is the weight's OWN config. Grouped ASYM: *asymView
 * (the CALLER's stack storage) is filled as a symQConfig-shaped VIEW of the
 * asym config -- legitimate because the kernels read only
 * scales/numGroups/groupSize/qBits, fields both grammars share, and the
 * funnel prologue has already shifted the codes into the signed-mantissa
 * image (ExecuteOp.c -- group-quant D5: the grouped ASYM compute path IS the
 * grouped SYM path on shifted mantissas), so the zeroPoints never reach the
 * kernel; the image's wider range travels as ctx->domain. VIEW LIFETIME:
 * scales is BORROWED from the asym config (never free through the view) and
 * the view lives in the caller's frame -- valid for the duration of the
 * executeOp call the ctx is passed into, never stored beyond it. Set together
 * with the op's groupedSymOperandPos (both or neither, Linear.c). */
static inline const symGroupedWeight_t *
symGroupedWeightOrNull(const tensor_t *weights, symQConfig_t *asymView, symGroupedWeight_t *ctx) {
    if (weights->quantization->type == SYM) {
        const symQConfig_t *qc = weights->quantization->qConfig;
        if (qc->numGroups <= 1) {
            return NULL;
        }
        *ctx = (symGroupedWeight_t){.groups = qc, .domain = SYM_CODES_TWOS_COMPLEMENT};
        return ctx;
    }
    if (weights->quantization->type == ASYM) {
        const asymQConfig_t *qc = weights->quantization->qConfig;
        if (qc->numGroups <= 1) {
            return NULL;
        }
        *asymView = (symQConfig_t){.scales = qc->scales, /* BORROWED (see above) */
                                   .numGroups = qc->numGroups,
                                   .groupSize = qc->groupSize,
                                   .roundingMode = qc->roundingMode,
                                   .qBits = qc->qBits};
        *ctx = (symGroupedWeight_t){.groups = asymView, .domain = SYM_CODES_ASYM_IMAGE};
        return ctx;
    }
    return NULL;
}

#endif // SYM_GROUPED_WEIGHT_H
