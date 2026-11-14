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

/* The grouped weight the SYM GEMM-family kernels consume: the group config plus
 * the code domain of the unpacked mantissas, which qBits alone cannot name. A
 * grouped SYM weight unpacks to two's-complement qBits codes; a grouped ASYM
 * weight to code - zp[g], spanning +-(2^qBits - 1). The kernels take the weight's
 * range from this tag for both the overflow proof and the declared output width.
 * It is the dtype's static range, not the current zero points, so a declared
 * width never drifts with the data. Build the ctx with symGroupedWeightOrNull. */
/* The wider ASYM range is 0, so a ctx whose .domain was never set gets a
 * conservative proof (never an unsafe one). */
typedef enum symCodeDomain {
    SYM_CODES_ASYM_IMAGE = 0,      /* code - zp[g] in [-(2^q - 1), 2^q - 1]: a stored grouped
                                      ASYM weight */
    SYM_CODES_TWOS_COMPLEMENT = 1, /* [-2^(q-1), 2^(q-1) - 1]: a stored grouped SYM weight */
} symCodeDomain_t;

typedef struct symGroupedWeight {
    const symQConfig_t *groups; /* per-group scales, group shape, stored qBits */
    symCodeDomain_t domain;
} symGroupedWeight_t;

/* The signed range of the weight's unpacked codes. ASYM qBits are in [1, 16]
 * (validated at quantization init), so the image range fits int32. */
static inline void symGroupedWeightRange(const symGroupedWeight_t *w, int32_t *min, int32_t *max) {
    switch (w->domain) {
    case SYM_CODES_TWOS_COMPLEMENT:
        symAccumWidthRange(w->groups->qBits, min, max);
        return;
    case SYM_CODES_ASYM_IMAGE: {
        int32_t reach = (int32_t)((1u << w->groups->qBits) - 1u);
        *min = -reach;
        *max = reach;
        return;
    }
    }
}

/* Returns the grouped-weight ctx for a grouped SYM or ASYM weight (numGroups > 1),
 * else NULL. For ASYM, *asymView (caller's stack) becomes a symQConfig-shaped view:
 * the kernels read only scales/numGroups/groupSize/qBits, and the funnel already
 * shifted the codes by the zero points. The view borrows scales: never free
 * through it, and use it only within the executeOp call the ctx is passed to.
 * Set the op's groupedSymOperandPos whenever the ctx is used (both or neither). */
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
        *asymView = (symQConfig_t){.scales = qc->scales,
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
