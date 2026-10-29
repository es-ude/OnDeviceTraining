#ifndef ODT_LAYER_WIRE_GUARDS_H
#define ODT_LAYER_WIRE_GUARDS_H

#include <stdlib.h>

#include "Arithmetic.h"
#include "Common.h"
#include "Tensor.h"

/* Op-entry shape guards shared by the layers. Header-only static inline -- no
 * new library; every layer target already reaches src/layer/include and links
 * Common, Tensor and Arithmetic. */

/* For the pool kernels (MaxPool1d, AvgPool1d, AdaptiveAvgPool1d), every arm:
 * they index their outputs as DENSE [batch][channels][length] arrays whose
 * batch and channels come from the OPERAND's dims (input forward, lossGrad
 * backward), so an output that disagrees on dim 0 or dim 1 is written past
 * its end even when its length is right — checking dimensions[2] alone is not
 * enough. Rank is checked first: dimensions[0..2] on a rank-2 shape is itself
 * an over-read, which is also why a backward arm rank-checks its dx output
 * before reading that output's length to pass in here. */
static inline void poolRequireDims3(const tensor_t *t, size_t d0, size_t d1, size_t d2,
                                    const char *what) {
    if (t->shape->numberOfDimensions != 3) {
        PRINT_ERROR("%s: expected a rank-3 [batch, channels, length] tensor, got rank %zu", what,
                    t->shape->numberOfDimensions);
        exit(1);
    }
    if (t->shape->dimensions[0] != d0 || t->shape->dimensions[1] != d1 ||
        t->shape->dimensions[2] != d2) {
        PRINT_ERROR("%s: expected shape [%zu, %zu, %zu], got [%zu, %zu, %zu]", what, d0, d1, d2,
                    t->shape->dimensions[0], t->shape->dimensions[1], t->shape->dimensions[2]);
        exit(1);
    }
}

static inline void layerRequireRank(const tensor_t *t, const shape_t *expected, const char *where,
                                    const char *what) {
    if (t->shape->numberOfDimensions != expected->numberOfDimensions) {
        PRINT_ERROR("%s: %s must have rank %zu, got rank %zu", where, what,
                    expected->numberOfDimensions, t->shape->numberOfDimensions);
        exit(1);
    }
}

/* For a wire a kernel addresses by raw flat index (every GEMM-produced wire,
 * and the conv kernels' consumed loss): they take the physical dims and ignore
 * orderOfDimensions, so a transposed view with the right logical dims would be
 * silently mis-placed. Hence physical dims AND identity order. */
static inline void layerRequireDenseWire(const tensor_t *t, const shape_t *expected,
                                         const char *where, const char *what) {
    layerRequireRank(t, expected, where, what);
    for (size_t d = 0; d < expected->numberOfDimensions; d++) {
        const char *kind = NULL;
        size_t want = 0;
        size_t got = 0;
        if (t->shape->dimensions[d] != expected->dimensions[d]) {
            kind = "dim";
            want = expected->dimensions[d];
            got = t->shape->dimensions[d];
        } else if (t->shape->orderOfDimensions[d] != d) {
            kind = "order";
            want = d;
            got = t->shape->orderOfDimensions[d];
        }
        if (kind != NULL) {
            PRINT_ERROR("%s: %s %s %zu must be %zu, got %zu (dense identity-order wire)", where,
                        what, kind, d, want, got);
            exit(1);
        }
    }
}

/* For a consumed wire whose kernel reads it order-aware (Matmul's operands):
 * a zero-copy transposed view is legal, so only the logical dims must match. */
static inline void layerRequireLogicalShape(const tensor_t *t, const shape_t *expected,
                                            const char *where, const char *what) {
    layerRequireRank(t, expected, where, what);
    for (size_t d = 0; d < expected->numberOfDimensions; d++) {
        size_t got = getDimensionsByIndex((tensor_t *)t, d);
        if (got != expected->dimensions[d]) {
            PRINT_ERROR("%s: %s logical dim %zu must be %zu, got %zu", where, what, d,
                        expected->dimensions[d], got);
            exit(1);
        }
    }
}

#endif // ODT_LAYER_WIRE_GUARDS_H
