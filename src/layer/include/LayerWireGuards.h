#ifndef ODT_LAYER_WIRE_GUARDS_H
#define ODT_LAYER_WIRE_GUARDS_H

#include <stdlib.h>

#include "Arithmetic.h"
#include "Common.h"
#include "Tensor.h"

/* Op-entry shape guards shared by the layers. Header-only static inline -- no
 * new library; every layer target already reaches src/layer/include and links
 * Common, Tensor and Arithmetic. */

static inline void layerRequireRank(const tensor_t *t, const shape_t *expected, const char *where,
                                    const char *what) {
    if (t->shape->numberOfDimensions != expected->numberOfDimensions) {
        PRINT_ERROR("%s: %s must have rank %zu, got rank %zu", where, what,
                    expected->numberOfDimensions, t->shape->numberOfDimensions);
        exit(1);
    }
}

/* For a wire a kernel addresses by raw flat index (every GEMM-produced wire,
 * the conv kernels' consumed loss, the pool kernels' output, loss and argmax): they take the
 * physical dims and ignore orderOfDimensions, so a transposed view with the right logical dims
 * would be silently mis-placed. Hence physical dims AND identity order. */
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
