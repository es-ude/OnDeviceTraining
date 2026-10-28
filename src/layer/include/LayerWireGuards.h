#ifndef ODT_LAYER_WIRE_GUARDS_H
#define ODT_LAYER_WIRE_GUARDS_H

#include <stdlib.h>

#include "Common.h"
#include "Tensor.h"

/* Op-entry shape guards shared by the layers. Header-only static inline -- no
 * new library; every layer target already reaches src/layer/include and links
 * Common and Tensor. */

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

#endif // ODT_LAYER_WIRE_GUARDS_H
