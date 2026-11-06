#ifndef TENSOR_ROWS_H
#define TENSOR_ROWS_H

#include <stdlib.h>

#include "Common.h"
#include "Tensor.h"

/* The row geometry of a class-probability wire (#152), shared by Softmax and
 * CrossEntropy's one-hot rows so the two cannot disagree: row = axis 0, every
 * row spans all elements after axis 0 (crossEntropyForwardFloat's MEAN rule);
 * a rank-1 tensor is one row. The row count is the STORAGE dimensions[0]
 * (transposeTensor permutes orderOfDimensions only, never dimensions). rowLen
 * is the product of the trailing dims, not count / rows, so an empty leading
 * axis cannot divide by zero. A walk over physical storage needs identity
 * order for more than one storage row (GroupNorm.c's rule): a permuted view
 * would silently treat a partition that is not its logical rows. One storage
 * row is the whole tensor for any order -- so a logical [1, N] stored as
 * [N, 1] + transpose fails fast. `where` names the caller in the message. */
static inline void tensorRowGeometry(const tensor_t *t, const char *where, size_t *rows,
                                     size_t *rowLen) {
    const shape_t *s = t->shape;
    size_t first = (s->numberOfDimensions >= 2) ? 1 : 0;
    *rows = (first == 1) ? s->dimensions[0] : 1;
    size_t len = 1;
    for (size_t d = first; d < s->numberOfDimensions; d++) {
        len *= s->dimensions[d];
    }
    *rowLen = len;
    if (*rows > 1) {
        for (size_t d = 0; d < s->numberOfDimensions; d++) {
            if (s->orderOfDimensions[d] != d) {
                PRINT_ERROR("%s: a wire with %zu storage rows must be identity-order (dim %zu "
                            "is order %zu)",
                            where, *rows, d, s->orderOfDimensions[d]);
                exit(1);
            }
        }
    }
}

#endif /* TENSOR_ROWS_H */
