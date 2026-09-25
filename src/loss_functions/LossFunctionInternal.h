#ifndef LOSS_FUNCTION_INTERNAL_H
#define LOSS_FUNCTION_INTERNAL_H

/* Private to src/loss_functions (MSE.c, CrossEntropy.c). Include AFTER the
 * file's SOURCE_FILE define: PRINT_ERROR reports the includer's name.
 *
 * Every operand of a loss must have the MODEL OUTPUT's exact shape (#153),
 * and the check sits at each PUBLIC dispatcher, above the dtype switch,
 * because every arm needs it (PR4 adversarial gate F1/F2, hoisted by delta
 * D0). Fake-quant arms size each VLA scratch from the output's count while
 * each convertTensor walks its OWN operand's count, so a longer operand
 * overruns the scratch; FLOAT32 arms index every operand at the output's
 * count, so a shorter one is read -- and for the backward's grad wire,
 * written -- out of bounds. Equal counts are not enough either. A label with
 * another layout at the same count ([B, C, L] vs [B, L, C]) would be scored
 * element-by-element against the wrong values, and a label with another rank
 * ([C] vs [1, C]) leaves the batch axis to be guessed -- the CE MEAN divisor
 * and the MSE mean scale read dims[0] -- and hides a missed or doubled batch
 * wrap.
 *
 * The output must be [B, ...] with a feature axis (rank >= 2; a rank-1
 * tensor would leave the batch axis to be guessed) and at least one element
 * (a zero anywhere in the shape makes zero-length VLAs and 0/0 means).
 * Fail fast: a mismatch is a caller's shape bug. The *Float arm bodies stay
 * unguarded on purpose; the dispatchers are the guarded API. */

#include <stdlib.h>

#include "Common.h"
#include "Tensor.h"

static inline void requireBatchedOutput(tensor_t *output, const char *role, const char *what) {
    size_t rank = output->shape->numberOfDimensions;
    if (rank < 2) {
        PRINT_ERROR("%s: %s shape has rank %zu, needs rank >= 2 ([B, ...] with a feature axis)",
                    what, role, rank);
        exit(1);
    }
    size_t count = calcNumberOfElementsByTensor(output);
    if (count == 0) {
        PRINT_ERROR("%s: %s shape has 0 elements (dims[0] = %zu), needs at least 1", what, role,
                    output->shape->dimensions[0]);
        exit(1);
    }
}

static inline void requireOperandMatchesOutput(tensor_t *output, tensor_t *operand,
                                               const char *operandName, const char *what) {
    requireBatchedOutput(output, "model output", what);
    size_t rank = output->shape->numberOfDimensions;
    size_t operandRank = operand->shape->numberOfDimensions;
    if (operandRank != rank) {
        PRINT_ERROR("%s: %s shape has rank %zu, the model output has rank %zu", what, operandName,
                    operandRank, rank);
        exit(1);
    }
    for (size_t k = 0; k < rank; k++) {
        size_t operandDim = operand->shape->dimensions[k];
        size_t outputDim = output->shape->dimensions[k];
        if (operandDim != outputDim) {
            PRINT_ERROR("%s: %s shape dims[%zu] = %zu, the model output has %zu", what, operandName,
                        k, operandDim, outputDim);
            exit(1);
        }
    }
}

#endif // LOSS_FUNCTION_INTERNAL_H
