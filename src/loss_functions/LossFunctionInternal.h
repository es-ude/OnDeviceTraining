#ifndef LOSS_FUNCTION_INTERNAL_H
#define LOSS_FUNCTION_INTERNAL_H

/* Private to src/loss_functions (MSE.c, CrossEntropy.c). Include AFTER the
 * file's SOURCE_FILE define: PRINT_ERROR reports the includer's name.
 *
 * Each public dispatcher validates its output once with requireBatchedOutput, then each
 * operand with requireOperandMatchesOutput. Output must be nonempty [B,...], rank >= 2; operands
 * must share its rank and dimensions (orderOfDimensions is not compared). Float kernels trust
 * these checks and read every operand at the output's count; fake-quant scratch is sized from
 * the output. MSE's FLOAT32 arm reads label/result data as float, so it also requires FLOAT32
 * operands (requireFloat32Operand); CE's FLOAT32 arm sends a quantized label or grad wire
 * through its converting arm instead. */

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

/* Call after requireBatchedOutput(output). */
static inline void requireOperandMatchesOutput(tensor_t *output, tensor_t *operand,
                                               const char *operandName, const char *what) {
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

static inline void requireFloat32Operand(tensor_t *operand, const char *operandName,
                                         const char *what) {
    if (operand->quantization->type != FLOAT32) {
        PRINT_ERROR("%s: %s dtype %d must be FLOAT32 for a FLOAT32 model output", what, operandName,
                    (int)operand->quantization->type);
        exit(1);
    }
}

#endif // LOSS_FUNCTION_INTERNAL_H
