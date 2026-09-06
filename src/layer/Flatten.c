#define SOURCE_FILE "FLATTEN"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Flatten.h"
#include "Quantization.h"

/* BFP epic PR4 (R-P5, spec §5 "exponent array carried verbatim"): Flatten is a
 * pure reshape — storage order and element count are unchanged, so the packed
 * mantissa payload moves byte-for-byte (calcNumberOfBytesForData is already
 * BFP-correct) and the per-group exponent VALUES are memcpy'd alongside.
 * Flatten has no arithmetic slot at all, so it dispatches on STORAGE dtype:
 * a BFP wire on EITHER side needs a BFP wire on the other (a mixed pair would
 * size the byte memcpy off the wrong side and overrun the packed buffer) and
 * an IDENTICAL grid (the payload is moved verbatim, so a differing grid would
 * silently reinterpret every code) and the SAME element count. The count is
 * checked explicitly and not left to the grid: Flatten's whole purpose is to
 * change the SHAPE, so the two wires legitimately have different dimensions
 * and only their products may agree — and a per-tensor {1, 0} grid matches any
 * product at all, which is exactly the pair that would slip through and let
 * the byte memcpy (sized off the source) overrun the destination.
 * Re-blocking belongs to the Quantization layer. Non-const because
 * calcNumberOfElementsByTensor takes a mutable tensor_t*. */
static void requireMatchingBfpWires(tensor_t *a, tensor_t *b, const char *what) {
    bool aBfp = a->quantization->type == BFP;
    bool bBfp = b->quantization->type == BFP;
    if (!aBfp && !bBfp) {
        return;
    }
    if (aBfp != bBfp) {
        PRINT_ERROR("%s: a BFP wire on one side needs a BFP wire on the other -- got dtypes %d "
                    "and %d; insert a Quantization layer to change dtype",
                    what, (int)a->quantization->type, (int)b->quantization->type);
        exit(1);
    }
    bfpRequireSameGeometry(a->quantization->qConfig, calcNumberOfElementsByTensor(a),
                           b->quantization->qConfig, calcNumberOfElementsByTensor(b), what);
}

void flattenForward(layer_t *flattenLayer, tensor_t *input, tensor_t *output) {
    (void)flattenLayer;
    requireMatchingBfpWires(input, output, "Flatten forward (input -> output)");

    size_t numberOfElements = calcNumberOfElementsByTensor(input);
    size_t numberOfBytes = calcNumberOfBytesForData(input->quantization, numberOfElements);
    memcpy(output->data, input->data, numberOfBytes);

    if (input->quantization->type == SYM_INT32) {
        symInt32QConfig_t *inputQC = input->quantization->qConfig;
        symInt32QConfig_t *outputQC = output->quantization->qConfig;
        outputQC->scale = inputQC->scale;
    } else if (input->quantization->type == BFP) {
        const bfpQConfig_t *inputQC = input->quantization->qConfig;
        bfpQConfig_t *outputQC = output->quantization->qConfig;
        memcpy(outputQC->exponents, inputQC->exponents, inputQC->numGroups);
    }
}

void flattenBackward(layer_t *flattenLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss) {
    (void)flattenLayer;
    (void)forwardInput; /* never dereferenced -> not guarded */
    requireMatchingBfpWires(loss, propLoss, "Flatten backward (loss -> propLoss)");

    size_t numberOfElements = calcNumberOfElementsByTensor(loss);
    size_t numberOfBytes = calcNumberOfBytesForData(loss->quantization, numberOfElements);
    memcpy(propLoss->data, loss->data, numberOfBytes);

    if (loss->quantization->type == SYM_INT32) {
        symInt32QConfig_t *lossQC = loss->quantization->qConfig;
        symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
        propLossQC->scale = lossQC->scale;
    } else if (loss->quantization->type == BFP) {
        const bfpQConfig_t *lossQC = loss->quantization->qConfig;
        bfpQConfig_t *propLossQC = propLoss->quantization->qConfig;
        memcpy(propLossQC->exponents, lossQC->exponents, lossQC->numGroups);
    }
}

void flattenCalcOutputShape(layer_t *flattenLayer, shape_t *inputShape, shape_t *outputShape) {
    (void)flattenLayer;

    size_t batch = inputShape->dimensions[0];
    size_t features = 1;
    for (size_t i = 1; i < inputShape->numberOfDimensions; i++) {
        features *= inputShape->dimensions[i];
    }

    // Precondition: caller allocates outputShape->dimensions and
    // ->orderOfDimensions with >= 2 slots, regardless of input rank.
    outputShape->dimensions[0] = batch;
    outputShape->dimensions[1] = features;
    outputShape->numberOfDimensions = 2;
    setOrderOfDimsForNewTensor(outputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
