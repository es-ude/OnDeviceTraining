#define SOURCE_FILE "FLATTEN"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Flatten.h"
#include "Quantization.h"
#include "ScaleCanon.h"

/* Copies physical storage without reblocking; BFP pairs require equal counts, grids and widths.
 * A BFP wire on one side needs one on the other, or the byte copy is sized off the wrong side. */
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

void flattenForward(layer_t *flattenLayer, tensor_t *input, tensor_t *output,
                    const layerExec_t *exec) {
    (void)flattenLayer;
    requireMatchingBfpWires(input, output, "Flatten forward (input -> output)");

    size_t numberOfElements = calcNumberOfElementsByTensor(input);
    size_t numberOfBytes = calcNumberOfBytesForData(input->quantization, numberOfElements);
    memcpy(output->data, input->data, numberOfBytes);

    if (input->quantization->type == SYM_INT32) {
        symInt32QConfig_t *inputQC = input->quantization->qConfig;
        symInt32QConfig_t *outputQC = output->quantization->qConfig;
        storeCanonScale(&outputQC->scale, loadCanonScale(&inputQC->scale));
    } else if (input->quantization->type == BFP) {
        const bfpQConfig_t *inputQC = input->quantization->qConfig;
        bfpQConfig_t *outputQC = output->quantization->qConfig;
        memcpy(outputQC->exponents, inputQC->exponents, inputQC->numGroups);
    }
}

void flattenBackward(layer_t *flattenLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss, const layerExec_t *exec) {
    (void)flattenLayer;
    (void)forwardInput; /* never dereferenced -> not guarded */
    requireMatchingBfpWires(loss, propLoss, "Flatten backward (loss -> propLoss)");

    size_t numberOfElements = calcNumberOfElementsByTensor(loss);
    size_t numberOfBytes = calcNumberOfBytesForData(loss->quantization, numberOfElements);
    /* propLoss may be bound over loss's bytes (an in-place dx plan does so). A
     * copy onto itself is the identity, and memcpy's restrict parameters make
     * memcpy(p, p, n) undefined, so each copy is skipped on its own alias (the
     * Dropout verbatim-carry idiom). */
    if (propLoss->data != loss->data) {
        memcpy(propLoss->data, loss->data, numberOfBytes);
    }

    if (loss->quantization->type == SYM_INT32) {
        symInt32QConfig_t *lossQC = loss->quantization->qConfig;
        symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
        storeCanonScale(&propLossQC->scale, loadCanonScale(&lossQC->scale));
    } else if (loss->quantization->type == BFP) {
        const bfpQConfig_t *lossQC = loss->quantization->qConfig;
        bfpQConfig_t *propLossQC = propLoss->quantization->qConfig;
        if (propLossQC->exponents != lossQC->exponents) {
            memcpy(propLossQC->exponents, lossQC->exponents, lossQC->numGroups);
        }
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
