#define SOURCE_FILE "FLATTEN"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Flatten.h"

/* BFP epic PR2 Task 8: Flatten is a raw byte memcpy sized off the SOURCE wire's
 * quantization, plus a hand-copy of the SYM_INT32 scale. Two distinct BFP
 * failures: BFP -> BFP moves the mantissa payload but NOT the per-group
 * exponents, leaving the destination on its zero-state grid (every value
 * silently rescaled by a power of two); FLOAT32 -> BFP sizes the copy off the
 * 32-bit side and overruns the (much smaller) packed buffer. Keyed on the wire's
 * STORAGE dtype — Flatten has no arithmetic slot at all. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: BFP Flatten semantics arrive with epic PR4 -- keep BFP off this wire or "
                    "use FLOAT32 wires",
                    what);
        exit(1);
    }
}

void flattenForward(layer_t *flattenLayer, tensor_t *input, tensor_t *output) {
    (void)flattenLayer;
    requireNoBfpWire(input, "Flatten forward (input)");
    requireNoBfpWire(output, "Flatten forward (output)");

    size_t numberOfElements = calcNumberOfElementsByTensor(input);
    size_t numberOfBytes = calcNumberOfBytesForData(input->quantization, numberOfElements);
    memcpy(output->data, input->data, numberOfBytes);

    if (input->quantization->type == SYM_INT32) {
        symInt32QConfig_t *inputQC = input->quantization->qConfig;
        symInt32QConfig_t *outputQC = output->quantization->qConfig;
        outputQC->scale = inputQC->scale;
    }
}

void flattenBackward(layer_t *flattenLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss) {
    (void)flattenLayer;
    (void)forwardInput; /* never dereferenced -> not guarded */
    requireNoBfpWire(loss, "Flatten backward (loss)");
    requireNoBfpWire(propLoss, "Flatten backward (propLoss)");

    size_t numberOfElements = calcNumberOfElementsByTensor(loss);
    size_t numberOfBytes = calcNumberOfBytesForData(loss->quantization, numberOfElements);
    memcpy(propLoss->data, loss->data, numberOfBytes);

    if (loss->quantization->type == SYM_INT32) {
        symInt32QConfig_t *lossQC = loss->quantization->qConfig;
        symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
        propLossQC->scale = lossQC->scale;
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
