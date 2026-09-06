#define SOURCE_FILE "RELU"

#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "Comparison.h"
#include "DTypes.h"
#include "Layer.h"
#include "Quantization.h"
#include "Relu.h"
#include "Tensor.h"
#include "TensorConversion.h"

/* BFP epic PR2 Task 8: ReLU runs OUTSIDE the executeOp funnel — every arm below
 * raw-casts ->data (float* or int32_t*) and, for SYM_INT32, hand-copies the
 * scale. A BFP wire stores PACKED mantissa codes under a per-GROUP exponent, so
 * such a view reads packed bytes as wide scalars and leaves the destination's
 * exponents stale: silent corruption, never a crash. Keyed on the wire's STORAGE
 * dtype, not the declared arithmetic — a BFP wire reaches this layer under
 * ARITH_FLOAT32 (fake-quant, pinned) just as under ARITH_BFP (derived); both
 * are wrong for this layer until it grows real BFP semantics. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: BFP Relu semantics arrive with epic PR4 -- keep BFP off this wire or use "
                    "FLOAT32 wires",
                    what);
        exit(1);
    }
}

void reluForwardFloat(tensor_t *input, tensor_t *output) {
    gteFloatValue(input, 0, 0, output);
}

void reluForwardSymInt32(tensor_t *input, tensor_t *output) {
    symInt32QConfig_t *inputSymInt32QC = input->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = output->quantization->qConfig;
    gteSymInt32Zero(input, 0, output);
    outputSymInt32QC->scale = inputSymInt32QC->scale;
}

void reluForwardBfp(tensor_t *input, tensor_t *output) {
    /* Deliberately redundant with gteBfpZero's own gate: this one names the
     * LAYER in the error, which is what a model author can act on. Both are
     * pure predicates with no side effects, so the duplicate costs nothing. */
    bfpRequireSameGeometry(input->quantization->qConfig, calcNumberOfElementsByTensor(input),
                           output->quantization->qConfig, calcNumberOfElementsByTensor(output),
                           "ReLU forward BFP");
    gteBfpZero(input, output);
}

void reluForward(layer_t *reluLayer, tensor_t *input, tensor_t *output) {
    reluConfig_t *reluConfig = reluLayer->config->relu;
    requireNoBfpWire(input, "ReLU forward (input)");
    requireNoBfpWire(output, "ReLU forward (output)");

    switch (reluConfig->forwardMath.type) {
    case ARITH_FLOAT32:
        reluForwardFloat(input, output);
        break;
    case ARITH_SYM_INT32:
        reluForwardSymInt32(input, output);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

void reluBackwardFloat(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);

    float *inputArray = (float *)forwardInput->data;
    float *gradOutArray = (float *)loss->data;
    float *gradInArray = (float *)propLoss->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        if (inputArray[i] <= 0) {
            gradInArray[i] = 0;
        } else {
            gradInArray[i] = gradOutArray[i];
        }
    }
}

void reluBackwardSymInt32(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);

    int32_t *inputArray = (int32_t *)forwardInput->data;
    int32_t *gradOutputArray = (int32_t *)loss->data;
    int32_t *gradInputArray = (int32_t *)propLoss->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        if (inputArray[i] <= 0) {
            gradInputArray[i] = 0;
        } else {
            gradInputArray[i] = gradOutputArray[i];
        }
    }

    symInt32QConfig_t *lossQC = loss->quantization->qConfig;
    symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
    propLossQC->scale = lossQC->scale;
}

void reluBackward(layer_t *reluLayer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    reluConfig_t *reluConfig = reluLayer->config->relu;
    /* Ahead of the per-arm #315 guards below, which also reject a BFP wire but
     * only as "not the dtype this arm wants" — this says WHY and what to do. */
    requireNoBfpWire(forwardInput, "ReLU backward (forwardInput)");
    requireNoBfpWire(loss, "ReLU backward (loss)");
    requireNoBfpWire(propLoss, "ReLU backward (propLoss)");

    switch (reluConfig->propLossMath.type) {
    case ARITH_FLOAT32:
        /* Relu backward bypasses the executeOp funnel (scale-transparent) and
         * raw-casts the wire data pointers. The FLOAT32 arm reads/writes
         * forwardInput/loss/propLoss as float*; fed a SYM_INT32 wire it silently
         * reads int mantissa codes as floats — garbage grads propagated with no
         * diagnostic. Guard the actual wire dtypes and fail fast, mirroring the
         * LayerNorm/GroupNorm backward guards (#315, #261). */
        if (forwardInput->quantization->type != FLOAT32 || loss->quantization->type != FLOAT32 ||
            propLoss->quantization->type != FLOAT32) {
            PRINT_ERROR("ReLU backward: FLOAT32 arm requires FLOAT32 wires — got forwardInput %d, "
                        "loss %d, propLoss %d",
                        (int)forwardInput->quantization->type, (int)loss->quantization->type,
                        (int)propLoss->quantization->type);
            exit(1);
        }
        reluBackwardFloat(forwardInput, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        /* The SYM_INT32 arm raw-casts to int32* and derefs loss/propLoss->qConfig
         * as symInt32QConfig_t*; a FLOAT32 wire carries qConfig == NULL, so the
         * mismatch is a NULL deref rather than mere garbage — same fail-fast. */
        if (forwardInput->quantization->type != SYM_INT32 ||
            loss->quantization->type != SYM_INT32 || propLoss->quantization->type != SYM_INT32) {
            PRINT_ERROR("ReLU backward: SYM_INT32 arm requires SYM_INT32 wires — got forwardInput "
                        "%d, loss %d, propLoss %d",
                        (int)forwardInput->quantization->type, (int)loss->quantization->type,
                        (int)propLoss->quantization->type);
            exit(1);
        }
        reluBackwardSymInt32(forwardInput, loss, propLoss);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

void reluCalcOutputShape(layer_t *reluLayer, shape_t *inputShape, shape_t *outputShape) {
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
