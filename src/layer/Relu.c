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

/* BFP epic PR4 (R-P2): ReLU still runs OUTSIDE the executeOp funnel — each arm
 * raw-views ->data in ITS OWN storage format, so the guard stays keyed on the
 * wire's STORAGE dtype and is only NARROWED per arm: the FLOAT32/SYM_INT32
 * arms keep rejecting a packed BFP wire (their float* / int32_t* views would read
 * packed bytes as wide scalars and leave the destination's exponents stale),
 * while the ARITH_BFP arm requires both wires BFP-stored. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: this arm raw-views the wire in its own storage format and cannot read "
                    "packed BFP mantissas -- derive ARITH_BFP from a BFP wire config, or keep "
                    "BFP off this wire",
                    what);
        exit(1);
    }
}

static void requireBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type != BFP) {
        PRINT_ERROR("%s: the ARITH_BFP arm requires BFP-stored wires (packed codes + per-group "
                    "exponents, carried verbatim) -- got dtype %d; see "
                    "docs/conventions/arithmetic-bfp.md §5.7",
                    what, (int)t->quantization->type);
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

    switch (reluConfig->forwardMath.type) {
    case ARITH_FLOAT32:
        requireNoBfpWire(input, "ReLU forward (input)");
        requireNoBfpWire(output, "ReLU forward (output)");
        reluForwardFloat(input, output);
        break;
    case ARITH_SYM_INT32:
        requireNoBfpWire(input, "ReLU forward (input)");
        requireNoBfpWire(output, "ReLU forward (output)");
        reluForwardSymInt32(input, output);
        break;
    case ARITH_BFP:
        requireBfpWire(input, "ReLU forward (input)");
        requireBfpWire(output, "ReLU forward (output)");
        reluForwardBfp(input, output);
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

/* BFP epic PR4 (R-P2 backward): mask by the SIGN of forwardInput's packed
 * codes (exponents are unsigned scale factors, so code <= 0 iff value <= 0 —
 * exact parity with the FLOAT32/SYM arms' `input[i] <= 0`), copy the loss code
 * where kept, write code 0 where dropped, and carry loss's group exponents
 * verbatim onto propLoss. Same transparency argument as the forward: zeroing
 * codes only shrinks a block's absmax, so no re-derivation (= no second
 * quantization, D8) is needed. forwardInput's GEOMETRY is NOT gated — only its
 * dtype and element count matter, since it is read sign-only.
 *
 * ALL THREE wires are length-gated: this loop unpacks from forwardInput AND
 * loss and packs into propLoss, all sized off ONE count, so any wire shorter
 * than the anchor is an over-read or an over-write. A per-tensor {1, 0} grid
 * matches any length, so validateBfpQConfigShape alone would not catch it. */
void reluBackwardBfp(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);
    const bfpQConfig_t *inQC = forwardInput->quantization->qConfig;
    const bfpQConfig_t *lossQC = loss->quantization->qConfig;
    bfpQConfig_t *propLossQC = propLoss->quantization->qConfig;
    /* forwardInput is the ANCHOR, so its count comparison is trivially true —
     * the call is here for the GRID half (does inQC tile numberOfElements?),
     * kept in the same idiom rather than a bare validateBfpQConfigShape so
     * every wire in this function is admitted through one door. */
    bfpRequireElementCount(inQC, numberOfElements, numberOfElements,
                           "ReLU backward BFP (forwardInput grid)");
    bfpRequireElementCount(lossQC, calcNumberOfElementsByTensor(loss), numberOfElements,
                           "ReLU backward BFP (loss vs forwardInput)");
    bfpRequireSameGeometry(lossQC, numberOfElements, propLossQC,
                           calcNumberOfElementsByTensor(propLoss),
                           "ReLU backward BFP (loss -> propLoss)");

    int32_t inCodes[ODT_CONVERSION_CHUNK_ELEMS];
    int32_t lossCodes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < numberOfElements; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = numberOfElements - off < ODT_CONVERSION_CHUNK_ELEMS
                           ? numberOfElements - off
                           : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtend((const uint8_t *)forwardInput->data + off * inQC->mantissaBits / 8,
                         inQC->mantissaBits, 0, inCodes, count);
        unpackSignExtend((const uint8_t *)loss->data + off * lossQC->mantissaBits / 8,
                         lossQC->mantissaBits, 0, lossCodes, count);
        for (size_t i = 0; i < count; i++) {
            if (inCodes[i] <= 0) {
                lossCodes[i] = 0;
            }
        }
        byteConversion((uint8_t *)lossCodes, 32,
                       (uint8_t *)propLoss->data + off * propLossQC->mantissaBits / 8,
                       propLossQC->mantissaBits, count);
    }
    memcpy(propLossQC->exponents, lossQC->exponents, lossQC->numGroups);
}

void reluBackward(layer_t *reluLayer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    reluConfig_t *reluConfig = reluLayer->config->relu;

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
    case ARITH_BFP:
        /* R-P2: all three wires are read/written in the PACKED code domain, so
         * all three must be BFP-stored; the loss/propLoss GRID identity is
         * enforced inside reluBackwardBfp (forwardInput is read sign-only and
         * may carry any BFP grid). */
        if (forwardInput->quantization->type != BFP || loss->quantization->type != BFP ||
            propLoss->quantization->type != BFP) {
            PRINT_ERROR("ReLU backward: ARITH_BFP arm requires BFP wires — got forwardInput %d, "
                        "loss %d, propLoss %d",
                        (int)forwardInput->quantization->type, (int)loss->quantization->type,
                        (int)propLoss->quantization->type);
            exit(1);
        }
        reluBackwardBfp(forwardInput, loss, propLoss);
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
