#define SOURCE_FILE "QUANTIZATION_LAYER"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "LayerExec.h"
#include "QuantizationLayer.h"

/* The Quantization layer is a pure conversion node (#266): no arithmetic, no
 * kernel — executeConvert dispatches straight through the conversionMatrix
 * (same-dtype SYM_INT32 and BFP route through their diagonals: SYM_INT32 is
 * the sanctioned requant, BFP the sanctioned re-block/width-change point).
 * Any other same-dtype pair stays a config error: a Quantization layer that
 * neither changes dtype nor requantizes is a misconfigured no-op. The one
 * exception is the forward of a BFP input already in the output wire's
 * storage format, which passes straight through (quantizationForward). */
static void dispatchQuantization(tensor_t *input, tensor_t *output, rngStream_t *convert) {
    qtype_t inputDType = input->quantization->type;
    qtype_t outputDType = output->quantization->type;

    if (inputDType == outputDType && inputDType != SYM_INT32 && inputDType != BFP) {
        PRINT_ERROR("Quantization layer: same input/output dtype %d is only supported "
                    "for SYM_INT32 (requant) and BFP (re-block/width change); other "
                    "same-dtype pairs are a config error",
                    (int)inputDType);
        exit(1);
    }
    executeConvert(input, output, convert);
}

/* A BFP input already in the output wire's storage format (mantissa and
 * exponent widths, group shape): the input of a model whose dataset was
 * converted offline into its input wire's format. Nothing to convert -- the
 * BFP -> BFP cell would dequantize to float and re-derive every exponent. */
static bool bfpSameStorageFormat(const tensor_t *input, const tensor_t *output) {
    if (input->quantization->type != BFP || output->quantization->type != BFP) {
        return false;
    }
    const bfpQConfig_t *in = input->quantization->qConfig;
    const bfpQConfig_t *out = output->quantization->qConfig;
    return in->mantissaBits == out->mantissaBits && in->exponentBits == out->exponentBits &&
           in->numGroups == out->numGroups && in->groupSize == out->groupSize &&
           calcNumberOfElementsByTensor((tensor_t *)input) ==
               calcNumberOfElementsByTensor((tensor_t *)output);
}

/* Straight-through: payload and group exponents copied. Integer, no rounding
 * (so no draw, whatever the template's mode), byte-identical to what the
 * offline conversion wrote. */
static void passThroughBfp(const tensor_t *input, tensor_t *output) {
    const bfpQConfig_t *in = input->quantization->qConfig;
    bfpQConfig_t *out = output->quantization->qConfig;
    size_t n = calcNumberOfElementsByTensor((tensor_t *)input);
    if (n == 0) {
        return; /* N=0 tensor data may be NULL (#160) */
    }
    memcpy(output->data, input->data, calcNumberOfBytesForData(input->quantization, n));
    memcpy(out->exponents, in->exponents, in->numGroups);
}

/* Out of line (BFP_WALK_NOINLINE) so quantizationForward's own body holds
 * only the straight-through: the bfp-census checks the dispatcher as a
 * `self` root and declares this converting arm (FLOAT32 samples) a boundary
 * the pre-converted integer path never takes. */
BFP_WALK_NOINLINE void quantizationForwardConvert(tensor_t *inputTensor, tensor_t *outputTensor,
                                                  const layerExec_t *exec) {
    rngStream_t convertStorage;
    dispatchQuantization(
        inputTensor, outputTensor,
        layerExecPurposeStream(exec, RNG_TAG_FORWARD, RNG_TAG_CONVERT, &convertStorage));
}

void quantizationForward(layer_t *layer, tensor_t *inputTensor, tensor_t *outputTensor,
                         const layerExec_t *exec) {
    (void)layer; /* dtype dispatch reads the tensors; config->quantization->outputQ
                  * already determined outputTensor's dtype when its header was
                  * derived (the remat wire table's bind / initBufferOutput). */
    if (bfpSameStorageFormat(inputTensor, outputTensor)) {
        passThroughBfp(inputTensor, outputTensor);
        return;
    }
    quantizationForwardConvert(inputTensor, outputTensor, exec);
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    QUANTIZATION_OP_PROP_LOSS = 0,
};

/* The integer BFP -> BFP backward (arithmetic-bfp.md section 14): ops[0] =
 * the loss as unpacked-BFP scratch (int32 codes under its own group
 * exponents), rawOut = BFP_RAW on the propLoss geometry. Each code is a row
 * of its source group (one exponent per row), so the raw is
 * bfpRawEmitFromRows' data rule: per target group E_t = max over its nonzero
 * codes of E_src + bitlen(|code|) - 31, every code shifted onto 2^E_t
 * (left exact; right half away only for a code more than 31 binades below
 * the group's top). The funnel's re-block then rounds onto the propLoss grid
 * by the op's mode. No float. The row exponents are read straight from the
 * loss's biased group bytes (bfpRawEmitFromGroupExponents): no per-group
 * table, nothing on the stack beyond the funnel's raw, whatever the group
 * count. */
static void quantizationBackwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut,
                                          tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    (void)rs;
    tensor_t *loss = ops[0];
    size_t count = calcNumberOfElementsByTensor(loss);
    if (loss->quantization->type != BFP || rawOut->quantization->type != BFP_RAW ||
        calcNumberOfElementsByTensor(rawOut) != count) {
        PRINT_ERROR("Quantization backward BFP: needs a BFP loss and a BFP_RAW output of the same "
                    "element count, got dtypes %d -> %d",
                    (int)loss->quantization->type, (int)rawOut->quantization->type);
        exit(1);
    }
    const bfpQConfig_t *qC = loss->quantization->qConfig;
    validateBfpQConfigShape(qC, count);
    bfpRawQConfig_t *rawQC = rawOut->quantization->qConfig;
    bfpRawRequireOutput(rawQC, count, "Quantization backward BFP");
    if (count == 0) {
        return;
    }
    const int32_t expBias = bfpExponentBias(qC);
    size_t rowLen = qC->groupSize == 0 ? count : qC->groupSize;
    int32_t *mant = (int32_t *)rawOut->data;
    memcpy(mant, loss->data, count * sizeof(int32_t));
    bfpRawEmitFromGroupExponents(mant, count, qC->exponents, expBias, rowLen, rawQC,
                                 "Quantization backward BFP");
}

void quantizationBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss,
                          tensor_t *propLoss, const layerExec_t *exec) {
    (void)layer;
    (void)forwardInput; /* straight-through: dy does not depend on x. Both forward
                         * and backward requant map their own absmax exactly onto
                         * qMax, so neither pass saturates and no STE clipping mask
                         * is needed (#192 D3). */
    if (loss->quantization->type == BFP && propLoss->quantization->type == BFP) {
        /* BFP -> BFP: the integer re-block, rounded by the target template's
         * mode (the conversion cell's rule), its SR draws from this op's OUT
         * stream. Every other pair (FLOAT32 export, SYM, ...) converts below. */
        const bfpQConfig_t *target = propLoss->quantization->qConfig;
        rngKey_t opKey;
        executeOp(
            &(opSpec_t){
                .kernel = quantizationBackwardKernelBfp,
                .ctx = NULL,
                .inputs = (tensor_t *[]){loss},
                .nInputs = 1,
                .arithmetic = {.type = ARITH_BFP, .roundingMode = target->roundingMode},
                .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, QUANTIZATION_OP_PROP_LOSS, &opKey),
                .mode = OUT_WRITE,
                .bfpRawOut = BFP_RAW_BACKWARD_DX,
            },
            propLoss);
        return;
    }
    rngStream_t convertStorage;
    dispatchQuantization(
        loss, propLoss,
        layerExecPurposeStream(exec, RNG_TAG_BACKWARD, RNG_TAG_CONVERT, &convertStorage));
}

void quantizationCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    (void)layer;
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
