#define SOURCE_FILE "LINEAR"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "Add.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Linear.h"
#include "Matmul.h"
#include "Rounding.h"
#include "TensorConversion.h"

void linearInitConfig(linearConfig_t *linearConfig, parameter_t *weights, parameter_t *bias,
                      quantization_t *forwardQ, quantization_t *backwardMath,
                      quantization_t *propLossQ) {
    linearConfig->weights = weights;
    linearConfig->bias = bias;
    linearConfig->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    linearConfig->weightGradMath = arithmeticFromQuantizationOrDefault(backwardMath);
    linearConfig->biasGradMath = arithmeticFromQuantizationOrDefault(backwardMath);
    linearConfig->propLossMath = arithmeticFromQuantizationOrDefault(backwardMath);
    linearConfig->outputQ = forwardQ;
    linearConfig->propLossQ = propLossQ;

    /* Today's per-callsite hardcodes (linearBackward, below), now carried on
     * the config so every caller of this init function -- factory-built or
     * hand-wired directly (test/unit/layer/UnitTestLinear.c) -- gets the
     * historical behavior without having to know about the PR3 knob. A
     * layerQuant_t-driven factory overrides these right after this call
     * (LinearApi.c) if the caller opted into a different mode. */
    linearConfig->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    linearConfig->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    linearConfig->frozen = false;
}

/* Group-quant PR4 (Task 3): grouped-weight detection across BOTH grouped
 * carrier dtypes (SYM and ASYM share the {numGroups, groupSize} shape
 * grammar, D6). Returns the symQConfig_t* to pass as the kernels'
 * weightGroups ctx iff the stored weight is grouped (numGroups > 1), else
 * NULL. Grouped SYM: the weight's OWN qConfig. Grouped ASYM: *asymView (the
 * CALLER's stack storage) is filled as a symQConfig-shaped VIEW of the asym
 * config — legitimate because the kernels read only
 * scales/numGroups/groupSize (plus the qBits operand-width validate), fields
 * both grammars share, and the funnel prologue has already shifted the codes
 * into the same signed-mantissa image the SYM arm produces (ExecuteOp.c —
 * D5: the grouped ASYM compute path IS the grouped SYM path on shifted
 * mantissas), so the zeroPoints never reach the kernel at all. VIEW
 * LIFETIME: scales is BORROWED from the asym config (never free through the
 * view) and the view lives in the caller's frame — valid for the duration of
 * the executeOp call it is passed into as ctx, never stored beyond it.
 * Duplicated verbatim in Conv1d.c / Conv1dTransposed.c (this copy is
 * canonical). */
static const symQConfig_t *groupedWeightViewOrNull(const tensor_t *weights,
                                                   symQConfig_t *asymView) {
    if (weights->quantization->type == SYM) {
        const symQConfig_t *qc = weights->quantization->qConfig;
        return qc->numGroups > 1 ? qc : NULL;
    }
    if (weights->quantization->type == ASYM) {
        const asymQConfig_t *qc = weights->quantization->qConfig;
        if (qc->numGroups <= 1) {
            return NULL;
        }
        *asymView = (symQConfig_t){.scales = qc->scales, /* BORROWED (see above) */
                                   .numGroups = qc->numGroups,
                                   .groupSize = qc->groupSize,
                                   .roundingMode = qc->roundingMode,
                                   .qBits = qc->qBits};
        return asymView;
    }
    return NULL;
}

void linearForwardFloat(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulFloat32TensorsWithBias(input, w, output, b);
    transposeTensor(w, 0, 1);
}

void linearForwardSymInt32(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulSymInt32TensorsWithBias(input, w, output, b);
    transposeTensor(w, 0, 1);
}

/* Group-quant PR2 (Task 3): `w` here is the executeOp prologue's grouped-SYM
 * scratch (unpacked mantissas, poisoned scale=1.0f — matmulSymInt32Tensors-
 * GroupedWeight ignores it, reading real per-group scales from
 * `weightGroups` instead). Same transpose dance as linearForwardSymInt32:
 * groups bind to STORAGE order, which transposeTensor exposes as the
 * physically-innermost (contiguous) axis for the reduction. */
void linearForwardSymInt32Grouped(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output,
                                  const symQConfig_t *weightGroups) {
    transposeTensor(w, 0, 1);
    matmulSymInt32TensorsGroupedWeight(input, w, b, output, weightGroups);
    transposeTensor(w, 0, 1);
}

/* executeOp forward kernel adapters — operands are {input, weights} or
 * {input, weights, bias} (bias omitted, not NULL-padded, when the layer has
 * no bias); ctx unused (matmul infers geometry from the tensors themselves),
 * EXCEPT linearForwardKernelSym: ctx carries the stored weight's own
 * symQConfig_t* (non-NULL) iff it is grouped SYM (linearForward sets ctx +
 * groupedSymOperandPos together, see below) — that routes to the grouped
 * matmul entry instead of the scalar one. */
static void linearForwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                     const void *ctx) {
    (void)auxOut;
    (void)ctx;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    linearForwardFloat(ops[1], bias, ops[0], rawOut);
}
static void linearForwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx) {
    (void)auxOut;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    const symQConfig_t *weightGroups = (const symQConfig_t *)ctx;
    if (weightGroups != NULL) {
        linearForwardSymInt32Grouped(ops[1], bias, ops[0], rawOut, weightGroups);
    } else {
        linearForwardSymInt32(ops[1], bias, ops[0], rawOut);
    }
}

void linearForward(layer_t *linearLayer, tensor_t *input, tensor_t *output) {
    linearConfig_t *linearConfig = linearLayer->config->linear;

    tensor_t *weights = getParamFromParameter(linearConfig->weights);
    tensor_t *bias = linearConfig->bias != NULL ? getParamFromParameter(linearConfig->bias) : NULL;

    /* Group-quant PR2 (+PR4: grouped ASYM via the symQConfig-shaped view,
     * see groupedWeightViewOrNull): a stored grouped weight routes the SYM
     * kernel adapter to the grouped matmul entry (weightGroups carried via
     * ctx) AND opts the funnel's prologue into unpacking the grouped operand
     * (groupedSymOperandPos) — always together, never independently (an
     * unpack without the routing has nowhere group-shaped to go; the
     * routing without the unpack would hand the kernel a still-packed
     * tensor). Per-tensor SYM/ASYM (numGroups==1) and SYM_INT32 weights are
     * untouched — ctx stays NULL, exactly like before this PR. weights is
     * always inputs[1] (bias present or not, see the .inputs literal below),
     * so the declared position is a constant 2 (i+1 for i=1). */
    symQConfig_t asymWeightView; /* lifetime: this frame (view doc above) */
    const symQConfig_t *weightGroups = groupedWeightViewOrNull(weights, &asymWeightView);
    bool grouped = weightGroups != NULL;

    executeOp(
        &(opSpec_t){
            .kernel = linearConfig->forwardMath.type == ARITH_SYM_INT32 ? linearForwardKernelSym
                                                                        : linearForwardKernelFloat,
            .ctx = weightGroups,
            .inputs = bias != NULL ? (tensor_t *[]){input, weights, bias}
                                   : (tensor_t *[]){input, weights},
            .nInputs = bias != NULL ? 3 : 2,
            .arithmetic = linearConfig->forwardMath,
            .mode = OUT_WRITE,
            .groupedSymOperandPos = grouped ? 2 : 0,
        },
        output);
}

void linearCalcWeightGradsFloat32(tensor_t *forwardInput, tensor_t *loss, tensor_t *weightGrads) {
    transposeTensor(loss, 0, 1);
    matmulFloat32Tensors(loss, forwardInput, weightGrads);
    transposeTensor(loss, 0, 1);
}

void linearCalcBiasGradsFloat32(tensor_t *loss, tensor_t *biasGrad) {
    /* Raw emit: per-feature batch sums of loss values; the executeOp epilogue adds. */
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrad);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    size_t batch = (numFeatures == 0) ? 0 : numLoss / numFeatures;
    float *bg = (float *)biasGrad->data;
    float *l = (float *)loss->data;
    for (size_t f = 0; f < numFeatures; f++) {
        float sum = 0.0f;
        for (size_t n = 0; n < batch; n++) {
            sum += l[n * numFeatures + f];
        }
        bg[f] = sum;
    }
}

void linearCalcPropLossFloat32(tensor_t *loss, tensor_t *weights, tensor_t *propLoss) {
    matmulFloat32Tensors(loss, weights, propLoss);
}

void linearCalcWeightGradsSymInt32(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads) {
    transposeTensor(loss, 1, 0);
    matmulSymInt32Tensors(loss, forwardInput, weightGrads);
    transposeTensor(loss, 1, 0);
}

void linearCalcBiasGradsSymInt32(tensor_t *biasGrads, tensor_t *loss) {
    /* Raw emit: per-feature batch sums of loss mantissas, at the LOSS scale.
     * The executeOp OUT_ACC_FIXED_SCALE epilogue rescales into the persistent
     * grad's existing scale and integer-adds (Deutel-adjacent ODT scheme). */
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrads);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    size_t batch = (numFeatures == 0) ? 0 : numLoss / numFeatures;
    int32_t *bg = (int32_t *)biasGrads->data;
    int32_t *l = (int32_t *)loss->data;
    for (size_t f = 0; f < numFeatures; f++) {
        /* int32 accumulator (NO int64 in SYM paths): loss mantissas are
         * int16-range per the qMaxBits<=16 contract, so the batch sum stays
         * within int32 for any batch <= 65536 - far beyond any real batch. */
        int32_t sum = 0;
        for (size_t n = 0; n < batch; n++) {
            sum += l[n * numFeatures + f];
        }
        bg[f] = sum;
    }
    ((symInt32QConfig_t *)biasGrads->quantization->qConfig)->scale =
        ((symInt32QConfig_t *)loss->quantization->qConfig)->scale;
}

void linearCalcPropLossSymInt32(tensor_t *weights, tensor_t *loss, tensor_t *propLoss) {
    matmulSymInt32Tensors(loss, weights, propLoss);
}

/* Group-quant PR3 (Task 1): unlike the forward's transpose dance
 * (linearForwardSymInt32Grouped), dx consumes the weight AS STORED — the
 * reduction runs over dim-0 (outFeatures), storage-strided by inFeatures.
 * The unified matmulIntCoreGrouped binds each visited element's group to its
 * actual storage index, so no orientation fix-up is needed (or possible:
 * groups partition flat storage, not the logical view). dx has no bias. */
void linearCalcPropLossSymInt32Grouped(tensor_t *weights, tensor_t *loss, tensor_t *propLoss,
                                       const symQConfig_t *weightGroups) {
    matmulSymInt32TensorsGroupedWeight(loss, weights, NULL, propLoss, weightGroups);
}

/* executeOp kernel adapters — ops convention: weight-grad {loss, fwdIn},
 * bias-grad {loss}, propLoss {loss, weightsParam}. auxOut unused; ctx unused
 * EXCEPT propLossKernelSym, which mirrors linearForwardKernelSym: ctx carries
 * the stored weight's own symQConfig_t* (non-NULL iff grouped SYM,
 * linearBackward sets ctx + groupedSymOperandPos together) and routes to the
 * grouped matmul entry. */
static void weightGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsFloat32(ops[1], ops[0], rawOut);
}
static void weightGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsSymInt32(ops[0], ops[1], rawOut);
}
static void biasGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsFloat32(ops[0], rawOut);
}
static void biasGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsSymInt32(rawOut, ops[0]);
}
static void propLossKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcPropLossFloat32(ops[0], ops[1], rawOut);
}
static void propLossKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    const symQConfig_t *weightGroups = (const symQConfig_t *)ctx;
    if (weightGroups != NULL) {
        linearCalcPropLossSymInt32Grouped(ops[1], ops[0], rawOut, weightGroups);
    } else {
        linearCalcPropLossSymInt32(ops[1], ops[0], rawOut);
    }
}

void linearBackward(layer_t *linearLayer, tensor_t *forwardInput, tensor_t *loss,
                    tensor_t *propLoss) {
    linearConfig_t *cfg = linearLayer->config->linear;

    if (!cfg->frozen) {
        executeOpValidateAccMode(cfg->weightGradAccMode, "Linear weightGradAccMode");
        executeOp(
            &(opSpec_t){
                .kernel = cfg->weightGradMath.type == ARITH_SYM_INT32 ? weightGradKernelSym
                                                                      : weightGradKernelFloat,
                .inputs = (tensor_t *[]){loss, forwardInput},
                .nInputs = 2,
                .arithmetic = cfg->weightGradMath,
                .mode = cfg->weightGradAccMode,
            },
            getGradFromParameter(cfg->weights));

        if (cfg->bias != NULL) {
            executeOpValidateAccMode(cfg->biasGradAccMode, "Linear biasGradAccMode");
            executeOp(
                &(opSpec_t){
                    .kernel = cfg->biasGradMath.type == ARITH_SYM_INT32 ? biasGradKernelSym
                                                                        : biasGradKernelFloat,
                    .inputs = (tensor_t *[]){loss},
                    .nInputs = 1,
                    .arithmetic = cfg->biasGradMath,
                    .mode = cfg->biasGradAccMode,
                },
                getGradFromParameter(cfg->bias));
        }
    }

    /* propLoss == NULL (#380 PR2): grads-only call -- skip the dx write
     * entirely rather than dereference the absent buffer. */
    if (propLoss != NULL) {
        tensor_t *weights = getParamFromParameter(cfg->weights);

        /* Group-quant PR3 (Task 1) + PR4 (grouped ASYM via the view): same
         * detection + always-together wiring as linearForward (see the
         * comment there) — ctx routes the SYM kernel adapter to the grouped
         * matmul entry, groupedSymOperandPos opts the funnel prologue into
         * unpacking (SYM arm) / group-aware dequant (FLOAT32 arm) of the
         * weight at inputs[1] (position 2). */
        symQConfig_t asymWeightView; /* lifetime: this frame (view doc above) */
        const symQConfig_t *weightGroups = groupedWeightViewOrNull(weights, &asymWeightView);
        bool grouped = weightGroups != NULL;

        executeOp(
            &(opSpec_t){
                .kernel = cfg->propLossMath.type == ARITH_SYM_INT32 ? propLossKernelSym
                                                                    : propLossKernelFloat,
                .ctx = weightGroups,
                .inputs = (tensor_t *[]){loss, weights},
                .nInputs = 2,
                .arithmetic = cfg->propLossMath,
                .mode = OUT_WRITE,
                .groupedSymOperandPos = grouped ? 2 : 0,
            },
            propLoss);
    }
}

void linearCalcOutputShape(layer_t *linearLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 2) {
        PRINT_ERROR("Linear layer expects 2D input, got %luD\n", inputShape->numberOfDimensions);
    }

    size_t batchSize = inputShape->dimensions[0];

    linearConfig_t *cfg = linearLayer->config->linear;
    shape_t *weightShape = cfg->weights->param->shape;
    size_t outFeatures = weightShape->dimensions[0];

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outFeatures;

    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
