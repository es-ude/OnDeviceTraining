#define SOURCE_FILE "LINEAR"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "Add.h"
#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
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

/* BFP epic PR2 (Task 7): all operands are the executeOp prologue's
 * unpacked-BFP scratch (int32 mantissa codes under a live bfpQConfig_t —
 * borrowed exponents for BFP-stored sources, funnel-staged for FLOAT32
 * ones). Same transpose dance as linearForwardSymInt32Grouped: BFP groups
 * bind to STORAGE order, which transposeTensor exposes zero-copy as the
 * physically-innermost axis for the reduction. matmulBfpTensors reads all
 * quant info from the operands themselves — no weightGroups ctx. */
void linearForwardBfp(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulBfpTensors(input, w, b, output);
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
static void linearForwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx) {
    (void)auxOut;
    (void)ctx;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    linearForwardBfp(ops[1], bias, ops[0], rawOut);
}

void linearForward(layer_t *linearLayer, tensor_t *input, tensor_t *output) {
    linearConfig_t *linearConfig = linearLayer->config->linear;

    tensor_t *weights = getParamFromParameter(linearConfig->weights);
    tensor_t *bias = linearConfig->bias != NULL ? getParamFromParameter(linearConfig->bias) : NULL;

    /* BFP epic PR2 (Task 7), ARITH_BFP arm — identical layer-side rules in
     * Conv1d.c/Conv1dTransposed.c (arm parity):
     *  1. BFP-stored weights REQUIRED (fail-fast below): the weight is the
     *     operand whose widths every FLOAT32 operand stages at, so a
     *     FLOAT32 weight has no width source (and would silently fake-quant).
     *  2. bfpStage wiring (plan Decision 1/2): FLOAT32-stored input/bias get
     *     the stack geometry TEMPLATE below — per-tensor {1,0} at the
     *     WEIGHTS' widths, rounded by the op (the funnel owns exponent
     *     backing and reads roundingMode from .arithmetic, not the
     *     template); BFP-stored operands get NULL (borrowed zero-copy).
     *     Weights are always NULL by rule 1.
     *  3. groupedSymOperandPos = 0: that gate is a SYM/ASYM-carrier detail —
     *     BFP blocking is per-operand-legal under ARITH_BFP (ExecuteOp.h),
     *     so nothing is declared. */
    if (linearConfig->forwardMath.type == ARITH_BFP) {
        if (weights->quantization->type != BFP) {
            PRINT_ERROR("Linear: ARITH_BFP forward requires BFP-stored weights (FLOAT32-init + "
                        "requantizeTensorInPlace, see docs/conventions/arithmetic-bfp.md); got "
                        "dtype %d",
                        (int)weights->quantization->type);
            exit(1);
        }
        const bfpQConfig_t *wQC = weights->quantization->qConfig;
        /* Stack template: lifetime covers the executeOp call (same frame). */
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = linearConfig->forwardMath.roundingMode,
                              .mantissaBits = wQC->mantissaBits,
                              .exponentBits = wQC->exponentBits};
        executeOp(
            &(opSpec_t){
                .kernel = linearForwardKernelBfp,
                .inputs = bias != NULL ? (tensor_t *[]){input, weights, bias}
                                       : (tensor_t *[]){input, weights},
                .nInputs = bias != NULL ? 3 : 2,
                .arithmetic = linearConfig->forwardMath,
                .mode = OUT_WRITE,
                .groupedSymOperandPos = 0,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL,
                             (bias != NULL && bias->quantization->type == FLOAT32) ? &stage : NULL},
            },
            output);
        return;
    }

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

/* BFP epic PR3: backward cores on the PR2 fold contract (D8 amendment) --
 * operands arrive as the funnel's unpacked-BFP scratch; matmulBfpTensors is
 * orientation-agnostic (per-element group lookup honors orderOfDimensions),
 * so weightGrad/propLoss are thin transpose-view wrappers. */
void linearCalcWeightGradsBfp(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads) {
    transposeTensor(loss, 0, 1);
    matmulBfpTensors(loss, forwardInput, NULL, weightGrads);
    transposeTensor(loss, 0, 1);
}

void linearCalcPropLossBfp(tensor_t *loss, tensor_t *weights, tensor_t *propLoss) {
    matmulBfpTensors(loss, weights, NULL, propLoss);
}

/* Per-feature batch sum of BFP loss mantissas: int32 partial per same-group
 * visited segment (the walk strides by numFeatures, so groups may change every
 * step), lossless ldexpf fold on group change + tail. Sum headroom via
 * bfpValidateSumHeadroom (the product bound does not apply to sums). */
void linearCalcBiasGradsBfp(tensor_t *loss, tensor_t *biasGrad) {
    if (loss->quantization->type != BFP) {
        PRINT_ERROR("linearCalcBiasGradsBfp: loss must be BFP (unpacked scratch form)");
        exit(1);
    }
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrad);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    if (numFeatures == 0 || numLoss % numFeatures != 0) {
        PRINT_ERROR("linearCalcBiasGradsBfp: loss elements %zu not divisible by features %zu",
                    numLoss, numFeatures);
        exit(1);
    }
    size_t batch = numLoss / numFeatures;
    const bfpQConfig_t *qC = loss->quantization->qConfig;
    validateBfpQConfigShape(qC, numLoss);
    bfpValidateSumHeadroom(qC, batch, "linearCalcBiasGradsBfp");
    int32_t expBias = bfpExponentBias(qC);
    const int32_t *codes = (const int32_t *)loss->data;
    float *out = (float *)biasGrad->data;
    for (size_t f = 0; f < numFeatures; f++) {
        float acc = 0.f;
        int32_t partial = 0;
        size_t currentGroup = 0;
        for (size_t n = 0; n < batch; n++) {
            size_t idx = n * numFeatures + f;
            size_t g = bfpGroupOf(qC, idx);
            if (n == 0) {
                currentGroup = g;
            } else if (g != currentGroup) {
                acc += ldexpf((float)partial, (int)qC->exponents[currentGroup] - expBias);
                partial = 0;
                currentGroup = g;
            }
            partial += codes[idx];
        }
        if (batch > 0) {
            acc += ldexpf((float)partial, (int)qC->exponents[currentGroup] - expBias);
        }
        out[f] = acc;
    }
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
static void weightGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsBfp(ops[0], ops[1], rawOut);
}
static void biasGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsBfp(ops[0], rawOut);
}
static void propLossKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcPropLossBfp(ops[0], ops[1], rawOut);
}

/* Backward kernel dispatch: native per-slot arms for FLOAT32/SYM_INT32/BFP
 * (epic PR3). Any future arithmetic enum member must still die HERE, not in
 * the funnel: for BFP-STORED operands the funnel's backward gate
 * (FLOAT32-stored + NULL bfpStage) never fires, and a fall-through ternary
 * would hand the FLOAT kernel unpacked int32 mantissa scratch through a
 * float* cast — silent wrong arithmetic rather than a crash (same hazard the
 * LayerNorm backward dispatch documents; Conv1d/ConvT1d guard their slots the
 * same way). */
static opKernelFn_t linearBackwardKernelForArithmetic(arithmetic_t math, opKernelFn_t floatKernel,
                                                      opKernelFn_t symKernel,
                                                      opKernelFn_t bfpKernel,
                                                      const char *slotName) {
    switch (math.type) {
    case ARITH_FLOAT32:
        return floatKernel;
    case ARITH_SYM_INT32:
        return symKernel;
    case ARITH_BFP:
        return bfpKernel;
    default:
        PRINT_ERROR("Linear backward (%s): quantization type not implemented", slotName);
        exit(1);
    }
}

void linearBackward(layer_t *linearLayer, tensor_t *forwardInput, tensor_t *loss,
                    tensor_t *propLoss) {
    linearConfig_t *cfg = linearLayer->config->linear;
    tensor_t *weights = getParamFromParameter(cfg->weights);

    /* BFP epic PR3, backward mirror of the forward's ARITH_BFP rules (see
     * linearForward): BFP-stored weights are the width anchor every
     * FLOAT32-stored operand stages at, and the stack template below covers
     * all three backward executeOp calls (per-tensor {1,0} at the weights'
     * widths; the funnel owns exponent backing and rounds by each op's own
     * arithmetic.roundingMode, not the template's). Zero-init keeps the
     * template inert when no slot runs ARITH_BFP: the .bfpStage ternaries
     * below wire &stage unconditionally on FLOAT32-stored operands
     * (ExecuteOp.h: entries are ignored under other arithmetics), so &stage
     * must never point at uninitialized stack. */
    bool anyBfpBackward = cfg->weightGradMath.type == ARITH_BFP ||
                          cfg->biasGradMath.type == ARITH_BFP ||
                          cfg->propLossMath.type == ARITH_BFP;
    bfpQConfig_t stage = {0}; /* lifetime: this frame, covers all three executeOp calls */
    if (anyBfpBackward) {
        if (weights->quantization->type != BFP) {
            PRINT_ERROR("Linear backward: ARITH_BFP math slots require BFP-stored weights (the "
                        "width anchor for FLOAT32-operand staging; FLOAT32-init + "
                        "requantizeTensorInPlace, see docs/conventions/arithmetic-bfp.md); got "
                        "dtype %d",
                        (int)weights->quantization->type);
            exit(1);
        }
        const bfpQConfig_t *wQC = weights->quantization->qConfig;
        stage = (bfpQConfig_t){.exponents = NULL,
                               .numGroups = 1,
                               .groupSize = 0,
                               .roundingMode = cfg->weightGradMath.roundingMode,
                               .mantissaBits = wQC->mantissaBits,
                               .exponentBits = wQC->exponentBits};
    }

    if (!cfg->frozen) {
        executeOpValidateAccMode(cfg->weightGradAccMode, "Linear weightGradAccMode");
        executeOp(
            &(opSpec_t){
                .kernel = linearBackwardKernelForArithmetic(
                    cfg->weightGradMath, weightGradKernelFloat, weightGradKernelSym,
                    weightGradKernelBfp, "weightGrad"),
                .inputs = (tensor_t *[]){loss, forwardInput},
                .nInputs = 2,
                .arithmetic = cfg->weightGradMath,
                .mode = cfg->weightGradAccMode,
                .bfpStage = {loss->quantization->type == FLOAT32 ? &stage : NULL,
                             forwardInput->quantization->type == FLOAT32 ? &stage : NULL, NULL},
            },
            getGradFromParameter(cfg->weights));

        if (cfg->bias != NULL) {
            executeOpValidateAccMode(cfg->biasGradAccMode, "Linear biasGradAccMode");
            executeOp(
                &(opSpec_t){
                    .kernel = linearBackwardKernelForArithmetic(
                        cfg->biasGradMath, biasGradKernelFloat, biasGradKernelSym,
                        biasGradKernelBfp, "biasGrad"),
                    .inputs = (tensor_t *[]){loss},
                    .nInputs = 1,
                    .arithmetic = cfg->biasGradMath,
                    .mode = cfg->biasGradAccMode,
                    .bfpStage = {loss->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
                },
                getGradFromParameter(cfg->bias));
        }
    }

    /* propLoss == NULL (#380 PR2): grads-only call -- skip the dx write
     * entirely rather than dereference the absent buffer. */
    if (propLoss != NULL) {
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
                .kernel = linearBackwardKernelForArithmetic(cfg->propLossMath, propLossKernelFloat,
                                                            propLossKernelSym, propLossKernelBfp,
                                                            "propLoss"),
                .ctx = weightGroups,
                .inputs = (tensor_t *[]){loss, weights},
                .nInputs = 2,
                .arithmetic = cfg->propLossMath,
                .mode = OUT_WRITE,
                .groupedSymOperandPos = grouped ? 2 : 0,
                /* weights operand: always BFP-stored under ARITH_BFP (rule-1
                 * mirror above) -> borrowed zero-copy, never staged. */
                .bfpStage = {loss->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
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
