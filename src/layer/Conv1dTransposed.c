#define SOURCE_FILE "ODT_CONV1D_TRANSPOSED"

#include "Conv1dTransposed.h"

#include <math.h>
#include <string.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Conv1dKernel.h"
#include "ConvTranspose1dKernel.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Mul.h"
#include "Quantization.h"
#include "SlidingWindow1d.h"
#include "Tensor.h"

void initConv1dTransposedConfigWithWeightsAndBias(
    conv1dTransposedConfig_t *cfg, kernel_t *kernel, parameter_t *weights, parameter_t *bias,
    size_t groups, size_t outputPadding, quantization_t *forwardQ, quantization_t *weightGradQ,
    quantization_t *biasGradQ, quantization_t *propLossQ) {
    if (groups == 0) {
        PRINT_ERROR("Conv1dTransposed: groups must be >= 1");
        exit(1);
    }
    if (kernel->paddingType != VALID) {
        PRINT_ERROR("Conv1dTransposed: only VALID paddingType supported in Phase 1");
        exit(1);
    }
    if (outputPadding != 0 &&
        outputPadding >=
            ((kernel->stride > kernel->dilation) ? kernel->stride : kernel->dilation)) {
        PRINT_ERROR("Conv1dTransposed: outputPadding (%zu) must be < max(stride=%zu, "
                    "dilation=%zu)",
                    outputPadding, kernel->stride, kernel->dilation);
        exit(1);
    }
    if (kernel->size != weights->param->shape->dimensions[2]) {
        PRINT_ERROR("Conv1dTransposed: kernel->size (%zu) must equal weight kernelSize (%zu)",
                    kernel->size, weights->param->shape->dimensions[2]);
        exit(1);
    }
    cfg->kernel = kernel;
    cfg->weights = weights;
    cfg->bias = bias;
    cfg->groups = groups;
    cfg->outputPadding = outputPadding;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->weightGradMath = arithmeticFromQuantizationOrDefault(weightGradQ);
    cfg->biasGradMath = arithmeticFromQuantizationOrDefault(biasGradQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;

    /* Today's per-callsite hardcodes (conv1dTransposedCalcWeightGradsFloat32/
     * SymInt32, conv1dTransposedCalcBiasGradsFloat32/SymInt32 below), now
     * carried on the config so every caller of this init function --
     * factory-built or hand-wired directly
     * (test/unit/layer/UnitTestConv1dTransposed.c) -- gets the historical
     * behavior without having to know about the PR3 knob. A layerQuant_t-
     * driven factory overrides these right after this call
     * (Conv1dTransposedApi.c) if the caller opted into a different mode. */
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    cfg->frozen = false;
}

/* Group-quant PR3 (Task 2): forward's ctx must carry BOTH the layer's
 * kernel_t/groups/outputPadding geometry (conv1dTransposedConfig_t, needed
 * by every adapter below) AND -- only when the stored weight is grouped SYM
 * -- the ORIGINAL symQConfig_t carrying the real per-group scales.
 * executeOp's prologue hands the kernel only the UNPACKED, poisoned-scale
 * scratch (ops[1]), never the source tensor (ExecuteOp.c), so the group
 * shape must travel via ctx -- the identical wrapper Conv1d.c grew in PR2.
 * Set together with groupedSymOperandPos (conv1dTransposedForward, below)
 * -- both or neither, same invariant as Linear.c/Conv1d.c. PR3 (Task 3):
 * the dx (propLoss) adapters carry the SAME wrapper -- dx consumes the
 * weight AS STORED through the adjoint gather, whose group binding is to
 * flat storage, so the identical {cfg, weightGroups} pair serves both
 * directions. */
typedef struct convT1dForwardCtx {
    const conv1dTransposedConfig_t *cfg;
    const symQConfig_t *weightGroups; /* NULL unless the weight is grouped SYM or ASYM (then a STACK
                                         VIEW, see groupedWeightViewOrNull) */
} convT1dForwardCtx_t;

/* executeOp forward kernel adapters — operands are {input, weights} or
 * {input, weights, bias} (bias omitted, not NULL-padded, when the layer has
 * no bias). */
static void forwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                               const void *ctx) {
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    convTranspose1dKernelFloat32(ops[0], ops[1], bias, cfg->kernel, cfg->groups, cfg->outputPadding,
                                 rawOut);
}
static void forwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                             const void *ctx) {
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    if (fctx->weightGroups != NULL) {
        convTranspose1dKernelSymInt32Grouped(ops[0], ops[1], bias, cfg->kernel, cfg->groups,
                                             cfg->outputPadding, rawOut, fctx->weightGroups);
    } else {
        convTranspose1dKernelSymInt32(ops[0], ops[1], bias, cfg->kernel, cfg->groups,
                                      cfg->outputPadding, rawOut);
    }
}
/* BFP epic PR2 (Task 7): operands arrive in the funnel's unpacked-BFP
 * scratch form; routes to the output-centric GATHER kernel (D9 — the SYM
 * scatter cores stay SYM-only). All quant info comes from the operands, so
 * only kernel_t/groups/outputPadding geometry is taken from ctx —
 * ctx->weightGroups stays NULL (BFP weights never take the SYM carrier
 * path, groupedWeightViewOrNull returns NULL for the BFP dtype). */
static void forwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                             const void *ctx) {
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    convTranspose1dKernelBfpGather(ops[0], ops[1], bias, cfg->kernel, cfg->groups,
                                   cfg->outputPadding, rawOut);
}

/* Group-quant PR4 (Task 3): grouped-weight detection across BOTH grouped
 * carrier dtypes — duplicated verbatim from Linear.c (the canonical copy;
 * full view-lifetime/BORROWED-scales doc lives there). Grouped ASYM fills
 * *asymView (the CALLER's stack storage) as a symQConfig-shaped VIEW, valid
 * only for the frame's executeOp call, never stored, never freed. */
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
        *asymView = (symQConfig_t){.scales = qc->scales, /* BORROWED (see Linear.c) */
                                   .numGroups = qc->numGroups,
                                   .groupSize = qc->groupSize,
                                   .roundingMode = qc->roundingMode,
                                   .qBits = qc->qBits};
        return asymView;
    }
    return NULL;
}

void conv1dTransposedForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    /* Group-quant PR3 (Task 2, inverting PR2's ConvT deny-pin) + PR4
     * (grouped ASYM via the symQConfig-shaped view): a stored grouped weight
     * routes the SYM kernel adapter to the grouped SCATTER-core entry
     * (weightGroups carried via ctx) AND opts the funnel's prologue into
     * unpacking the grouped operand (groupedSymOperandPos), always together
     * -- mirrors conv1dForward's identical wiring (Conv1d.c) exactly.
     * Per-tensor SYM/ASYM (numGroups==1) and SYM_INT32 weights are
     * untouched. weightTensor is always inputs[1] (bias present or not) in
     * BOTH math arms -- the FLOAT32 arm must declare the SAME position as
     * the SYM arm (PR2 final-review Fix 3(b) lesson: a grouped weight
     * forwarded under FLOAT32 math dequantizes via the funnel's group-aware
     * convertTensor cell, gated on this field exactly like the SYM arm's
     * unpack). */
    symQConfig_t asymWeightView; /* lifetime: this frame (Linear.c view doc) */
    const symQConfig_t *weightGroups = groupedWeightViewOrNull(weightTensor, &asymWeightView);
    bool grouped = weightGroups != NULL;
    convT1dForwardCtx_t fctx = {.cfg = cfg, .weightGroups = weightGroups};

    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        executeOp(
            &(opSpec_t){
                .kernel = forwardKernelFloat,
                .ctx = &fctx,
                .inputs = biasTensor != NULL ? (tensor_t *[]){input, weightTensor, biasTensor}
                                             : (tensor_t *[]){input, weightTensor},
                .nInputs = biasTensor != NULL ? 3 : 2,
                .arithmetic = cfg->forwardMath,
                .mode = OUT_WRITE,
                .groupedSymOperandPos = grouped ? 2 : 0,
            },
            output);
        break;
    case ARITH_SYM_INT32:
        executeOp(
            &(opSpec_t){
                .kernel = forwardKernelSym,
                .ctx = &fctx,
                .inputs = biasTensor != NULL ? (tensor_t *[]){input, weightTensor, biasTensor}
                                             : (tensor_t *[]){input, weightTensor},
                .nInputs = biasTensor != NULL ? 3 : 2,
                .arithmetic = cfg->forwardMath,
                .mode = OUT_WRITE,
                .groupedSymOperandPos = grouped ? 2 : 0,
            },
            output);
        break;
    case ARITH_BFP: {
        /* BFP epic PR2 (Task 7) — identical layer-side rules in Linear.c
         * (canonical rule comment there, arm parity): 1. BFP-stored weights
         * required (the width source for every staged FLOAT32 operand);
         * 2. FLOAT32-stored input/bias get the per-tensor stack template at
         * the WEIGHTS' widths, BFP-stored operands NULL (borrowed);
         * 3. groupedSymOperandPos = 0 (SYM/ASYM-carrier detail — BFP
         * blocking is per-operand-legal). */
        if (weightTensor->quantization->type != BFP) {
            PRINT_ERROR("Conv1dTransposed: ARITH_BFP forward requires BFP-stored weights "
                        "(FLOAT32-init + requantizeTensorInPlace, see "
                        "docs/conventions/arithmetic-bfp.md); got dtype %d",
                        (int)weightTensor->quantization->type);
            exit(1);
        }
        const bfpQConfig_t *wQC = weightTensor->quantization->qConfig;
        /* Stack template: lifetime covers the executeOp call (same frame). */
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->forwardMath.roundingMode,
                              .mantissaBits = wQC->mantissaBits,
                              .exponentBits = wQC->exponentBits};
        executeOp(
            &(opSpec_t){
                .kernel = forwardKernelBfp,
                .ctx = &fctx,
                .inputs = biasTensor != NULL ? (tensor_t *[]){input, weightTensor, biasTensor}
                                             : (tensor_t *[]){input, weightTensor},
                .nInputs = biasTensor != NULL ? 3 : 2,
                .arithmetic = cfg->forwardMath,
                .mode = OUT_WRITE,
                .groupedSymOperandPos = 0,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL,
                             (biasTensor != NULL && biasTensor->quantization->type == FLOAT32)
                                 ? &stage
                                 : NULL},
            },
            output);
        break;
    }
    default:
        PRINT_ERROR("Conv1dTransposed forward: quantization type not implemented");
        exit(1);
    }
}

/* executeOp kernel adapters (ctx = conv1dTransposedConfig_t*, for
 * kernel_t/groups geometry — mirrors Conv1d.c). Weight-grad kernels `+=`
 * across many (b, inPos) iterations, so they memset rawOut first; bias-grad
 * kernels write each output-channel index exactly once. SYM weight-grad
 * sets the raw intermediate's scale itself (s_in*s_loss); SYM bias-grad
 * emits the raw per-channel sum at the loss scale, letting the
 * OUT_ACC_FIXED_SCALE epilogue's rescaleIntoAccumulatorScale (target
 * roundingMode, spec D4) do the rescale that used to happen inline here. */
static void weightGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx) {
    (void)n;
    (void)auxOut;
    const conv1dTransposedConfig_t *cfg = ctx;
    tensor_t *forwardInput = ops[0];
    tensor_t *lossGrad = ops[1];

    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];

    size_t expectedOutLen =
        convTranspose1dOutputLength(inputLength, cfg->kernel, cfg->outputPadding);
    if (expectedOutLen != outputLength) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outputLength (%zu) does "
                    "not match the transpose geometry from forwardInput (expected %zu)",
                    outputLength, expectedOutLen);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    // Conv1dTransposed weight shape [Cin, Cout/groups, K] -> Cout = dim[1] * groups.
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[1] * groups;
    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad batch (%zu) does not "
                    "match forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outChannels (%zu) does "
                    "not match weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }

    /* #420 G1: the weight-grad write below indexes the [Cin, Cout/groups, K]
     * grad intermediate by the INPUT's channel index -- a mismatch would be an
     * OOB write, not just a wrong result (the BFP twin's guard, ported). */
    if (inChannels != cfg->weights->param->shape->dimensions[0]) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): forwardInput inChannels (%zu) does "
                    "not match weight Cin (%zu)",
                    inChannels, cfg->weights->param->shape->dimensions[0]);
        exit(1);
    }

    long long outputLengthSigned = (long long)outputLength;
    long long dilation = (long long)cfg->kernel->dilation;
    size_t stride = cfg->kernel->stride;

    float const *xArr = (float const *)forwardInput->data;
    float const *gyArr = (float const *)lossGrad->data;
    float *gwArr = (float *)rawOut->data;
    memset(gwArr, 0,
           calcNumberOfBytesForData(rawOut->quantization, calcNumberOfElementsByTensor(rawOut)));

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    float xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    long long outBase = (long long)(inPos * stride); // VALID -> padLeft=0

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        for (size_t k = 0; k < kernelSize; k++) {
                            long long outIdx = outBase + (long long)k * dilation;
                            if (outIdx < 0 || outIdx >= outputLengthSigned) {
                                continue;
                            }
                            float gy =
                                gyArr[(b * outChannels + oc) * outputLength + (size_t)outIdx];
                            gwArr[(ic * outChPerGroup + ocOffset) * kernelSize + k] += xv * gy;
                        }
                    }
                }
            }
        }
    }
}

static void conv1dTransposedCalcWeightGradsFloat32(conv1dTransposedConfig_t *cfg,
                                                   tensor_t *forwardInput, tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->weightGradAccMode, "Conv1dTransposed weightGradAccMode");
    executeOp(
        &(opSpec_t){
            .kernel = weightGradKernelFloat,
            .ctx = cfg,
            .inputs = (tensor_t *[]){forwardInput, lossGrad},
            .nInputs = 2,
            .arithmetic = cfg->weightGradMath,
            .mode = cfg->weightGradAccMode,
        },
        cfg->weights->grad);
}

static void biasGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    const conv1dTransposedConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];

    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];
    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (biasGrad): lossGrad outChannels (%zu) does not "
                    "match bias Cout (%zu)",
                    outChannels, biasOutChannels);
        exit(1);
    }

    float const *gyArr = (float const *)lossGrad->data;
    float *rawArr = (float *)rawOut->data;

    for (size_t oc = 0; oc < outChannels; oc++) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                sum += gyArr[(b * outChannels + oc) * outputLength + outPos];
            }
        }
        rawArr[oc] = sum;
    }
}

static void conv1dTransposedCalcBiasGradsFloat32(conv1dTransposedConfig_t *cfg,
                                                 tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->biasGradAccMode, "Conv1dTransposed biasGradAccMode");
    executeOp(
        &(opSpec_t){
            .kernel = biasGradKernelFloat,
            .ctx = cfg,
            .inputs = (tensor_t *[]){lossGrad},
            .nInputs = 1,
            .arithmetic = cfg->biasGradMath,
            .mode = cfg->biasGradAccMode,
        },
        cfg->bias->grad);
}

static void weightGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    const conv1dTransposedConfig_t *cfg = ctx;
    tensor_t *forwardInput = ops[0];
    tensor_t *lossGrad = ops[1];

    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];

    size_t expectedOutLen =
        convTranspose1dOutputLength(inputLength, cfg->kernel, cfg->outputPadding);
    if (expectedOutLen != outputLength) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outputLength (%zu) does "
                    "not match the transpose geometry from forwardInput (expected %zu)",
                    outputLength, expectedOutLen);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    // Conv1dTransposed weight shape [Cin, Cout/groups, K] -> Cout = dim[1] * groups.
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[1] * groups;
    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad batch (%zu) does not "
                    "match forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outChannels (%zu) does "
                    "not match weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }

    /* #420 G1: the weight-grad write below indexes the [Cin, Cout/groups, K]
     * grad intermediate by the INPUT's channel index -- a mismatch would be an
     * OOB write, not just a wrong result (the BFP twin's guard, ported). */
    if (inChannels != cfg->weights->param->shape->dimensions[0]) {
        PRINT_ERROR(
            "Conv1dTransposed backward (SYM weightGrad): forwardInput inChannels (%zu) does "
            "not match weight Cin (%zu)",
            inChannels, cfg->weights->param->shape->dimensions[0]);
        exit(1);
    }

    float inScale = ((symInt32QConfig_t *)forwardInput->quantization->qConfig)->scale;
    float lossScale = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale;

    int32_t *interData = (int32_t *)rawOut->data;
    memset(interData, 0,
           calcNumberOfBytesForData(rawOut->quantization, calcNumberOfElementsByTensor(rawOut)));
    ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale = inScale * lossScale;

    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *gyArr = (int32_t const *)lossGrad->data;

    long long outputLengthSigned = (long long)outputLength;
    long long dilation = (long long)cfg->kernel->dilation;
    size_t stride = cfg->kernel->stride;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    int32_t xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    long long outBase = (long long)(inPos * stride); // VALID -> padLeft=0

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        for (size_t k = 0; k < kernelSize; k++) {
                            long long outIdx = outBase + (long long)k * dilation;
                            if (outIdx < 0 || outIdx >= outputLengthSigned) {
                                continue;
                            }
                            int32_t gy =
                                gyArr[(b * outChannels + oc) * outputLength + (size_t)outIdx];
                            interData[(ic * outChPerGroup + ocOffset) * kernelSize + k] +=
                                mulInt32s(xv, gy);
                        }
                    }
                }
            }
        }
    }
}

void conv1dTransposedCalcWeightGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *forwardInput,
                                             tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->weightGradAccMode, "Conv1dTransposed weightGradAccMode");
    executeOp(
        &(opSpec_t){
            .kernel = weightGradKernelSym,
            .ctx = cfg,
            .inputs = (tensor_t *[]){forwardInput, lossGrad},
            .nInputs = 2,
            .arithmetic = cfg->weightGradMath,
            .mode = cfg->weightGradAccMode,
        },
        cfg->weights->grad);
}

/* BFP epic PR3 (Task 4): OUTPUT-CENTRIC ConvT weight grad -- one reduction
 * per gw element (its contributors walked b outer / inPos inner, the goldgen
 * refs' normative order), restoring the int32 block-partial fold contract
 * the SYM/float twins' scatter-style `+=` accumulation cannot offer (see
 * Conv1d.c's weightGradKernelBfp). Unlike Conv1d's windowed walk, the
 * contributor map is the ConvT AFFINE outIdx = inPos*stride + k*dilation.
 * Each gw element is written exactly once, so there is NO memset. */
static void weightGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    const conv1dTransposedConfig_t *cfg = ctx;
    tensor_t *forwardInput = ops[0];
    tensor_t *lossGrad = ops[1];

    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];

    size_t expectedOutLen =
        convTranspose1dOutputLength(inputLength, cfg->kernel, cfg->outputPadding);
    if (expectedOutLen != outputLength) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outputLength (%zu) does "
                    "not match the transpose geometry from forwardInput (expected %zu)",
                    outputLength, expectedOutLen);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    // Conv1dTransposed weight shape [Cin, Cout/groups, K] -> Cout = dim[1] * groups.
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[1] * groups;
    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad batch (%zu) does not "
                    "match forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outChannels (%zu) does "
                    "not match weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }
    /* inChPerGroup below is derived from the INPUT's channel dim but indexes
     * the WEIGHT-sized grad intermediate ([Cin, Cout/groups, K]) -- a
     * mismatch would be an OOB write, not just a wrong result. */
    if (inChannels != cfg->weights->param->shape->dimensions[0]) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): forwardInput inChannels (%zu) does "
                    "not match weight Cin (%zu)",
                    inChannels, cfg->weights->param->shape->dimensions[0]);
        exit(1);
    }

    bfpQConfig_t *xQC = forwardInput->quantization->qConfig;
    bfpQConfig_t *gyQC = lossGrad->quantization->qConfig;
    validateBfpQConfigShape(xQC, calcNumberOfElementsByShape(forwardInput->shape));
    validateBfpQConfigShape(gyQC, calcNumberOfElementsByShape(lossGrad->shape));
    /* Reduction length per gw element: at most one contributor per
     * (b, inPos) pair. */
    bfpValidateBlockHeadroom(xQC, gyQC, batch * inputLength, "conv1dTransposedCalcWeightGradsBfp");

    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    float *gwArr = (float *)rawOut->data;

    int32_t xExpBias = bfpExponentBias(xQC);
    int32_t gyExpBias = bfpExponentBias(gyQC);

    size_t stride = cfg->kernel->stride;
    size_t dilation = cfg->kernel->dilation;

    for (size_t g = 0; g < groups; g++) {
        size_t inLo = g * inChPerGroup;
        size_t outLo = g * outChPerGroup;

        for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
            size_t ic = inLo + icOffset;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;

                for (size_t k = 0; k < kernelSize; k++) {
                    float acc = 0.0f;
                    int32_t partial = 0;
                    size_t currentXGroup = 0;
                    size_t currentGyGroup = SIZE_MAX;

                    for (size_t b = 0; b < batch; b++) {
                        for (size_t inPos = 0; inPos < inputLength; inPos++) {
                            size_t outIdx = inPos * stride + k * dilation;
                            /* Defensive clip, mirroring the float/SYM twins:
                             * under the outputLength guard above it is
                             * unreachable (max outIdx = outputLength -
                             * outputPadding - 1) -- it only bites if non-VALID
                             * padding ever shifts the affine base. */
                            if (outIdx >= outputLength) {
                                continue;
                            }
                            size_t xIdx = (b * inChannels + ic) * inputLength + inPos;
                            size_t gyIdx = (b * outChannels + oc) * outputLength + outIdx;
                            /* Per-element division on BOTH operands (the SYM
                             * grouped kernels' gap rationale): consecutive
                             * contributors hop by stride on the loss and by
                             * inputLength/outputLength across channels and
                             * batches. */
                            size_t xGroup = bfpGroupOf(xQC, xIdx);
                            size_t gyGroup = bfpGroupOf(gyQC, gyIdx);

                            if (currentGyGroup == SIZE_MAX) {
                                currentXGroup = xGroup;
                                currentGyGroup = gyGroup;
                            } else if (xGroup != currentXGroup || gyGroup != currentGyGroup) {
                                /* Boundary fold on EITHER operand's group
                                 * change: the finished same-exponent segment's
                                 * raw int32 partial enters the float
                                 * accumulator via a pure exponent shift --
                                 * rounding-free by contract. */
                                acc += ldexpf((float)partial,
                                              (int)xQC->exponents[currentXGroup] - xExpBias +
                                                  (int)gyQC->exponents[currentGyGroup] - gyExpBias);
                                partial = 0;
                                currentXGroup = xGroup;
                                currentGyGroup = gyGroup;
                            }

                            partial += mulInt32s(xArr[xIdx], gyArr[gyIdx]);
                        }
                    }
                    /* Tail fold, guarded on >= 1 visited contributor (with the
                     * clip dead under VALID, (b=0, inPos=0) always
                     * contributes, but the guard keeps the fold contract
                     * self-contained). */
                    if (currentGyGroup != SIZE_MAX) {
                        acc += ldexpf((float)partial,
                                      (int)xQC->exponents[currentXGroup] - xExpBias +
                                          (int)gyQC->exponents[currentGyGroup] - gyExpBias);
                    }

                    gwArr[(ic * outChPerGroup + ocOffset) * kernelSize + k] = acc;
                }
            }
        }
    }
}

void conv1dTransposedCalcWeightGradsBfp(conv1dTransposedConfig_t *cfg, tensor_t *forwardInput,
                                        tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->weightGradAccMode, "Conv1dTransposed weightGradAccMode");
    tensor_t *weightTensor = cfg->weights->param;
    /* Public-boundary re-guard of the conv1dTransposedBackward dispatch rule
     * (the BFP-weights authority there): a direct caller with FLOAT32
     * weights would otherwise read a bfpQConfig_t out of a float32QConfig
     * (garbage staging widths). */
    if (weightTensor->quantization->type != BFP) {
        PRINT_ERROR("conv1dTransposedCalcWeightGradsBfp: requires BFP-stored weights (the "
                    "width anchor for FLOAT32-operand staging); got dtype %d",
                    (int)weightTensor->quantization->type);
        exit(1);
    }
    const bfpQConfig_t *wQC = weightTensor->quantization->qConfig;
    /* Stack template at the weights' widths (lifetime: this executeOp call);
     * the funnel owns exponent backing and rounds by .arithmetic, not the
     * template. */
    bfpQConfig_t stage = {.exponents = NULL,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = cfg->weightGradMath.roundingMode,
                          .mantissaBits = wQC->mantissaBits,
                          .exponentBits = wQC->exponentBits};
    executeOp(
        &(opSpec_t){
            .kernel = weightGradKernelBfp,
            .ctx = cfg,
            .inputs = (tensor_t *[]){forwardInput, lossGrad},
            .nInputs = 2,
            .arithmetic = cfg->weightGradMath,
            .mode = cfg->weightGradAccMode,
            .bfpStage = {forwardInput->quantization->type == FLOAT32 ? &stage : NULL,
                         lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL},
        },
        cfg->weights->grad);
}

static void biasGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    const conv1dTransposedConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];

    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];
    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (biasGrad): lossGrad outChannels (%zu) does not "
                    "match bias Cout (%zu)",
                    outChannels, biasOutChannels);
        exit(1);
    }

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t *rawArr = (int32_t *)rawOut->data;
    float lossScale = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale;

    for (size_t oc = 0; oc < outChannels; oc++) {
        /* int32 accumulator (NO int64): loss mantissas are int12-range per the
         * qMaxBits=12 operand contract, so the batch*outputLength sum stays
         * well within int32. */
        int32_t sum = 0;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                sum += gyArr[(b * outChannels + oc) * outputLength + outPos];
            }
        }
        rawArr[oc] = sum;
    }
    ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale = lossScale;
}

void conv1dTransposedCalcBiasGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->biasGradAccMode, "Conv1dTransposed biasGradAccMode");
    executeOp(
        &(opSpec_t){
            .kernel = biasGradKernelSym,
            .ctx = cfg,
            .inputs = (tensor_t *[]){lossGrad},
            .nInputs = 1,
            .arithmetic = cfg->biasGradMath,
            .mode = cfg->biasGradAccMode,
        },
        cfg->bias->grad);
}

/* BFP epic PR3 (Task 4): per-oc batch*outputLength sum of BFP loss mantissas
 * -- int32 partial per same-group visited segment, lossless ldexpf fold on
 * group change + tail (Conv1d.c's biasGradKernelBfp on the identical
 * [B, C, L] loss walk, indices (b*outChannels + oc)*outputLength + outPos,
 * b outer / outPos inner -- this sum ALSO covers the outputPadding tail
 * positions the weightGrad affine never reaches). Sum headroom via
 * bfpValidateSumHeadroom (the product bound does not apply to sums). */
static void biasGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    const conv1dTransposedConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];

    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (biasGrad): lossGrad outChannels (%zu) does not "
                    "match bias Cout (%zu)",
                    outChannels, biasOutChannels);
        exit(1);
    }

    bfpQConfig_t *gyQC = lossGrad->quantization->qConfig;
    validateBfpQConfigShape(gyQC, calcNumberOfElementsByShape(lossGrad->shape));
    bfpValidateSumHeadroom(gyQC, batch * outputLength, "conv1dTransposedCalcBiasGradsBfp");

    int32_t expBias = bfpExponentBias(gyQC);
    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    float *out = (float *)rawOut->data;

    for (size_t oc = 0; oc < outChannels; oc++) {
        float acc = 0.0f;
        int32_t partial = 0;
        size_t currentGroup = 0;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                size_t idx = (b * outChannels + oc) * outputLength + outPos;
                size_t g = bfpGroupOf(gyQC, idx);
                if (b == 0 && outPos == 0) {
                    currentGroup = g;
                } else if (g != currentGroup) {
                    acc += ldexpf((float)partial, (int)gyQC->exponents[currentGroup] - expBias);
                    partial = 0;
                    currentGroup = g;
                }
                partial += gyArr[idx];
            }
        }
        if (batch > 0 && outputLength > 0) {
            acc += ldexpf((float)partial, (int)gyQC->exponents[currentGroup] - expBias);
        }
        out[oc] = acc;
    }
}

void conv1dTransposedCalcBiasGradsBfp(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->biasGradAccMode, "Conv1dTransposed biasGradAccMode");
    tensor_t *weightTensor = cfg->weights->param;
    /* Public-boundary re-guard of the conv1dTransposedBackward dispatch rule
     * (the BFP-weights authority there) -- see
     * conv1dTransposedCalcWeightGradsBfp. */
    if (weightTensor->quantization->type != BFP) {
        PRINT_ERROR("conv1dTransposedCalcBiasGradsBfp: requires BFP-stored weights (the width "
                    "anchor for FLOAT32-operand staging); got dtype %d",
                    (int)weightTensor->quantization->type);
        exit(1);
    }
    const bfpQConfig_t *wQC = weightTensor->quantization->qConfig;
    /* Stack template at the weights' widths (lifetime: this executeOp call);
     * the funnel owns exponent backing and rounds by .arithmetic, not the
     * template. */
    bfpQConfig_t stage = {.exponents = NULL,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = cfg->biasGradMath.roundingMode,
                          .mantissaBits = wQC->mantissaBits,
                          .exponentBits = wQC->exponentBits};
    executeOp(
        &(opSpec_t){
            .kernel = biasGradKernelBfp,
            .ctx = cfg,
            .inputs = (tensor_t *[]){lossGrad},
            .nInputs = 1,
            .arithmetic = cfg->biasGradMath,
            .mode = cfg->biasGradAccMode,
            .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
        },
        cfg->bias->grad);
}

/* dL/dx via the adjoint: conv1d-correlation of lossGrad with weight. The
 * kernel here uses VALID (Phase-1 contract); conv1dKernelFloat32/SymInt32
 * accept the same weight tensor (no flip needed, per spec §5.2).
 * PR3 (Task 3): ctx = convT1dForwardCtx_t* (the same wrapper as forward) --
 * the gather kernel's weight index arithmetic
 * ((oc*inChPerGroup + icOffset)*K + k, Conv1dKernel.c) computes exactly the
 * ConvT1d weight's [Cin, Cout/groups, K] flat storage index in this adjoint
 * role, so a grouped weight's storage-bound quantization groups apply
 * unchanged: route to the grouped GATHER entry with the stored weight's own
 * symQConfig_t. */
static void propLossKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    conv1dKernelFloat32(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, rawOut);
}

static void propLossKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    if (fctx->weightGroups != NULL) {
        conv1dKernelSymInt32Grouped(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, rawOut,
                                    fctx->weightGroups);
    } else {
        conv1dKernelSymInt32(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, rawOut);
    }
}

/* BFP epic PR3 (Task 4): dx via the output-centric GATHER adjoint (D9 fold
 * contract; the SYM cores stay SYM-only) -- conv1dKernelBfp reads the ConvT
 * weight's [Cin, Cout/groups, K] flat storage index unchanged in the adjoint
 * role (see the dx adapter block comment above). Same {cfg, weightGroups}
 * ctx wrapper as the other dx adapters, weightGroups always NULL for BFP
 * (groupedWeightViewOrNull returns NULL for the BFP dtype). */
static void propLossKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    conv1dKernelBfp(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, rawOut);
}

void conv1dTransposedBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                              tensor_t *propLoss) {
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    tensor_t *weightTensor = cfg->weights->param;

    /* BFP epic PR3, backward mirror of the forward's ARITH_BFP rule 1 (see
     * conv1dTransposedForward; Linear.c's linearBackward is the canonical
     * backward copy): BFP-stored weights are the width anchor every
     * FLOAT32-stored operand stages at. The stack template below serves the
     * propLoss executeOp call in THIS frame (the two grad WRAPPERS rebuild
     * it internally -- they are public, callable without passing here).
     * Zero-init keeps the template inert when no slot runs ARITH_BFP: the
     * .bfpStage ternary below wires &stage unconditionally on FLOAT32-stored
     * operands (ExecuteOp.h: entries are ignored under other arithmetics),
     * so &stage must never point at uninitialized stack. */
    bool anyBfpBackward = cfg->weightGradMath.type == ARITH_BFP ||
                          cfg->biasGradMath.type == ARITH_BFP ||
                          cfg->propLossMath.type == ARITH_BFP;
    bfpQConfig_t stage = {0}; /* lifetime: this frame, covers the propLoss executeOp call */
    if (anyBfpBackward) {
        if (weightTensor->quantization->type != BFP) {
            PRINT_ERROR("Conv1dTransposed backward: ARITH_BFP math slots require BFP-stored "
                        "weights (the width anchor for FLOAT32-operand staging; FLOAT32-init + "
                        "requantizeTensorInPlace, see docs/conventions/arithmetic-bfp.md); got "
                        "dtype %d",
                        (int)weightTensor->quantization->type);
            exit(1);
        }
        const bfpQConfig_t *wQC = weightTensor->quantization->qConfig;
        stage = (bfpQConfig_t){.exponents = NULL,
                               .numGroups = 1,
                               .groupSize = 0,
                               .roundingMode = cfg->propLossMath.roundingMode,
                               .mantissaBits = wQC->mantissaBits,
                               .exponentBits = wQC->exponentBits};
    }

    if (!cfg->frozen) {
        switch (cfg->weightGradMath.type) {
        case ARITH_FLOAT32:
            conv1dTransposedCalcWeightGradsFloat32(cfg, forwardInput, lossGrad);
            break;
        case ARITH_SYM_INT32:
            conv1dTransposedCalcWeightGradsSymInt32(cfg, forwardInput, lossGrad);
            break;
        case ARITH_BFP:
            conv1dTransposedCalcWeightGradsBfp(cfg, forwardInput, lossGrad);
            break;
        default:
            PRINT_ERROR(
                "Conv1dTransposed backward (weightGrad): quantization type not implemented");
            exit(1);
        }

        switch (cfg->biasGradMath.type) {
        case ARITH_FLOAT32:
            if (cfg->bias) {
                conv1dTransposedCalcBiasGradsFloat32(cfg, lossGrad);
            }
            break;
        case ARITH_SYM_INT32:
            if (cfg->bias) {
                conv1dTransposedCalcBiasGradsSymInt32(cfg, lossGrad);
            }
            break;
        case ARITH_BFP:
            if (cfg->bias) {
                conv1dTransposedCalcBiasGradsBfp(cfg, lossGrad);
            }
            break;
        default:
            PRINT_ERROR("Conv1dTransposed backward (biasGrad): quantization type not implemented");
            exit(1);
        }
    }

    /* propLoss (dx wire): OUT_WRITE. For a SYM_INT32 target this now requants
     * through the conversionMatrix diagonal (width-restored at the producer,
     * design D3) instead of the old direct kernel write of raw, unrestored
     * accumulator-range mantissas — the #187 dtype guard is superseded by the
     * funnel's own prologue/epilogue and is deleted (recon-conv-backward §4:
     * zero test coverage, confirmed tautology post-#221).
     * propLoss == NULL (#380 PR2): grads-only call -- skip the dx write
     * entirely rather than dereference the absent buffer. */
    if (propLoss != NULL) {
        /* Group-quant PR3 (Task 3) + PR4 (grouped ASYM via the view): same
         * detection + always-together wiring as conv1dTransposedForward (see
         * the comment there) -- ctx routes the SYM dx adapter to the grouped
         * GATHER entry, groupedSymOperandPos opts the funnel prologue into
         * unpacking (SYM arm) / group-aware dequant (FLOAT32 arm) of the
         * weight at inputs[1] (position 2), declared on BOTH math arms (PR2
         * final-review arm-parity lesson). */
        symQConfig_t asymWeightView; /* lifetime: this frame (Linear.c view doc) */
        const symQConfig_t *weightGroups = groupedWeightViewOrNull(weightTensor, &asymWeightView);
        bool grouped = weightGroups != NULL;
        convT1dForwardCtx_t fctx = {.cfg = cfg, .weightGroups = weightGroups};

        switch (cfg->propLossMath.type) {
        case ARITH_FLOAT32:
            executeOp(
                &(opSpec_t){
                    .kernel = propLossKernelFloat,
                    .ctx = &fctx,
                    .inputs = (tensor_t *[]){lossGrad, weightTensor},
                    .nInputs = 2,
                    .arithmetic = cfg->propLossMath,
                    .mode = OUT_WRITE,
                    .groupedSymOperandPos = grouped ? 2 : 0,
                },
                propLoss);
            break;
        case ARITH_SYM_INT32:
            executeOp(
                &(opSpec_t){
                    .kernel = propLossKernelSym,
                    .ctx = &fctx,
                    .inputs = (tensor_t *[]){lossGrad, weightTensor},
                    .nInputs = 2,
                    .arithmetic = cfg->propLossMath,
                    .mode = OUT_WRITE,
                    .groupedSymOperandPos = grouped ? 2 : 0,
                },
                propLoss);
            break;
        case ARITH_BFP:
            executeOp(
                &(opSpec_t){
                    .kernel = propLossKernelBfp,
                    .ctx = &fctx,
                    .inputs = (tensor_t *[]){lossGrad, weightTensor},
                    .nInputs = 2,
                    .arithmetic = cfg->propLossMath,
                    .mode = OUT_WRITE,
                    /* SYM/ASYM-carrier gate -- BFP blocking is
                     * per-operand-legal, nothing is declared. */
                    .groupedSymOperandPos = 0,
                    /* weights operand: always BFP-stored under ARITH_BFP
                     * (rule-1 mirror above) -> borrowed zero-copy, never
                     * staged. */
                    .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL,
                                 NULL},
                },
                propLoss);
            break;
        default:
            PRINT_ERROR("Conv1dTransposed backward (propLoss): quantization type not implemented");
            exit(1);
        }
    }
}

void conv1dTransposedCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1dTransposed expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    size_t batchSize = inputShape->dimensions[0];
    size_t inputLength = inputShape->dimensions[2];
    // Conv1dTransposed weight shape: [Cin, Cout/groups, K]
    // Cout = (Cout/groups) * groups
    size_t outChannelsPerGroup = cfg->weights->param->shape->dimensions[1];
    size_t outChannels = outChannelsPerGroup * cfg->groups;

    size_t outputLength = convTranspose1dOutputLength(inputLength, cfg->kernel, cfg->outputPadding);

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outChannels;
    outputShape->dimensions[2] = outputLength;
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
