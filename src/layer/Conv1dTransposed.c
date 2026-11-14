#define SOURCE_FILE "ODT_CONV1D_TRANSPOSED"

#include "Conv1dTransposed.h"

#include <math.h>
#include <string.h>

#include "BfpKernelSupport.h"
#include "CheckedSize.h"
#include "Common.h"
#include "Conv1dKernel.h"
#include "ConvTranspose1dKernel.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "LayerExec.h"
#include "LayerWireGuards.h"
#include "Mul.h"
#include "Quantization.h"
#include "ScaleCanon.h"
#include "SlidingWindow1d.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"
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
    cfg->ownsQuantizations = false;
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    cfg->frozen = false;
}

/* Carries the layer geometry plus, for a grouped SYM/ASYM weight, the stored
 * weight's group metadata, because executeOp's unpacked scratch does not retain
 * its per-group scales. Pair a non-NULL groupedWeight with
 * groupedSymOperandPos = 2, for forward and dx alike: dx reads the weight in
 * its stored flat layout, so the same groups apply. */
typedef struct convT1dForwardCtx {
    const conv1dTransposedConfig_t *cfg;
    const symGroupedWeight_t *groupedWeight; /* NULL unless the weight is grouped SYM or ASYM
                                                (then STACK storage of the frame, see
                                                symGroupedWeightOrNull) */
} convT1dForwardCtx_t;

/* Forward operands are {input, weights} or {input, weights, bias}; a missing
 * bias is omitted, not NULL-padded. */
static void forwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                               const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    convTranspose1dKernelFloat32(ops[0], ops[1], bias, cfg->kernel, cfg->groups, cfg->outputPadding,
                                 rawOut);
}
static void forwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                             const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    if (fctx->groupedWeight != NULL) {
        convTranspose1dKernelSymInt32Grouped(ops[0], ops[1], bias, cfg->kernel, cfg->groups,
                                             cfg->outputPadding, rawOut, fctx->groupedWeight, rs);
    } else {
        convTranspose1dKernelSymInt32(ops[0], ops[1], bias, cfg->kernel, cfg->groups,
                                      cfg->outputPadding, rawOut, rs);
    }
}
/* BFP operands arrive unpacked and carry their own block metadata, so only the
 * geometry comes from ctx; groupedWeight is always NULL for BFP weights. BFP
 * uses the gather kernel, which reduces per output cell to keep the
 * block-partial folding order. */
static void forwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                             const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    convTranspose1dKernelBfpGather(ops[0], ops[1], bias, cfg->kernel, cfg->groups,
                                   cfg->outputPadding, rawOut);
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    CONV1D_TRANSPOSED_OP_FORWARD = 0,
    CONV1D_TRANSPOSED_OP_WEIGHT_GRAD = 1,
    CONV1D_TRANSPOSED_OP_BIAS_GRAD = 2,
    CONV1D_TRANSPOSED_OP_PROP_LOSS = 3,
};

void conv1dTransposedForward(layer_t *layer, tensor_t *input, tensor_t *output,
                             const layerExec_t *exec) {
    rngKey_t opKey;
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    /* A grouped SYM/ASYM weight needs groupedSymOperandPos = 2 on the FLOAT32
     * arm too: executeOp's group-aware dequantization is gated on the same
     * field. The weight is always inputs[1], with or without bias. */
    symQConfig_t asymWeightView; /* lifetime: this frame (SymGroupedWeight.h view doc) */
    symGroupedWeight_t groupedWeightCtx;
    const symGroupedWeight_t *groupedWeight =
        symGroupedWeightOrNull(weightTensor, &asymWeightView, &groupedWeightCtx);
    bool grouped = groupedWeight != NULL;
    convT1dForwardCtx_t fctx = {.cfg = cfg, .groupedWeight = groupedWeight};

    size_t outDims[3];
    size_t outOrder[3];
    shape_t outExpected = {.dimensions = outDims, .orderOfDimensions = outOrder};
    conv1dTransposedCalcOutputShape(layer, input->shape, &outExpected);
    layerRequireDenseWire(output, &outExpected, "Conv1dTransposed forward", "output");

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
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, CONV1D_TRANSPOSED_OP_FORWARD, &opKey),
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
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, CONV1D_TRANSPOSED_OP_FORWARD, &opKey),
                .mode = OUT_WRITE,
                .groupedSymOperandPos = grouped ? 2 : 0,
            },
            output);
        break;
    case ARITH_BFP: {
        /* BFP-stored weights set the widths at which FLOAT32-stored input and
         * bias are staged; BFP-stored operands are borrowed unstaged. */
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
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, CONV1D_TRANSPOSED_OP_FORWARD, &opKey),
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

/* Gradient adapters take ctx = conv1dTransposedConfig_t* for the geometry.
 * FLOAT32/SYM weight gradients accumulate into uninitialized executeOp scratch,
 * so clear it first. SYM weight gradients set the scale s_in*s_loss; SYM bias
 * gradients keep the loss scale and leave the rescale to the
 * OUT_ACC_FIXED_SCALE epilogue. */
static void weightGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx, rngStream_t *rs) {
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

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    size_t dilation = cfg->kernel->dilation;
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
                    size_t kLo, kHi;
                    convTValidTapRange(inPos, stride, dilation, /* VALID: padLeft 0 */ 0,
                                       outputLength, kernelSize, &kLo, &kHi);
                    size_t outFirst = inPos * stride + kLo * dilation;

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        size_t outIdx = outFirst;
                        for (size_t k = kLo; k < kHi; k++, outIdx += dilation) {
                            float gy = gyArr[(b * outChannels + oc) * outputLength + outIdx];
                            gwArr[(ic * outChPerGroup + ocOffset) * kernelSize + k] += xv * gy;
                        }
                    }
                }
            }
        }
    }
}

static void conv1dTransposedCalcWeightGradsFloat32(conv1dTransposedConfig_t *cfg,
                                                   tensor_t *forwardInput, tensor_t *lossGrad,
                                                   const rngKey_t *key) {
    executeOp(
        &(opSpec_t){
            .kernel = weightGradKernelFloat,
            .ctx = cfg,
            .inputs = (tensor_t *[]){forwardInput, lossGrad},
            .nInputs = 2,
            .arithmetic = cfg->weightGradMath,
            .key = key,
            .mode = cfg->weightGradAccMode,
        },
        cfg->weights->grad);
}

static void biasGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];

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

static void conv1dTransposedCalcBiasGradsFloat32(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad,
                                                 const rngKey_t *key) {
    executeOp(
        &(opSpec_t){
            .kernel = biasGradKernelFloat,
            .ctx = cfg,
            .inputs = (tensor_t *[]){lossGrad},
            .nInputs = 1,
            .arithmetic = cfg->biasGradMath,
            .key = key,
            .mode = cfg->biasGradAccMode,
        },
        cfg->bias->grad);
}

/* Weight-grad scatter, instantiated twice by weightGradKernelSym with `checked`
 * a compile-time constant: false is a plain multiply-accumulate, true routes
 * every accumulate through the SYM accumulator guard (SymAccumGuard.h). */
SYM_ACCUM_ALWAYS_INLINE void conv1dTransposedWeightGradSymScatter(
    size_t batch, size_t groups, size_t inChPerGroup, size_t outChPerGroup, size_t inChannels,
    size_t outChannels, size_t inputLength, size_t outputLength, size_t kernelSize,
    const kernel_t *kernel, const int32_t *xArr, const int32_t *gyArr, int32_t *interData,
    const bool checked, const symAccumSite_t *site) {
    size_t dilation = kernel->dilation;
    size_t stride = kernel->stride;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    int32_t xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    size_t kLo, kHi;
                    convTValidTapRange(inPos, stride, dilation, /* VALID: padLeft 0 */ 0,
                                       outputLength, kernelSize, &kLo, &kHi);
                    size_t outFirst = inPos * stride + kLo * dilation;

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        size_t outIdx = outFirst;
                        for (size_t k = kLo; k < kHi; k++, outIdx += dilation) {
                            int32_t gy = gyArr[(b * outChannels + oc) * outputLength + outIdx];
                            size_t gwIdx = (ic * outChPerGroup + ocOffset) * kernelSize + k;
                            interData[gwIdx] = symMac(interData[gwIdx], xv, gy, checked, site);
                        }
                    }
                }
            }
        }
    }
}

#define CONVT_WG_PARAMS                                                                            \
    size_t batch, size_t groups, size_t inChPerGroup, size_t outChPerGroup, size_t inChannels,     \
        size_t outChannels, size_t inputLength, size_t outputLength, size_t kernelSize,            \
        const kernel_t *kernel, const int32_t *xArr, const int32_t *gyArr, int32_t *interData,     \
        const symAccumSite_t *site
#define CONVT_WG_ARGS(checked)                                                                     \
    batch, groups, inChPerGroup, outChPerGroup, inChannels, outChannels, inputLength,              \
        outputLength, kernelSize, kernel, xArr, gyArr, interData, checked, site

SYM_ACCUM_NOINLINE void conv1dTransposedWeightGradSymScatterSafe(CONVT_WG_PARAMS) {
    conv1dTransposedWeightGradSymScatter(CONVT_WG_ARGS(false));
}

SYM_ACCUM_NOINLINE void conv1dTransposedWeightGradSymScatterChecked(CONVT_WG_PARAMS) {
    conv1dTransposedWeightGradSymScatter(CONVT_WG_ARGS(true));
}

static void weightGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
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

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    float inScale =
        loadCanonScale(&((symInt32QConfig_t *)forwardInput->quantization->qConfig)->scale);
    float lossScale =
        loadCanonScale(&((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale);

    int32_t *interData = (int32_t *)rawOut->data;
    memset(interData, 0,
           calcNumberOfBytesForData(rawOut->quantization, calcNumberOfElementsByTensor(rawOut)));
    storeCanonScale(&((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale,
                    canonProductScale(inScale, lossScale, ODT_SCALE_NEAREST));

    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *gyArr = (int32_t const *)lossGrad->data;

    /* A fixed tap k reaches a weight-grad element from each (batch, inPos) at
     * most once, so N = batch * inputLength; no seed (interData starts zeroed). */
    uint8_t xBits = ((symInt32QConfig_t *)forwardInput->quantization->qConfig)->qMaxBits;
    uint8_t gyBits = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->qMaxBits;
    symAccumSite_t site =
        symAccumSiteInit("conv1dTransposedCalcWeightGradsSymInt32 (batch*inputLength)",
                         batch * inputLength, xBits, gyBits, 0);
    if (symAccumProvenSafe(&site)) {
        conv1dTransposedWeightGradSymScatterSafe(
            batch, groups, inChPerGroup, outChPerGroup, inChannels, outChannels, inputLength,
            outputLength, kernelSize, cfg->kernel, xArr, gyArr, interData, &site);
    } else {
        conv1dTransposedWeightGradSymScatterChecked(
            batch, groups, inChPerGroup, outChPerGroup, inChannels, outChannels, inputLength,
            outputLength, kernelSize, cfg->kernel, xArr, gyArr, interData, &site);
    }
}

void conv1dTransposedCalcWeightGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *forwardInput,
                                             tensor_t *lossGrad, const rngKey_t *key) {
    executeOp(
        &(opSpec_t){
            .kernel = weightGradKernelSym,
            .ctx = cfg,
            .inputs = (tensor_t *[]){forwardInput, lossGrad},
            .nInputs = 2,
            .arithmetic = cfg->weightGradMath,
            .key = key,
            .mode = cfg->weightGradAccMode,
        },
        cfg->weights->grad);
}

/* Reduce each weight cell separately to preserve BFP block-partial folding
 * order; write each cell once, so no clearing is needed. Contributors are
 * walked b outer / inPos inner, the reference order, along the affine map
 * outIdx = inPos*stride + k*dilation. */
static void weightGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
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

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    bfpQConfig_t *xQC = forwardInput->quantization->qConfig;
    bfpQConfig_t *gyQC = lossGrad->quantization->qConfig;
    /* Reduction length per gw element: at most one contributor per
     * (b, inPos) pair. */
    bfpValidateBlockHeadroom(xQC, gyQC, batch * inputLength, "conv1dTransposedCalcWeightGradsBfp");

    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    float *gwArr = (float *)rawOut->data;

    int32_t xExpBias = bfpExponentBias(xQC);
    int32_t gyExpBias = bfpExponentBias(gyQC);
    bfpGroupRange_t xRange = {0};
    bfpGroupRange_t gyRange = {0};

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
                            /* Unreachable under VALID padding once backward's
                             * loss gate has fixed outputLength (max outIdx =
                             * outputLength - outputPadding - 1); it only
                             * matters if non-VALID padding shifts the affine
                             * base. */
                            if (outIdx >= outputLength) {
                                continue;
                            }
                            size_t xIdx = (b * inChannels + ic) * inputLength + inPos;
                            size_t gyIdx = (b * outChannels + oc) * outputLength + outIdx;
                            /* Per-element group lookup on BOTH operands (the SYM
                             * grouped kernels' gap rationale): consecutive
                             * contributors hop by stride on the loss and by
                             * inputLength/outputLength across channels and
                             * batches. bfpGroupAt divides only when an index
                             * leaves the current group's range. */
                            size_t xGroup = bfpGroupAt(&xRange, xQC, xIdx);
                            size_t gyGroup = bfpGroupAt(&gyRange, gyQC, gyIdx);

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
                                        tensor_t *lossGrad, const rngKey_t *key) {
    tensor_t *weightTensor = cfg->weights->param;
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
            .key = key,
            .mode = cfg->weightGradAccMode,
            .bfpStage = {forwardInput->quantization->type == FLOAT32 ? &stage : NULL,
                         lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL},
        },
        cfg->weights->grad);
}

/* Per-channel sum of loss codes over batch * outputLength terms. The loop body
 * is instantiated twice (SymAccumGuard.h): unchanged when the loss wire's
 * declared width proves the sum fits int32, overflow-checked otherwise. */
SYM_ACCUM_ALWAYS_INLINE void biasGradSumLoopsSym(const int32_t *gyArr, int32_t *rawArr,
                                                 size_t batch, size_t outChannels,
                                                 size_t outputLength, const bool checked,
                                                 const symAccumSite_t *site) {
    for (size_t oc = 0; oc < outChannels; oc++) {
        int32_t sum = 0;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                sum = symAdd(sum, gyArr[(b * outChannels + oc) * outputLength + outPos], checked,
                             site);
            }
        }
        rawArr[oc] = sum;
    }
}

SYM_ACCUM_NOINLINE void biasGradSumLoopsSymSafe(const int32_t *gyArr, int32_t *rawArr, size_t batch,
                                                size_t outChannels, size_t outputLength,
                                                const symAccumSite_t *site) {
    biasGradSumLoopsSym(gyArr, rawArr, batch, outChannels, outputLength, false, site);
}

SYM_ACCUM_NOINLINE void biasGradSumLoopsSymChecked(const int32_t *gyArr, int32_t *rawArr,
                                                   size_t batch, size_t outChannels,
                                                   size_t outputLength,
                                                   const symAccumSite_t *site) {
    biasGradSumLoopsSym(gyArr, rawArr, batch, outChannels, outputLength, true, site);
}

static void biasGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t *rawArr = (int32_t *)rawOut->data;
    float lossScale =
        loadCanonScale(&((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale);

    /* int32 accumulator (NO int64). The loss wire's declared qMaxBits bounds each
     * code, so N * 2^(q-1) <= INT32_MAX proves the sum safe (16-bit wire:
     * N <= 65535); otherwise the checked twin aborts on the first real overflow.
     * checkedMulSize: a wrapped count must not select the unchecked loop. */
    size_t terms;
    if (!checkedMulSize(batch, outputLength, &terms)) {
        PRINT_ERROR("Conv1dTransposed biasGrad: batch %zu * outputLength %zu overflows size_t",
                    batch, outputLength);
        exit(1);
    }
    uint8_t lossBits = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->qMaxBits;
    symAccumSite_t site = symAccumSiteInitSum("Conv1dTransposed biasGrad", terms, lossBits);
    if (symAccumProvenSafe(&site)) {
        biasGradSumLoopsSymSafe(gyArr, rawArr, batch, outChannels, outputLength, &site);
    } else {
        biasGradSumLoopsSymChecked(gyArr, rawArr, batch, outChannels, outputLength, &site);
    }
    storeCanonScale(&((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale, lossScale);
}

void conv1dTransposedCalcBiasGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad,
                                           const rngKey_t *key) {
    executeOp(
        &(opSpec_t){
            .kernel = biasGradKernelSym,
            .ctx = cfg,
            .inputs = (tensor_t *[]){lossGrad},
            .nInputs = 1,
            .arithmetic = cfg->biasGradMath,
            .key = key,
            .mode = cfg->biasGradAccMode,
        },
        cfg->bias->grad);
}

/* Per-oc sum of BFP loss mantissas over b outer / outPos inner: an int32
 * partial per same-group segment, folded losslessly with ldexpf on each group
 * change and at the tail. The sum also covers the outputPadding tail positions
 * the weight-grad map never reaches. Sums need bfpValidateSumHeadroom, not the
 * product bound. */
static void biasGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];

    bfpQConfig_t *gyQC = lossGrad->quantization->qConfig;
    bfpValidateSumHeadroom(gyQC, batch * outputLength, "conv1dTransposedCalcBiasGradsBfp");

    int32_t expBias = bfpExponentBias(gyQC);
    bfpGroupRange_t gyRange = {0};
    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    float *out = (float *)rawOut->data;

    for (size_t oc = 0; oc < outChannels; oc++) {
        float acc = 0.0f;
        int32_t partial = 0;
        size_t currentGroup = 0;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                size_t idx = (b * outChannels + oc) * outputLength + outPos;
                size_t g = bfpGroupAt(&gyRange, gyQC, idx);
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

void conv1dTransposedCalcBiasGradsBfp(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad,
                                      const rngKey_t *key) {
    tensor_t *weightTensor = cfg->weights->param;
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
            .key = key,
            .mode = cfg->biasGradAccMode,
            .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
        },
        cfg->bias->grad);
}

/* dx adapters (ctx = convT1dForwardCtx_t*): dL/dx is the VALID conv1d
 * correlation of lossGrad with the unflipped weight. The gather index
 * (oc*inChPerGroup + icOffset)*K + k is exactly the ConvT weight's
 * [Cin, Cout/groups, K] flat storage index here, so a grouped weight's
 * quantization groups apply unchanged. */
static void propLossKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    conv1dKernelFloat32(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, rawOut);
}

static void propLossKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    if (fctx->groupedWeight != NULL) {
        conv1dKernelSymInt32Grouped(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, rawOut,
                                    fctx->groupedWeight, rs);
    } else {
        conv1dKernelSymInt32(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, rawOut, rs);
    }
}

/* groupedWeight is always NULL for BFP weights. */
static void propLossKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const convT1dForwardCtx_t *fctx = ctx;
    const conv1dTransposedConfig_t *cfg = fctx->cfg;
    conv1dKernelBfp(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, rawOut);
}

void conv1dTransposedBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                              tensor_t *propLoss, const layerExec_t *exec) {
    rngKey_t opKey;
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    tensor_t *weightTensor = cfg->weights->param;

    size_t lossDims[3];
    size_t lossOrder[3];
    shape_t lossExpected = {.dimensions = lossDims, .orderOfDimensions = lossOrder};
    conv1dTransposedCalcOutputShape(layer, forwardInput->shape, &lossExpected);
    layerRequireDenseWire(lossGrad, &lossExpected, "Conv1dTransposed backward", "loss");
    if (propLoss != NULL) {
        size_t inOrder[3] = {0, 1, 2};
        shape_t inExpected = {.numberOfDimensions = 3,
                              .dimensions = forwardInput->shape->dimensions,
                              .orderOfDimensions = inOrder};
        layerRequireDenseWire(propLoss, &inExpected, "Conv1dTransposed backward", "propLoss");
    }

    /* BFP-stored weights set the staging widths for FLOAT32-stored operands.
     * The stack staging template must outlive the propLoss executeOp; inactive
     * arithmetic slots ignore it. The gradient helpers build their own
     * template. */
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
            conv1dTransposedCalcWeightGradsFloat32(
                cfg, forwardInput, lossGrad,
                layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_WEIGHT_GRAD, &opKey));
            break;
        case ARITH_SYM_INT32:
            conv1dTransposedCalcWeightGradsSymInt32(
                cfg, forwardInput, lossGrad,
                layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_WEIGHT_GRAD, &opKey));
            break;
        case ARITH_BFP:
            conv1dTransposedCalcWeightGradsBfp(
                cfg, forwardInput, lossGrad,
                layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_WEIGHT_GRAD, &opKey));
            break;
        default:
            PRINT_ERROR(
                "Conv1dTransposed backward (weightGrad): quantization type not implemented");
            exit(1);
        }

        switch (cfg->biasGradMath.type) {
        case ARITH_FLOAT32:
            if (cfg->bias) {
                conv1dTransposedCalcBiasGradsFloat32(
                    cfg, lossGrad,
                    layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_BIAS_GRAD, &opKey));
            }
            break;
        case ARITH_SYM_INT32:
            if (cfg->bias) {
                conv1dTransposedCalcBiasGradsSymInt32(
                    cfg, lossGrad,
                    layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_BIAS_GRAD, &opKey));
            }
            break;
        case ARITH_BFP:
            if (cfg->bias) {
                conv1dTransposedCalcBiasGradsBfp(
                    cfg, lossGrad,
                    layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_BIAS_GRAD, &opKey));
            }
            break;
        default:
            PRINT_ERROR("Conv1dTransposed backward (biasGrad): quantization type not implemented");
            exit(1);
        }
    }

    /* NULL propLoss requests parameter gradients only. */
    if (propLoss != NULL) {
        /* Same grouped-weight wiring as conv1dTransposedForward, on every arm. */
        symQConfig_t asymWeightView; /* lifetime: this frame (SymGroupedWeight.h view doc) */
        symGroupedWeight_t groupedWeightCtx;
        const symGroupedWeight_t *groupedWeight =
            symGroupedWeightOrNull(weightTensor, &asymWeightView, &groupedWeightCtx);
        bool grouped = groupedWeight != NULL;
        convT1dForwardCtx_t fctx = {.cfg = cfg, .groupedWeight = groupedWeight};

        switch (cfg->propLossMath.type) {
        case ARITH_FLOAT32:
            executeOp(
                &(opSpec_t){
                    .kernel = propLossKernelFloat,
                    .ctx = &fctx,
                    .inputs = (tensor_t *[]){lossGrad, weightTensor},
                    .nInputs = 2,
                    .arithmetic = cfg->propLossMath,
                    .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_PROP_LOSS,
                                          &opKey),
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
                    .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_PROP_LOSS,
                                          &opKey),
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
                    .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_TRANSPOSED_OP_PROP_LOSS,
                                          &opKey),
                    .mode = OUT_WRITE,
                    .groupedSymOperandPos = 0,
                    /* The weights are BFP-stored (checked above), so they are
                     * borrowed, never staged. */
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
        PRINT_ERROR("Conv1dTransposed expects 3D input [batch, channel, length], got %zuD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    // Conv1dTransposed weight shape: [Cin, Cout/groups, K]
    size_t inChannels = inputShape->dimensions[1];
    size_t weightInChannels = cfg->weights->param->shape->dimensions[0];
    if (inChannels != weightInChannels) {
        PRINT_ERROR("Conv1dTransposed: input channels (%zu) must equal weight in-channels (%zu)",
                    inChannels, weightInChannels);
        exit(1);
    }
    size_t batchSize = inputShape->dimensions[0];
    size_t inputLength = inputShape->dimensions[2];
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
