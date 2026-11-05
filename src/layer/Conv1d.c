#define SOURCE_FILE "ODT_CONV1D"

#include "Conv1d.h"

#include <math.h>
#include <stdint.h>
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
#include "SlidingWindow1d.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"

void initConv1dConfigWithWeightsAndBias(conv1dConfig_t *conv1dConfig, kernel_t *kernel,
                                        parameter_t *weights, parameter_t *bias, size_t groups,
                                        quantization_t *forwardQ, quantization_t *weightGradQ,
                                        quantization_t *biasGradQ, quantization_t *propLossQ) {
    if (groups == 0) {
        PRINT_ERROR("Conv1d: groups must be >= 1");
        exit(1);
    }
    if (kernel->size != weights->param->shape->dimensions[2]) {
        PRINT_ERROR("Conv1d: kernel->size (%zu) must equal weight kernelSize (%zu)", kernel->size,
                    weights->param->shape->dimensions[2]);
        exit(1);
    }
    conv1dConfig->kernel = kernel;
    conv1dConfig->weights = weights;
    conv1dConfig->bias = bias;
    conv1dConfig->groups = groups;
    conv1dConfig->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    conv1dConfig->weightGradMath = arithmeticFromQuantizationOrDefault(weightGradQ);
    conv1dConfig->biasGradMath = arithmeticFromQuantizationOrDefault(biasGradQ);
    conv1dConfig->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    conv1dConfig->outputQ = forwardQ;
    conv1dConfig->propLossQ = propLossQ;
    conv1dConfig->ownsQuantizations = false;

    /* Today's per-callsite hardcodes (conv1dCalcWeightGradsFloat32/SymInt32,
     * conv1dCalcBiasGradsFloat32/SymInt32 below), now carried on the config so
     * every caller of this init function -- factory-built or hand-wired
     * directly (test/unit/layer/UnitTestConv1d.c) -- gets the historical
     * behavior without having to know about the PR3 knob. A layerQuant_t-
     * driven factory overrides these right after this call (Conv1dApi.c) if
     * the caller opted into a different mode. */
    conv1dConfig->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    conv1dConfig->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    conv1dConfig->frozen = false;
}

/* Group-quant PR2 (Task 4): forward's ctx must carry BOTH the layer's
 * kernel_t/groups geometry (conv1dConfig_t, needed by every adapter below)
 * AND -- only when the stored weight is grouped SYM or ASYM -- the
 * symGroupedWeight_t carrying the ORIGINAL symQConfig_t with the real
 * per-group scales plus the weight's code domain. executeOp's prologue
 * hands the kernel only the UNPACKED, poisoned-scale scratch (ops[1]), never
 * the source tensor (ExecuteOp.c), so the group shape must travel via ctx,
 * exactly like Linear's linearForwardKernelSym (which gets away with ctx ==
 * the symGroupedWeight_t alone because matmul infers geometry from the
 * tensors themselves; Conv1d's kernel additionally needs kernel_t/groups,
 * hence this wrapper). Set together with groupedSymOperandPos
 * (conv1dForward, below) -- both or neither, same invariant as Linear.c.
 * PR3 (Task 3): the dx (propLoss) adapters carry the SAME wrapper -- dx
 * consumes the weight AS STORED through the adjoint scatter, whose group
 * binding is to flat storage, so the identical {cfg, groupedWeight} pair
 * serves both directions. */
typedef struct conv1dForwardCtx {
    const conv1dConfig_t *cfg;
    const symGroupedWeight_t *groupedWeight; /* NULL unless the weight is grouped SYM or ASYM
                                                (then STACK storage of the frame, see
                                                symGroupedWeightOrNull) */
} conv1dForwardCtx_t;

/* executeOp forward kernel adapters — operands are {input, weights} or
 * {input, weights, bias} (bias omitted, not NULL-padded, when the layer has
 * no bias) — same convention Linear's forward adapters use (Linear.c). */
static void forwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                               const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    conv1dKernelFloat32(ops[0], ops[1], bias, cfg->kernel, cfg->groups, rawOut);
}
static void forwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                             const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    if (fctx->groupedWeight != NULL) {
        conv1dKernelSymInt32Grouped(ops[0], ops[1], bias, cfg->kernel, cfg->groups, rawOut,
                                    fctx->groupedWeight, rs);
    } else {
        conv1dKernelSymInt32(ops[0], ops[1], bias, cfg->kernel, cfg->groups, rawOut, rs);
    }
}
/* BFP epic PR2 (Task 7): operands arrive in the funnel's unpacked-BFP
 * scratch form; the BFP kernel reads all quant info from the operands
 * themselves, so only the kernel_t/groups geometry is taken from ctx —
 * ctx->groupedWeight stays NULL (BFP weights never take the SYM carrier
 * path, symGroupedWeightOrNull returns NULL for the BFP dtype).
 * Integer-only forward (arithmetic-bfp.md §13): rawOut is the BFP_RAW the
 * forward arm opts into (opSpec_t.bfpRawOut), folded in int32 by the
 * kernel; the OUT_WRITE epilogue re-blocks or dequantizes it. */
static void forwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    (void)rs; /* SR belongs to the re-block, never to the fold */
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    conv1dKernelBfpRaw(ops[0], ops[1], bias, cfg->kernel, cfg->groups, rawOut);
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    CONV1D_OP_FORWARD = 0,
    CONV1D_OP_WEIGHT_GRAD = 1,
    CONV1D_OP_BIAS_GRAD = 2,
    CONV1D_OP_PROP_LOSS = 3,
};

void conv1dForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec) {
    rngKey_t opKey;
    conv1dConfig_t *cfg = layer->config->conv1d;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    /* Group-quant PR2 (+PR4: grouped ASYM via the symQConfig-shaped view):
     * a stored grouped weight routes the SYM kernel adapter to the grouped
     * gather-core entry (groupedWeight carried via ctx) AND opts the funnel's
     * prologue into unpacking the grouped operand (groupedSymOperandPos),
     * always together -- mirrors linearForward's identical wiring (Linear.c)
     * exactly. Per-tensor SYM/ASYM (numGroups==1) and SYM_INT32 weights are
     * untouched. weightTensor is always inputs[1] (bias present or not, see
     * the .inputs literals below) in BOTH math arms: the FLOAT32 arm must declare the SAME position
     * as the SYM arm (a grouped weight forwarded under FLOAT32 math dequantizes via the funnel's
     * group-aware convertTensor cell -- gated on this field exactly like the SYM arm's unpack, not
     * a different mechanism) -- omitting it here would make Conv1d's FLOAT32-math grouped forward
     * regress once the funnel's FLOAT32 arm gate lands. */
    symQConfig_t asymWeightView; /* lifetime: this frame (SymGroupedWeight.h view doc) */
    symGroupedWeight_t groupedWeightCtx;
    const symGroupedWeight_t *groupedWeight =
        symGroupedWeightOrNull(weightTensor, &asymWeightView, &groupedWeightCtx);
    bool grouped = groupedWeight != NULL;
    conv1dForwardCtx_t fctx = {.cfg = cfg, .groupedWeight = groupedWeight};

    size_t outDims[3];
    size_t outOrder[3];
    shape_t outExpected = {.dimensions = outDims, .orderOfDimensions = outOrder};
    conv1dCalcOutputShape(layer, input->shape, &outExpected);
    layerRequireDenseWire(output, &outExpected, "Conv1d forward", "output");

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
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, CONV1D_OP_FORWARD, &opKey),
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
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, CONV1D_OP_FORWARD, &opKey),
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
            PRINT_ERROR("Conv1d: ARITH_BFP forward requires BFP-stored weights (FLOAT32-init + "
                        "requantizeTensorInPlace, see docs/conventions/arithmetic-bfp.md); got "
                        "dtype %d",
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
                .kernel = forwardKernelBfpRaw,
                .ctx = &fctx,
                .inputs = biasTensor != NULL ? (tensor_t *[]){input, weightTensor, biasTensor}
                                             : (tensor_t *[]){input, weightTensor},
                .nInputs = biasTensor != NULL ? 3 : 2,
                .arithmetic = cfg->forwardMath,
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, CONV1D_OP_FORWARD, &opKey),
                .mode = OUT_WRITE,
                .groupedSymOperandPos = 0,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL,
                             (biasTensor != NULL && biasTensor->quantization->type == FLOAT32)
                                 ? &stage
                                 : NULL},
                .bfpRawOut = BFP_RAW_FORWARD, /* the integer forward raw */
            },
            output);
        break;
    }
    default:
        PRINT_ERROR("Conv1d forward: quantization type not implemented");
        exit(1);
    }
}

/* executeOp kernel adapters (ctx = conv1dConfig_t*, for kernel_t/groups
 * geometry: the fixed opKernelFn_t shape has no
 * per-op-instance geometry slot other than ctx). The FLOAT32/SYM weight-grad
 * kernels (and their BFP twin, on int32 accumulators) `+=` into the same
 * weight cell across many (b, outPos) iterations, so they memset rawOut first
 * (the executeOp Phase-2 scratch is an uninitialized VLA, unlike the
 * reserveMemory-backed intermediate they replace). Bias-grad kernels write each
 * output-channel index exactly once (no
 * zero-init hazard). SYM weight-grad sets the raw intermediate's scale
 * itself (s_in*s_loss); SYM bias-grad emits the raw per-channel sum at the
 * loss scale and lets the OUT_ACC_FIXED_SCALE epilogue's
 * rescaleIntoAccumulatorScale (target roundingMode, PR1b.2 D4) do the rescale
 * that used to happen inline here. */
static void weightGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const conv1dConfig_t *cfg = ctx;
    tensor_t *forwardInput = ops[0];
    tensor_t *lossGrad = ops[1];

    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[0];

    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1d backward (weightGrad): lossGrad batch (%zu) does not match "
                    "forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1d backward (weightGrad): lossGrad outChannels (%zu) does not match "
                    "weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }

    /* #420 G1: inChPerGroup below is derived from the INPUT's channel dim but
     * indexes the WEIGHT-sized grad intermediate -- a mismatch would be an OOB
     * write, not just a wrong result (the BFP twin's guard, ported). */
    size_t weightInChannels = cfg->weights->param->shape->dimensions[1] * cfg->groups;
    if (inChannels != weightInChannels) {
        PRINT_ERROR("Conv1d backward (weightGrad): forwardInput inChannels (%zu) does not match "
                    "weight Cin-per-group * groups (%zu)",
                    inChannels, weightInChannels);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("Conv1d backward: lossGrad outputLength (%zu) does not match "
                    "geometry derived from forwardInput (%zu)",
                    outputLength, geom.outputLength);
        exit(1);
    }

    float const *xArr = (float const *)forwardInput->data;
    float const *gyArr = (float const *)lossGrad->data;
    float *gwArr = (float *)rawOut->data;
    memset(gwArr, 0,
           calcNumberOfBytesForData(rawOut->quantization, calcNumberOfElementsByTensor(rawOut)));

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    float gy = gyArr[(b * outChannels + oc) * outputLength + outPos];

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;

                            float xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            gwArr[(oc * inChPerGroup + icOffset) * kernelSize + kernelIdx] +=
                                xv * gy;
                        }
                    }
                }
            }
        }
    }
}

static void conv1dCalcWeightGradsFloat32(conv1dConfig_t *cfg, tensor_t *forwardInput,
                                         tensor_t *lossGrad, const rngKey_t *key) {
    executeOpValidateAccMode(cfg->weightGradAccMode, "Conv1d weightGradAccMode");
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
    const conv1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];

    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1d backward (biasGrad): lossGrad outChannels (%zu) does not match "
                    "bias Cout (%zu)",
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

static void conv1dCalcBiasGradsFloat32(conv1dConfig_t *cfg, tensor_t *lossGrad,
                                       const rngKey_t *key) {
    executeOpValidateAccMode(cfg->biasGradAccMode, "Conv1d biasGradAccMode");
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
 * a compile-time constant: false is the historical `+= mulInt32s`, true routes
 * every accumulate through the SYM accumulator guard (SymAccumGuard.h). */
SYM_ACCUM_ALWAYS_INLINE void
conv1dWeightGradSymScatter(size_t batch, size_t groups, size_t inChPerGroup, size_t outChPerGroup,
                           size_t inChannels, size_t outChannels, size_t inputLength,
                           size_t outputLength, size_t kernelSize, const windowGeometry1d_t *geomIn,
                           const int32_t *xArr, const int32_t *gyArr, int32_t *interData,
                           const bool checked, const symAccumSite_t *site) {
    const windowGeometry1d_t geom = *geomIn;
    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    int32_t gy = gyArr[(b * outChannels + oc) * outputLength + outPos];

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;

                            int32_t xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            size_t gwIdx = (oc * inChPerGroup + icOffset) * kernelSize + kernelIdx;
                            interData[gwIdx] = symMac(interData[gwIdx], xv, gy, checked, site);
                        }
                    }
                }
            }
        }
    }
}

#define CONV1D_WG_PARAMS                                                                           \
    size_t batch, size_t groups, size_t inChPerGroup, size_t outChPerGroup, size_t inChannels,     \
        size_t outChannels, size_t inputLength, size_t outputLength, size_t kernelSize,            \
        const windowGeometry1d_t *geomIn, const int32_t *xArr, const int32_t *gyArr,               \
        int32_t *interData, const symAccumSite_t *site
#define CONV1D_WG_ARGS(checked)                                                                    \
    batch, groups, inChPerGroup, outChPerGroup, inChannels, outChannels, inputLength,              \
        outputLength, kernelSize, geomIn, xArr, gyArr, interData, checked, site

SYM_ACCUM_NOINLINE void conv1dWeightGradSymScatterSafe(CONV1D_WG_PARAMS) {
    conv1dWeightGradSymScatter(CONV1D_WG_ARGS(false));
}

SYM_ACCUM_NOINLINE void conv1dWeightGradSymScatterChecked(CONV1D_WG_PARAMS) {
    conv1dWeightGradSymScatter(CONV1D_WG_ARGS(true));
}

static void weightGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const conv1dConfig_t *cfg = ctx;
    tensor_t *forwardInput = ops[0];
    tensor_t *lossGrad = ops[1];

    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[0];

    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1d backward (SYM weightGrad): lossGrad batch (%zu) does not match "
                    "forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1d backward (SYM weightGrad): lossGrad outChannels (%zu) does not match "
                    "weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }

    /* #420 G1: inChPerGroup below is derived from the INPUT's channel dim but
     * indexes the WEIGHT-sized grad intermediate -- a mismatch would be an OOB
     * write, not just a wrong result (the BFP twin's guard, ported). */
    size_t weightInChannels = cfg->weights->param->shape->dimensions[1] * cfg->groups;
    if (inChannels != weightInChannels) {
        PRINT_ERROR(
            "Conv1d backward (SYM weightGrad): forwardInput inChannels (%zu) does not match "
            "weight Cin-per-group * groups (%zu)",
            inChannels, weightInChannels);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("Conv1d backward (SYM weightGrad): lossGrad outputLength (%zu) does not "
                    "match geometry derived from forwardInput (%zu)",
                    outputLength, geom.outputLength);
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

    /* Each weight-grad element receives one product per (batch, outPos) pair,
     * so N = batch * outputLength; no seed (interData starts zeroed). */
    uint8_t xBits = ((symInt32QConfig_t *)forwardInput->quantization->qConfig)->qMaxBits;
    uint8_t gyBits = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->qMaxBits;
    symAccumSite_t site = symAccumSiteInit("conv1dCalcWeightGradsSymInt32 (batch*outputLength)",
                                           batch * outputLength, xBits, gyBits, 0);
    if (symAccumProvenSafe(&site)) {
        conv1dWeightGradSymScatterSafe(batch, groups, inChPerGroup, outChPerGroup, inChannels,
                                       outChannels, inputLength, outputLength, kernelSize, &geom,
                                       xArr, gyArr, interData, &site);
    } else {
        conv1dWeightGradSymScatterChecked(batch, groups, inChPerGroup, outChPerGroup, inChannels,
                                          outChannels, inputLength, outputLength, kernelSize, &geom,
                                          xArr, gyArr, interData, &site);
    }
}

void conv1dCalcWeightGradsSymInt32(conv1dConfig_t *cfg, tensor_t *forwardInput, tensor_t *lossGrad,
                                   const rngKey_t *key) {
    executeOpValidateAccMode(cfg->weightGradAccMode, "Conv1d weightGradAccMode");
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

/* weightGradKernelFloat's loop with an int32 raw accumulator per gw element
 * (acc, gwElems entries). Every product x * gy enters as product *
 * 2^-(E_acc - Ex - Egy) (bfpAlignPartial: exact left shift or half-away right
 * shift), E_acc fixed per output channel before the walk (eAccOf, one entry
 * per output channel): max over the batch of the input rows' max exponent
 * (the channel's conv group) + the loss row's max exponent, plus the guard
 * of bfpFoldGuardBits over N = batch * (most output positions one tap
 * reaches). So no accumulator can leave int32; every product is its own
 * one-product segment. Shared by both weightGrad kernels below, which
 * differ only in what they make of the finished sums. */
static void weightGradBfpScatter(const conv1dConfig_t *cfg, tensor_t *forwardInput,
                                 tensor_t *lossGrad, int32_t *acc, size_t gwElems, int *eAccOf) {
    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[0];

    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1d backward (BFP weightGrad): lossGrad batch (%zu) does not match "
                    "forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1d backward (BFP weightGrad): lossGrad outChannels (%zu) does not match "
                    "weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }

    /* inChPerGroup below is derived from the INPUT's channel dim but indexes
     * the WEIGHT-sized grad intermediate -- a mismatch would be an OOB
     * write, not just a wrong result. */
    size_t weightInChannels = cfg->weights->param->shape->dimensions[1] * cfg->groups;
    if (inChannels != weightInChannels) {
        PRINT_ERROR(
            "Conv1d backward (BFP weightGrad): forwardInput inChannels (%zu) does not match "
            "weight Cin-per-group * groups (%zu)",
            inChannels, weightInChannels);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("Conv1d backward (BFP weightGrad): lossGrad outputLength (%zu) does not "
                    "match geometry derived from forwardInput (%zu)",
                    outputLength, geom.outputLength);
        exit(1);
    }

    bfpQConfig_t *xQC = forwardInput->quantization->qConfig;
    bfpQConfig_t *gyQC = lossGrad->quantization->qConfig;
    validateBfpQConfigShape(xQC, calcNumberOfElementsByShape(forwardInput->shape));
    validateBfpQConfigShape(gyQC, calcNumberOfElementsByShape(lossGrad->shape));

    for (size_t oc = 0; oc < outChannels; oc++) {
        eAccOf[oc] = 0; /* a channel without products keeps 0 */
    }
    memset(acc, 0, gwElems * sizeof(int32_t));
    if (gwElems == 0) {
        return;
    }

    size_t maxTaps = 0;
    for (size_t k = 0; k < kernelSize; k++) {
        size_t tapLo;
        size_t tapHi;
        windowTapOutputRange1d(&geom, k, &tapLo, &tapHi);
        maxTaps = tapHi - tapLo > maxTaps ? tapHi - tapLo : maxTaps;
    }
    int guard = bfpFoldGuardBits(batch * maxTaps, xQC->mantissaBits, gyQC->mantissaBits, 0,
                                 "conv1dCalcWeightGradsBfp");

    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t xExpBias = bfpExponentBias(xQC);
    int32_t gyExpBias = bfpExponentBias(gyQC);

    /* Pre-pass (stack tables sized by the topology): E_acc per output
     * channel. */
    if (batch > 0 && inputLength > 0 && outputLength > 0) {
        int xMaxOf[batch * groups];
        bfpRowCursor_t xc = {.group = 0, .groupEnd = xQC->groupSize};
        for (size_t b = 0; b < batch; b++) {
            for (size_t g = 0; g < groups; g++) {
                xMaxOf[b * groups + g] = bfpRowMaxExponent(
                    &xc, xQC, xExpBias, (b * inChannels + g * inChPerGroup) * inputLength,
                    inChPerGroup * inputLength);
            }
        }
        int eTop[outChannels];
        bfpRowCursor_t gc = {.group = 0, .groupEnd = gyQC->groupSize};
        for (size_t b = 0; b < batch; b++) {
            for (size_t oc = 0; oc < outChannels; oc++) {
                int e = bfpRowMaxExponent(&gc, gyQC, gyExpBias,
                                          (b * outChannels + oc) * outputLength, outputLength) +
                        xMaxOf[b * groups + oc / outChPerGroup];
                eTop[oc] = (b == 0 || e > eTop[oc]) ? e : eTop[oc];
            }
        }
        for (size_t oc = 0; oc < outChannels; oc++) {
            eAccOf[oc] = eTop[oc] + guard;
        }
    }

    bfpGroupRange_t xRange = {0};
    bfpGroupRange_t gyRange = {0};

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    size_t gyIdx = (b * outChannels + oc) * outputLength + outPos;
                    int32_t gy = gyArr[gyIdx];
                    int gyShift =
                        eAccOf[oc] -
                        ((int)gyQC->exponents[bfpGroupAt(&gyRange, gyQC, gyIdx)] - gyExpBias);

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;

                            size_t xIdx = (b * inChannels + ic) * inputLength + inputIdx;
                            int shift =
                                gyShift -
                                ((int)xQC->exponents[bfpGroupAt(&xRange, xQC, xIdx)] - xExpBias);
                            acc[(oc * inChPerGroup + icOffset) * kernelSize + kernelIdx] +=
                                bfpAlignPartial(mulInt32s(xArr[xIdx], gy), shift);
                        }
                    }
                }
            }
        }
    }
}

/* FLOAT32 grads and the fixed grid (BFP_RAW_NONE): the scatter's sums become
 * the float acc * 2^E_acc in place -- the bridge into the FLOAT32 raw. */
static void weightGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    const conv1dConfig_t *cfg = ctx;
    size_t outChannels = ops[1]->shape->dimensions[1];
    int eAccOf[outChannels > 0 ? outChannels : 1];
    size_t gwElems = calcNumberOfElementsByTensor(rawOut);
    int32_t *acc = (int32_t *)rawOut->data;
    weightGradBfpScatter(cfg, ops[0], ops[1], acc, gwElems, eAccOf);
    if (gwElems == 0) {
        return;
    }
    size_t rowLen = gwElems / outChannels;
    for (size_t oc = 0; oc < outChannels; oc++) {
        for (size_t j = 0; j < rowLen; j++) {
            float v = bfpLdexpInt32(acc[oc * rowLen + j], eAccOf[oc]);
            memcpy(&acc[oc * rowLen + j], &v, sizeof(v));
        }
    }
}

/* BFP grads under DYNAMIC_RESCALE (BFP_RAW_BACKWARD_ACC): the scatter's sums
 * become the BFP_RAW on the grad's geometry (bfpRawEmitFromRows: one
 * exponent per grad group from the data, rows = output channels of cinG * K
 * sums each), integer only; the funnel accumulates it into the grad. */
static void weightGradKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    if (rawOut->quantization->type != BFP_RAW) {
        PRINT_ERROR("Conv1d backward (BFP weightGrad): the integer kernel needs a BFP_RAW raw "
                    "(opSpec_t.bfpRawOut = BFP_RAW_BACKWARD_ACC)");
        exit(1);
    }
    const conv1dConfig_t *cfg = ctx;
    size_t outChannels = ops[1]->shape->dimensions[1];
    int eAccOf[outChannels > 0 ? outChannels : 1];
    size_t gwElems = calcNumberOfElementsByTensor(rawOut);
    int32_t *acc = (int32_t *)rawOut->data;
    weightGradBfpScatter(cfg, ops[0], ops[1], acc, gwElems, eAccOf);
    bfpRawEmitFromRows(acc, gwElems, eAccOf, gwElems > 0 ? gwElems / outChannels : 0,
                       rawOut->quantization->qConfig, "conv1dCalcWeightGradsBfp");
}

void conv1dCalcWeightGradsBfp(conv1dConfig_t *cfg, tensor_t *forwardInput, tensor_t *lossGrad,
                              const rngKey_t *key) {
    executeOpValidateAccMode(cfg->weightGradAccMode, "Conv1d weightGradAccMode");
    tensor_t *weightTensor = cfg->weights->param;
    /* Public-boundary re-guard of the conv1dBackward dispatch rule (the
     * BFP-weights authority there): a direct caller with FLOAT32 weights
     * would otherwise read a bfpQConfig_t out of a float32QConfig (garbage
     * staging widths). */
    if (weightTensor->quantization->type != BFP) {
        PRINT_ERROR("conv1dCalcWeightGradsBfp: requires BFP-stored weights (the width anchor "
                    "for FLOAT32-operand staging); got dtype %d",
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
    /* BFP grad storage under DYNAMIC_RESCALE: integer end to end (BFP_RAW
     * into the integer accumulate engine). FLOAT32 grads and the fixed grid
     * keep the float bridge. */
    bool intAcc = cfg->weights->grad->quantization->type == BFP &&
                  cfg->weightGradAccMode == OUT_ACC_DYNAMIC_RESCALE;
    executeOp(
        &(opSpec_t){
            .kernel = intAcc ? weightGradKernelBfpRaw : weightGradKernelBfp,
            .ctx = cfg,
            .inputs = (tensor_t *[]){forwardInput, lossGrad},
            .nInputs = 2,
            .arithmetic = cfg->weightGradMath,
            .key = key,
            .mode = cfg->weightGradAccMode,
            .bfpStage = {forwardInput->quantization->type == FLOAT32 ? &stage : NULL,
                         lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL},
            .bfpRawOut = intAcc ? BFP_RAW_BACKWARD_ACC : BFP_RAW_NONE,
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
    const conv1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];

    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1d backward (biasGrad): lossGrad outChannels (%zu) does not match "
                    "bias Cout (%zu)",
                    outChannels, biasOutChannels);
        exit(1);
    }

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t *rawArr = (int32_t *)rawOut->data;
    float lossScale = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale;

    /* int32 accumulator (NO int64). The loss wire's declared qMaxBits bounds each
     * code, so N * 2^(q-1) <= INT32_MAX proves the sum safe (16-bit wire:
     * N <= 65535); otherwise the checked twin aborts on the first real overflow.
     * checkedMulSize: a wrapped count must not select the unchecked loop. */
    size_t terms;
    if (!checkedMulSize(batch, outputLength, &terms)) {
        PRINT_ERROR("Conv1d biasGrad: batch %zu * outputLength %zu overflows size_t", batch,
                    outputLength);
        exit(1);
    }
    uint8_t lossBits = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->qMaxBits;
    symAccumSite_t site = symAccumSiteInitSum("Conv1d biasGrad", terms, lossBits);
    if (symAccumProvenSafe(&site)) {
        biasGradSumLoopsSymSafe(gyArr, rawArr, batch, outChannels, outputLength, &site);
    } else {
        biasGradSumLoopsSymChecked(gyArr, rawArr, batch, outChannels, outputLength, &site);
    }
    ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale = lossScale;
}

void conv1dCalcBiasGradsSymInt32(conv1dConfig_t *cfg, tensor_t *lossGrad, const rngKey_t *key) {
    executeOpValidateAccMode(cfg->biasGradAccMode, "Conv1d biasGradAccMode");
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

/* BFP epic PR3 (Task 3): per-oc batch*outputLength sum of BFP loss mantissas
 * -- int32 partial per same-group visited segment, lossless power-of-two fold on
 * group change + tail (linearCalcBiasGradsBfp's core on the conv walk,
 * indices (b*outChannels + oc)*outputLength + outPos, b outer / outPos
 * inner). Sum headroom via bfpValidateSumHeadroom (the product bound does
 * not apply to sums). */
static void biasGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const conv1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];

    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1d backward (biasGrad): lossGrad outChannels (%zu) does not match "
                    "bias Cout (%zu)",
                    outChannels, biasOutChannels);
        exit(1);
    }

    bfpQConfig_t *gyQC = lossGrad->quantization->qConfig;
    validateBfpQConfigShape(gyQC, calcNumberOfElementsByShape(lossGrad->shape));
    bfpValidateSumHeadroom(gyQC, batch * outputLength, "conv1dCalcBiasGradsBfp");

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
                    acc += bfpLdexpInt32(partial, (int)gyQC->exponents[currentGroup] - expBias);
                    partial = 0;
                    currentGroup = g;
                }
                partial += gyArr[idx];
            }
        }
        if (batch > 0 && outputLength > 0) {
            acc += bfpLdexpInt32(partial, (int)gyQC->exponents[currentGroup] - expBias);
        }
        out[oc] = acc;
    }
}

/* biasGradKernelBfp's walk and segments with an int32 destination instead
 * of the float fold (BFP grads under DYNAMIC_RESCALE, BFP_RAW_BACKWARD_ACC):
 * each same-group segment's exact code sum enters as partial *
 * 2^-(E_acc - E_segment) (bfpAlignPartial), E_acc per output channel = max
 * over the batch of the loss row's max exponent + the guard of a pure sum of
 * batch * outputLength codes (bfpSumFoldGuardBits: the fold proof with the
 * second factor 1), so no destination can leave int32. Then the BFP_RAW on
 * the grad's geometry (bfpRawEmitFromRows, rows of one sum). */
static void biasGradKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                 const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    const conv1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];

    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1d backward (biasGrad): lossGrad outChannels (%zu) does not match "
                    "bias Cout (%zu)",
                    outChannels, biasOutChannels);
        exit(1);
    }
    if (rawOut->quantization->type != BFP_RAW) {
        PRINT_ERROR("Conv1d backward (BFP biasGrad): the integer kernel needs a BFP_RAW raw "
                    "(opSpec_t.bfpRawOut = BFP_RAW_BACKWARD_ACC)");
        exit(1);
    }

    bfpQConfig_t *gyQC = lossGrad->quantization->qConfig;
    validateBfpQConfigShape(gyQC, calcNumberOfElementsByShape(lossGrad->shape));
    bfpValidateSumHeadroom(gyQC, batch * outputLength, "conv1dCalcBiasGradsBfp");

    int32_t expBias = bfpExponentBias(gyQC);
    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t *acc = (int32_t *)rawOut->data;

    int eAccOf[outChannels > 0 ? outChannels : 1];
    for (size_t oc = 0; oc < outChannels; oc++) {
        eAccOf[oc] = 0;
    }
    if (batch > 0 && outputLength > 0) {
        int guard =
            bfpSumFoldGuardBits(batch * outputLength, gyQC->mantissaBits, "conv1dCalcBiasGradsBfp");
        bfpRowCursor_t gc = {.group = 0, .groupEnd = gyQC->groupSize};
        for (size_t b = 0; b < batch; b++) {
            for (size_t oc = 0; oc < outChannels; oc++) {
                int e = bfpRowMaxExponent(&gc, gyQC, expBias, (b * outChannels + oc) * outputLength,
                                          outputLength) +
                        guard;
                eAccOf[oc] = (b == 0 || e > eAccOf[oc]) ? e : eAccOf[oc];
            }
        }
    }

    bfpGroupRange_t gyRange = {0};
    for (size_t oc = 0; oc < outChannels; oc++) {
        int32_t sum = 0;
        int32_t partial = 0;
        size_t currentGroup = 0;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                size_t idx = (b * outChannels + oc) * outputLength + outPos;
                size_t g = bfpGroupAt(&gyRange, gyQC, idx);
                if (b == 0 && outPos == 0) {
                    currentGroup = g;
                } else if (g != currentGroup) {
                    sum += bfpAlignPartial(
                        partial, eAccOf[oc] - ((int)gyQC->exponents[currentGroup] - expBias));
                    partial = 0;
                    currentGroup = g;
                }
                partial += gyArr[idx];
            }
        }
        if (batch > 0 && outputLength > 0) {
            sum += bfpAlignPartial(partial,
                                   eAccOf[oc] - ((int)gyQC->exponents[currentGroup] - expBias));
        }
        acc[oc] = sum;
    }
    bfpRawEmitFromRows(acc, outChannels, eAccOf, 1, rawOut->quantization->qConfig,
                       "conv1dCalcBiasGradsBfp");
}

void conv1dCalcBiasGradsBfp(conv1dConfig_t *cfg, tensor_t *lossGrad, const rngKey_t *key) {
    executeOpValidateAccMode(cfg->biasGradAccMode, "Conv1d biasGradAccMode");
    tensor_t *weightTensor = cfg->weights->param;
    /* Public-boundary re-guard of the conv1dBackward dispatch rule (the
     * BFP-weights authority there) -- see conv1dCalcWeightGradsBfp. */
    if (weightTensor->quantization->type != BFP) {
        PRINT_ERROR("conv1dCalcBiasGradsBfp: requires BFP-stored weights (the width anchor "
                    "for FLOAT32-operand staging); got dtype %d",
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
    /* As the weightGrad: integer end to end into BFP grads under
     * DYNAMIC_RESCALE, the float fold otherwise. */
    bool intAcc = cfg->bias->grad->quantization->type == BFP &&
                  cfg->biasGradAccMode == OUT_ACC_DYNAMIC_RESCALE;
    executeOp(
        &(opSpec_t){
            .kernel = intAcc ? biasGradKernelBfpRaw : biasGradKernelBfp,
            .ctx = cfg,
            .inputs = (tensor_t *[]){lossGrad},
            .nInputs = 1,
            .arithmetic = cfg->biasGradMath,
            .key = key,
            .mode = cfg->biasGradAccMode,
            .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
            .bfpRawOut = intAcc ? BFP_RAW_BACKWARD_ACC : BFP_RAW_NONE,
        },
        cfg->bias->grad);
}

/* dx adapters (ctx = conv1dForwardCtx_t*, PR3 Task 3): dL/dx via the
 * adjoint -- convTranspose1d of lossGrad with the Conv1d weight [oc][ic][K]
 * (every arithmetic takes a SCATTER core; BFP its integer twin).
 * The scatter kernel's weight index arithmetic
 * ((ic*outChPerGroup + ocOffset)*K + k, ConvTranspose1dKernel.c) computes
 * exactly the Conv1d weight's flat storage index in this adjoint role, so a
 * grouped weight's storage-bound quantization groups apply unchanged: route
 * to the grouped SCATTER entry with the stored weight's symGroupedWeight_t. */
static void propLossKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    convTranspose1dKernelFloat32(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, 0u, rawOut);
}

static void propLossKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    if (fctx->groupedWeight != NULL) {
        convTranspose1dKernelSymInt32Grouped(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, 0u,
                                             rawOut, fctx->groupedWeight, rs);
    } else {
        convTranspose1dKernelSymInt32(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, 0u, rawOut,
                                      rs);
    }
}

/* dx on the FLOAT32 scatter loop with integer accumulators into the BFP_RAW
 * (convTranspose1dKernelBfpScatter) -- same {cfg, groupedWeight}
 * ctx wrapper as the other dx adapters, groupedWeight always NULL for BFP
 * (symGroupedWeightOrNull returns NULL for the BFP dtype). */
static void propLossKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    convTranspose1dKernelBfpScatter(ops[0], ops[1], cfg->kernel, cfg->groups, 0u, rawOut);
}

void conv1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad, tensor_t *propLoss,
                    const layerExec_t *exec) {
    rngKey_t opKey;
    conv1dConfig_t *cfg = layer->config->conv1d;
    tensor_t *weightTensor = cfg->weights->param;

    size_t lossDims[3];
    size_t lossOrder[3];
    shape_t lossExpected = {.dimensions = lossDims, .orderOfDimensions = lossOrder};
    conv1dCalcOutputShape(layer, forwardInput->shape, &lossExpected);
    layerRequireDenseWire(lossGrad, &lossExpected, "Conv1d backward", "loss");
    if (propLoss != NULL) {
        size_t inOrder[3] = {0, 1, 2};
        shape_t inExpected = {.numberOfDimensions = 3,
                              .dimensions = forwardInput->shape->dimensions,
                              .orderOfDimensions = inOrder};
        layerRequireDenseWire(propLoss, &inExpected, "Conv1d backward", "propLoss");
    }

    /* BFP epic PR3, backward mirror of the forward's ARITH_BFP rule 1 (see
     * conv1dForward; Linear.c's linearBackward is the canonical backward
     * copy): BFP-stored weights are the width anchor every FLOAT32-stored
     * operand stages at. The stack template below serves the propLoss
     * executeOp call in THIS frame (the two grad WRAPPERS rebuild it
     * internally -- they are public, callable without passing here).
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
            PRINT_ERROR("Conv1d backward: ARITH_BFP math slots require BFP-stored weights (the "
                        "width anchor for FLOAT32-operand staging; FLOAT32-init + "
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
            conv1dCalcWeightGradsFloat32(
                cfg, forwardInput, lossGrad,
                layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_WEIGHT_GRAD, &opKey));
            break;
        case ARITH_SYM_INT32:
            conv1dCalcWeightGradsSymInt32(
                cfg, forwardInput, lossGrad,
                layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_WEIGHT_GRAD, &opKey));
            break;
        case ARITH_BFP:
            conv1dCalcWeightGradsBfp(
                cfg, forwardInput, lossGrad,
                layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_WEIGHT_GRAD, &opKey));
            break;
        default:
            PRINT_ERROR("Conv1d backward (weightGrad): quantization type not implemented");
            exit(1);
        }

        switch (cfg->biasGradMath.type) {
        case ARITH_FLOAT32:
            if (cfg->bias) {
                conv1dCalcBiasGradsFloat32(
                    cfg, lossGrad,
                    layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_BIAS_GRAD, &opKey));
            }
            break;
        case ARITH_SYM_INT32:
            if (cfg->bias) {
                conv1dCalcBiasGradsSymInt32(
                    cfg, lossGrad,
                    layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_BIAS_GRAD, &opKey));
            }
            break;
        case ARITH_BFP:
            if (cfg->bias) {
                conv1dCalcBiasGradsBfp(
                    cfg, lossGrad,
                    layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_BIAS_GRAD, &opKey));
            }
            break;
        default:
            PRINT_ERROR("Conv1d backward (biasGrad): quantization type not implemented");
            exit(1);
        }
    }

    /* propLoss (dx wire): OUT_WRITE. For a SYM_INT32 target this now requants
     * through the conversionMatrix diagonal (width-restored at the producer,
     * PR1b.2 D3) instead of the old direct kernel write of raw, unrestored
     * accumulator-range mantissas — the #187 dtype guard is superseded by the
     * funnel's own prologue/epilogue and is deleted (it had
     * zero test coverage and was a tautology after #221).
     * propLoss == NULL (#380 PR2): grads-only call -- skip the dx write
     * entirely rather than dereference the absent buffer. */
    if (propLoss != NULL) {
        /* Group-quant PR3 (Task 3) + PR4 (grouped ASYM via the view): same
         * detection + always-together wiring as conv1dForward (see the
         * comment there) -- ctx routes the SYM dx adapter to the grouped
         * SCATTER entry, groupedSymOperandPos opts the funnel prologue into
         * unpacking (SYM arm) / group-aware dequant (FLOAT32 arm) of the
         * weight at inputs[1] (position 2), declared on BOTH math arms (the same
         * arm-parity rule as conv1dForward). */
        symQConfig_t asymWeightView; /* lifetime: this frame (SymGroupedWeight.h view doc) */
        symGroupedWeight_t groupedWeightCtx;
        const symGroupedWeight_t *groupedWeight =
            symGroupedWeightOrNull(weightTensor, &asymWeightView, &groupedWeightCtx);
        bool grouped = groupedWeight != NULL;
        conv1dForwardCtx_t fctx = {.cfg = cfg, .groupedWeight = groupedWeight};

        switch (cfg->propLossMath.type) {
        case ARITH_FLOAT32:
            executeOp(
                &(opSpec_t){
                    .kernel = propLossKernelFloat,
                    .ctx = &fctx,
                    .inputs = (tensor_t *[]){lossGrad, weightTensor},
                    .nInputs = 2,
                    .arithmetic = cfg->propLossMath,
                    .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_PROP_LOSS, &opKey),
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
                    .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_PROP_LOSS, &opKey),
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
                    .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, CONV1D_OP_PROP_LOSS, &opKey),
                    .mode = OUT_WRITE,
                    /* SYM/ASYM-carrier gate -- BFP blocking is
                     * per-operand-legal, nothing is declared. */
                    .groupedSymOperandPos = 0,
                    /* weights operand: always BFP-stored under ARITH_BFP
                     * (rule-1 mirror above) -> borrowed zero-copy, never
                     * staged. */
                    .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL,
                                 NULL},
                    /* integer dx end to end: the kernel emits the BFP_RAW,
                     * the epilogue re-blocks it into the dx wire by the op's
                     * mode (arithmetic-bfp.md §14) */
                    .bfpRawOut = BFP_RAW_BACKWARD_DX,
                },
                propLoss);
            break;
        default:
            PRINT_ERROR("Conv1d backward (propLoss): quantization type not implemented");
            exit(1);
        }
    }
}

void conv1dCalcOutputShape(layer_t *conv1dLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3D input [batch, channel, length], got %zuD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;
    size_t batchSize = inputShape->dimensions[0];
    size_t inputLength = inputShape->dimensions[2];
    size_t outChannels = cfg->weights->param->shape->dimensions[0];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outChannels;
    outputShape->dimensions[2] = geom.outputLength;
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
