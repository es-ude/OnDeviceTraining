#define SOURCE_FILE "ODT_CONV1D"

#include "Conv1d.h"

#include <string.h>

#include "Common.h"
#include "Conv1dKernel.h"
#include "ConvTranspose1dKernel.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Mul.h"
#include "Quantization.h"
#include "SlidingWindow1d.h"
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
 * AND -- only when the stored weight is grouped SYM -- the ORIGINAL
 * symQConfig_t carrying the real per-group scales. executeOp's prologue
 * hands the kernel only the UNPACKED, poisoned-scale scratch (ops[1]), never
 * the source tensor (ExecuteOp.c), so the group shape must travel via ctx,
 * exactly like Linear's linearForwardKernelSym (which gets away with ctx ==
 * weightGroups alone because matmul infers geometry from the tensors
 * themselves; Conv1d's kernel additionally needs kernel_t/groups, hence this
 * wrapper). Set together with groupedSymOperandPos (conv1dForward, below)
 * -- both or neither, same invariant as Linear.c. PR3 (Task 3): the dx
 * (propLoss) adapters carry the SAME wrapper -- dx consumes the weight AS
 * STORED through the adjoint scatter, whose group binding is to flat
 * storage, so the identical {cfg, weightGroups} pair serves both
 * directions. */
typedef struct conv1dForwardCtx {
    const conv1dConfig_t *cfg;
    const symQConfig_t *weightGroups; /* NULL unless cfg->weights->param is grouped SYM */
} conv1dForwardCtx_t;

/* executeOp forward kernel adapters — operands are {input, weights} or
 * {input, weights, bias} (bias omitted, not NULL-padded, when the layer has
 * no bias) — same convention Linear's forward adapters use (Linear.c). */
static void forwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                               const void *ctx) {
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    conv1dKernelFloat32(ops[0], ops[1], bias, cfg->kernel, cfg->groups, rawOut);
}
static void forwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                             const void *ctx) {
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    if (fctx->weightGroups != NULL) {
        conv1dKernelSymInt32Grouped(ops[0], ops[1], bias, cfg->kernel, cfg->groups, rawOut,
                                    fctx->weightGroups);
    } else {
        conv1dKernelSymInt32(ops[0], ops[1], bias, cfg->kernel, cfg->groups, rawOut);
    }
}

void conv1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    conv1dConfig_t *cfg = layer->config->conv1d;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    /* Group-quant PR2: a stored SYM weight with numGroups > 1 routes the SYM
     * kernel adapter to the grouped gather-core entry (weightGroups carried
     * via ctx) AND opts the funnel's prologue into unpacking the grouped
     * operand (groupedSymOperandPos), always together -- mirrors
     * linearForward's identical wiring (Linear.c) exactly. Per-tensor SYM
     * (numGroups==1) and SYM_INT32 weights are untouched. weightTensor is
     * always inputs[1] (bias present or not, see the .inputs literals below)
     * in BOTH math arms -- final-review Fix 3(b): the FLOAT32 arm must
     * declare the SAME position as the SYM arm (a grouped weight forwarded
     * under FLOAT32 math dequantizes via the funnel's group-aware
     * convertTensor cell, Task 2 -- gated on this field exactly like the SYM
     * arm's unpack, not a different mechanism) -- omitting it here (as
     * pre-final-review code did) would make Conv1d's FLOAT32-math grouped
     * forward regress once the funnel's FLOAT32 arm gate lands. */
    bool grouped = weightTensor->quantization->type == SYM &&
                   ((symQConfig_t *)weightTensor->quantization->qConfig)->numGroups > 1;
    const symQConfig_t *weightGroups =
        grouped ? (const symQConfig_t *)weightTensor->quantization->qConfig : NULL;
    conv1dForwardCtx_t fctx = {.cfg = cfg, .weightGroups = weightGroups};

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
    default:
        PRINT_ERROR("Conv1d forward: quantization type not implemented");
        exit(1);
    }
}

/* executeOp kernel adapters (ctx = conv1dConfig_t*, for kernel_t/groups
 * geometry — recon-conv-backward §8: the fixed opKernelFn_t shape has no
 * per-op-instance geometry slot other than ctx). Weight-grad kernels `+=`
 * into the same weight cell across many (b, outPos) iterations, so they
 * memset rawOut first (the executeOp Phase-2 scratch is an uninitialized
 * VLA, unlike the reserveMemory-backed intermediate they replace — recon
 * §2). Bias-grad kernels write each output-channel index exactly once (no
 * zero-init hazard). SYM weight-grad sets the raw intermediate's scale
 * itself (s_in*s_loss); SYM bias-grad emits the raw per-channel sum at the
 * loss scale and lets the OUT_ACC_FIXED_SCALE epilogue's
 * rescaleIntoAccumulatorScale (target roundingMode, spec D4) do the rescale
 * that used to happen inline here. */
static void weightGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx) {
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
                                         tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->weightGradAccMode, "Conv1d weightGradAccMode");
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

static void conv1dCalcBiasGradsFloat32(conv1dConfig_t *cfg, tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->biasGradAccMode, "Conv1d biasGradAccMode");
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
                            interData[(oc * inChPerGroup + icOffset) * kernelSize + kernelIdx] +=
                                mulInt32s(xv, gy);
                        }
                    }
                }
            }
        }
    }
}

void conv1dCalcWeightGradsSymInt32(conv1dConfig_t *cfg, tensor_t *forwardInput,
                                   tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->weightGradAccMode, "Conv1d weightGradAccMode");
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

static void biasGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
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

    for (size_t oc = 0; oc < outChannels; oc++) {
        /* int32 accumulator (NO int64): loss mantissas are int12-range per the
         * qMaxBits=12 operand contract, so the batch*outputLength sum stays
         * well within int32 (even more headroom than the old int16 path). */
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

void conv1dCalcBiasGradsSymInt32(conv1dConfig_t *cfg, tensor_t *lossGrad) {
    executeOpValidateAccMode(cfg->biasGradAccMode, "Conv1d biasGradAccMode");
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

/* dx adapters (ctx = conv1dForwardCtx_t*, PR3 Task 3): dL/dx via the adjoint
 * scatter -- convTranspose1d of lossGrad with the Conv1d weight [oc][ic][K].
 * The scatter kernel's weight index arithmetic
 * ((ic*outChPerGroup + ocOffset)*K + k, ConvTranspose1dKernel.c) computes
 * exactly the Conv1d weight's flat storage index in this adjoint role, so a
 * grouped weight's storage-bound quantization groups apply unchanged: route
 * to the grouped SCATTER entry with the stored weight's own symQConfig_t. */
static void propLossKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    convTranspose1dKernelFloat32(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, 0u, rawOut);
}

static void propLossKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    const conv1dForwardCtx_t *fctx = ctx;
    const conv1dConfig_t *cfg = fctx->cfg;
    if (fctx->weightGroups != NULL) {
        convTranspose1dKernelSymInt32Grouped(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, 0u,
                                             rawOut, fctx->weightGroups);
    } else {
        convTranspose1dKernelSymInt32(ops[0], ops[1], NULL, cfg->kernel, cfg->groups, 0u, rawOut);
    }
}

void conv1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                    tensor_t *propLoss) {
    conv1dConfig_t *cfg = layer->config->conv1d;

    if (!cfg->frozen) {
        switch (cfg->weightGradMath.type) {
        case ARITH_FLOAT32:
            conv1dCalcWeightGradsFloat32(cfg, forwardInput, lossGrad);
            break;
        case ARITH_SYM_INT32:
            conv1dCalcWeightGradsSymInt32(cfg, forwardInput, lossGrad);
            break;
        default:
            PRINT_ERROR("Conv1d backward (weightGrad): quantization type not implemented");
            exit(1);
        }

        switch (cfg->biasGradMath.type) {
        case ARITH_FLOAT32:
            if (cfg->bias) {
                conv1dCalcBiasGradsFloat32(cfg, lossGrad);
            }
            break;
        case ARITH_SYM_INT32:
            if (cfg->bias) {
                conv1dCalcBiasGradsSymInt32(cfg, lossGrad);
            }
            break;
        default:
            PRINT_ERROR("Conv1d backward (biasGrad): quantization type not implemented");
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
        tensor_t *weightTensor = cfg->weights->param;

        /* Group-quant PR3 (Task 3): same detection + always-together wiring
         * as conv1dForward (see the comment there) -- ctx routes the SYM dx
         * adapter to the grouped SCATTER entry, groupedSymOperandPos opts the
         * funnel prologue into unpacking (SYM arm) / group-aware dequant
         * (FLOAT32 arm) of the weight at inputs[1] (position 2), declared on
         * BOTH math arms (PR2 final-review arm-parity lesson). */
        bool grouped = weightTensor->quantization->type == SYM &&
                       ((symQConfig_t *)weightTensor->quantization->qConfig)->numGroups > 1;
        const symQConfig_t *weightGroups =
            grouped ? (const symQConfig_t *)weightTensor->quantization->qConfig : NULL;
        conv1dForwardCtx_t fctx = {.cfg = cfg, .weightGroups = weightGroups};

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
        default:
            PRINT_ERROR("Conv1d backward (propLoss): quantization type not implemented");
            exit(1);
        }
    }
}

void conv1dCalcOutputShape(layer_t *conv1dLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3D input [batch, channel, length], got %luD",
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
