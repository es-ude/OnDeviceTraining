#define SOURCE_FILE "LINEAR"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "Arithmetic.h"
#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "LayerExec.h"
#include "LayerWireGuards.h"
#include "Linear.h"
#include "Matmul.h"
#include "Rounding.h"
#include "ScaleCanon.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"

void linearValidateConfig(const linearConfig_t *cfg, const char *who) {
    size_t outFeatures = cfg->weights->param->shape->dimensions[0];
    if (cfg->bias != NULL && calcNumberOfElementsByTensor(cfg->bias->param) != outFeatures) {
        PRINT_ERROR("%s: bias length (%zu) must equal outFeatures (%zu)", who,
                    calcNumberOfElementsByTensor(cfg->bias->param), outFeatures);
        exit(1);
    }
}

void linearForwardFloat(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulFloat32TensorsWithBias(input, w, output, b);
    transposeTensor(w, 0, 1);
}

void linearForwardSymInt32(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output,
                           rngStream_t *rs) {
    transposeTensor(w, 0, 1);
    matmulSymInt32TensorsWithBias(input, w, output, b, rs);
    transposeTensor(w, 0, 1);
}

/* Grouped scratch uses groupedWeight's scales and code range; groups follow physical storage
 * across the transpose. */
void linearForwardSymInt32Grouped(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output,
                                  const symGroupedWeight_t *groupedWeight, rngStream_t *rs) {
    transposeTensor(w, 0, 1);
    matmulSymInt32TensorsGroupedWeight(input, w, b, output, groupedWeight, rs);
    transposeTensor(w, 0, 1);
}

/* All operands are the executeOp prologue's unpacked-BFP scratch (int32
 * mantissa codes under a live bfpQConfig_t — borrowed exponents for
 * BFP-stored sources, funnel-staged for FLOAT32 ones); `output` is the
 * BFP_RAW raw (integer-only forward, arithmetic-bfp.md §13). Same transpose
 * dance as linearForwardSymInt32Grouped: BFP groups bind to STORAGE order,
 * which transposeTensor exposes zero-copy as the physically-innermost axis
 * for the reduction (the row-top pre-pass of matmulBfpRawTensors relies on
 * that to walk weight rows as contiguous runs). */
void linearForwardBfpRaw(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulBfpRawTensors(input, w, b, output);
    transposeTensor(w, 0, 1);
}

/* executeOp forward kernel adapters — operands are {input, weights} or
 * {input, weights, bias} (bias omitted, not NULL-padded, when the layer has
 * no bias); ctx unused (matmul infers geometry from the tensors themselves),
 * EXCEPT linearForwardKernelSym: ctx carries the stored weight's
 * symGroupedWeight_t* (non-NULL) iff it is grouped SYM or ASYM (linearForward
 * sets ctx + groupedSymOperandPos together, see below) — that routes to the
 * grouped matmul entry instead of the scalar one. */
static void linearForwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                     const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    (void)ctx;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    linearForwardFloat(ops[1], bias, ops[0], rawOut);
}
static void linearForwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    const symGroupedWeight_t *groupedWeight = (const symGroupedWeight_t *)ctx;
    if (groupedWeight != NULL) {
        linearForwardSymInt32Grouped(ops[1], bias, ops[0], rawOut, groupedWeight, rs);
    } else {
        linearForwardSymInt32(ops[1], bias, ops[0], rawOut, rs);
    }
}
static void linearForwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                      const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    (void)ctx;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    linearForwardBfpRaw(ops[1], bias, ops[0], rawOut);
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    LINEAR_OP_FORWARD = 0,
    LINEAR_OP_WEIGHT_GRAD = 1,
    LINEAR_OP_BIAS_GRAD = 2,
    LINEAR_OP_PROP_LOSS = 3,
};

void linearForward(layer_t *linearLayer, tensor_t *input, tensor_t *output,
                   const layerExec_t *exec) {
    rngKey_t opKey;
    linearConfig_t *linearConfig = linearLayer->config->linear;

    tensor_t *weights = getParamFromParameter(linearConfig->weights);
    tensor_t *bias = linearConfig->bias != NULL ? getParamFromParameter(linearConfig->bias) : NULL;

    size_t outDims[2];
    size_t outOrder[2];
    shape_t outExpected = {.dimensions = outDims, .orderOfDimensions = outOrder};
    linearCalcOutputShape(linearLayer, input->shape, &outExpected);
    layerRequireDenseWire(output, &outExpected, "Linear forward", "output");

    /* BFP-stored weights anchor FLOAT32 operand staging widths. The stack template lives
     * through executeOp; executeOp owns exponents and applies each operation's rounding. */
    if (linearConfig->forwardMath.type == ARITH_BFP) {
        if (weights->quantization->type != BFP) {
            PRINT_ERROR("Linear: ARITH_BFP forward requires BFP-stored weights (FLOAT32-init + "
                        "requantizeTensorInPlace, see docs/conventions/arithmetic-bfp.md); got "
                        "dtype %d",
                        (int)weights->quantization->type);
            exit(1);
        }
        const bfpQConfig_t *wQC = weights->quantization->qConfig;
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = linearConfig->forwardMath.roundingMode,
                              .mantissaBits = wQC->mantissaBits,
                              .exponentBits = wQC->exponentBits};
        executeOp(
            &(opSpec_t){
                .kernel = linearForwardKernelBfpRaw,
                .inputs = bias != NULL ? (tensor_t *[]){input, weights, bias}
                                       : (tensor_t *[]){input, weights},
                .nInputs = bias != NULL ? 3 : 2,
                .arithmetic = linearConfig->forwardMath,
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, LINEAR_OP_FORWARD, &opKey),
                .mode = OUT_WRITE,
                .groupedSymOperandPos = 0,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL,
                             (bias != NULL && bias->quantization->type == FLOAT32) ? &stage : NULL},
                /* Forward raw contract (the forward marker); the backward
                 * ops set their own (BACKWARD_DX / BACKWARD_ACC). */
                .bfpRawOut = BFP_RAW_FORWARD,
            },
            output);
        return;
    }

    /* Grouped weights require both grouped-kernel routing and prologue unpacking. The weight
     * is operand 1, encoded as groupedSymOperandPos = 2. */
    symQConfig_t asymWeightView; /* lifetime: this frame (SymGroupedWeight.h view doc) */
    symGroupedWeight_t groupedWeightCtx;
    const symGroupedWeight_t *groupedWeight =
        symGroupedWeightOrNull(weights, &asymWeightView, &groupedWeightCtx);
    bool grouped = groupedWeight != NULL;

    executeOp(
        &(opSpec_t){
            .kernel = linearConfig->forwardMath.type == ARITH_SYM_INT32 ? linearForwardKernelSym
                                                                        : linearForwardKernelFloat,
            .ctx = groupedWeight,
            .inputs = bias != NULL ? (tensor_t *[]){input, weights, bias}
                                   : (tensor_t *[]){input, weights},
            .nInputs = bias != NULL ? 3 : 2,
            .arithmetic = linearConfig->forwardMath,
            .key = layerExecOpKey(exec, RNG_TAG_FORWARD, LINEAR_OP_FORWARD, &opKey),
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
    size_t batch = numLoss / numFeatures;
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

/* Per-feature batch sum of loss codes. The loop body is instantiated twice
 * (SymAccumGuard.h): unchanged when the loss wire's declared width proves the
 * sum fits int32, overflow-checked otherwise. */
SYM_ACCUM_ALWAYS_INLINE void linearBiasGradSumLoops(const int32_t *l, int32_t *bg,
                                                    size_t numFeatures, size_t batch,
                                                    const bool checked,
                                                    const symAccumSite_t *site) {
    for (size_t f = 0; f < numFeatures; f++) {
        int32_t sum = 0;
        for (size_t n = 0; n < batch; n++) {
            sum = symAdd(sum, l[n * numFeatures + f], checked, site);
        }
        bg[f] = sum;
    }
}

SYM_ACCUM_NOINLINE void linearBiasGradSumLoopsSafe(const int32_t *l, int32_t *bg,
                                                   size_t numFeatures, size_t batch,
                                                   const symAccumSite_t *site) {
    linearBiasGradSumLoops(l, bg, numFeatures, batch, false, site);
}

SYM_ACCUM_NOINLINE void linearBiasGradSumLoopsChecked(const int32_t *l, int32_t *bg,
                                                      size_t numFeatures, size_t batch,
                                                      const symAccumSite_t *site) {
    linearBiasGradSumLoops(l, bg, numFeatures, batch, true, site);
}

void linearCalcBiasGradsSymInt32(tensor_t *biasGrads, tensor_t *loss) {
    /* Raw emit: per-feature batch sums of loss mantissas, at the LOSS scale.
     * The executeOp OUT_ACC_FIXED_SCALE epilogue rescales into the persistent
     * grad's existing scale and integer-adds (Deutel-adjacent ODT scheme). */
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrads);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    size_t batch = numLoss / numFeatures;
    int32_t *bg = (int32_t *)biasGrads->data;
    int32_t *l = (int32_t *)loss->data;
    /* int32 accumulator. N * 2^(qMaxBits-1) <= INT32_MAX proves the batch sum safe; otherwise
     * the checked twin aborts on the first real overflow. */
    uint8_t lossBits = ((symInt32QConfig_t *)loss->quantization->qConfig)->qMaxBits;
    symAccumSite_t site = symAccumSiteInitSum("linearCalcBiasGradsSymInt32", batch, lossBits);
    if (symAccumProvenSafe(&site)) {
        linearBiasGradSumLoopsSafe(l, bg, numFeatures, batch, &site);
    } else {
        linearBiasGradSumLoopsChecked(l, bg, numFeatures, batch, &site);
    }
    /* Stamp the loss scale and the sum's static width: on a user tensor this output can
     * re-enter a SYM op, whose overflow proof reads the declared width. */
    symInt32QConfig_t *bgQC = biasGrads->quantization->qConfig;
    storeCanonScale(&bgQC->scale,
                    loadCanonScale(&((symInt32QConfig_t *)loss->quantization->qConfig)->scale));
    bgQC->qMaxBits = symAccumResultBits(&site);
}

void linearCalcPropLossSymInt32(tensor_t *weights, tensor_t *loss, tensor_t *propLoss) {
    matmulSymInt32Tensors(loss, weights, propLoss);
}

/* dx uses stored weights without transposition; group lookup follows each visited storage
 * index. */
void linearCalcPropLossSymInt32Grouped(tensor_t *weights, tensor_t *loss, tensor_t *propLoss,
                                       const symGroupedWeight_t *groupedWeight, rngStream_t *rs) {
    matmulSymInt32TensorsGroupedWeight(loss, weights, NULL, propLoss, groupedWeight, rs);
}

/* BFP matmul honors transpose views; restore the shared shape after weight-gradient
 * computation. The float-fold weightGrad serves FLOAT32 grads and the fixed grid; the integer
 * ones BFP grads under DYNAMIC_RESCALE and every dx. */
void linearCalcWeightGradsBfp(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads) {
    transposeTensor(loss, 0, 1);
    matmulBfpTensors(loss, forwardInput, NULL, weightGrads);
    transposeTensor(loss, 0, 1);
}

void linearCalcWeightGradsBfpRaw(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads) {
    transposeTensor(loss, 0, 1);
    matmulBfpRawTensors(loss, forwardInput, NULL, weightGrads);
    transposeTensor(loss, 0, 1);
}

void linearCalcPropLossBfp(tensor_t *loss, tensor_t *weights, tensor_t *propLoss) {
    matmulBfpRawTensors(loss, weights, NULL, propLoss);
}

/* Per-feature batch sum of BFP loss mantissas: int32 partial per same-group
 * visited segment (the walk strides by numFeatures, so groups may change every
 * step), lossless power-of-two fold on group change + tail. Sum headroom via
 * bfpValidateSumHeadroom (the product bound does not apply to sums). */
void linearCalcBiasGradsBfp(tensor_t *loss, tensor_t *biasGrad) {
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrad);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    size_t batch = numLoss / numFeatures;
    const bfpQConfig_t *qC = loss->quantization->qConfig;
    bfpValidateSumHeadroom(qC, batch, "linearCalcBiasGradsBfp");
    int32_t expBias = bfpExponentBias(qC);
    bfpGroupRange_t qRange = {0};
    const int32_t *codes = (const int32_t *)loss->data;
    float *out = (float *)biasGrad->data;
    for (size_t f = 0; f < numFeatures; f++) {
        float acc = 0.f;
        int32_t partial = 0;
        size_t currentGroup = 0;
        for (size_t n = 0; n < batch; n++) {
            size_t idx = n * numFeatures + f;
            size_t g = bfpGroupAt(&qRange, qC, idx);
            if (n == 0) {
                currentGroup = g;
            } else if (g != currentGroup) {
                acc += bfpLdexpInt32(partial, (int)qC->exponents[currentGroup] - expBias);
                partial = 0;
                currentGroup = g;
            }
            partial += codes[idx];
        }
        if (batch > 0) {
            acc += bfpLdexpInt32(partial, (int)qC->exponents[currentGroup] - expBias);
        }
        out[f] = acc;
    }
}

/* linearCalcBiasGradsBfp's walk (feature outer, batch inner) and segments
 * with an int32 destination instead of the float fold: each same-group
 * segment's exact code sum enters as partial * 2^-(E_acc - E_segment)
 * (bfpAlignPartial), E_acc per feature = max over the batch of its NONZERO
 * codes' group exponents (none: 0) + the guard of a pure sum of `batch`
 * codes (bfpSumFoldGuardBits), so no destination can leave int32. Then the
 * BFP_RAW on the grad's geometry with the exponent from the data
 * (bfpRawEmitFromRows, rows of one sum): a loose feature bound never
 * re-grids the other features. */
void linearCalcBiasGradsBfpRaw(tensor_t *loss, tensor_t *biasGrad) {
    if (loss->quantization->type != BFP) {
        PRINT_ERROR("linearCalcBiasGradsBfpRaw: loss must be BFP (unpacked scratch form)");
        exit(1);
    }
    if (biasGrad->quantization->type != BFP_RAW) {
        PRINT_ERROR("linearCalcBiasGradsBfpRaw: output must be BFP_RAW (opSpec_t.bfpRawOut = "
                    "BFP_RAW_BACKWARD_ACC)");
        exit(1);
    }
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrad);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    if (numFeatures == 0 || numLoss % numFeatures != 0) {
        PRINT_ERROR("linearCalcBiasGradsBfpRaw: loss elements %zu not divisible by features %zu",
                    numLoss, numFeatures);
        exit(1);
    }
    size_t batch = numLoss / numFeatures;
    const bfpQConfig_t *qC = loss->quantization->qConfig;
    validateBfpQConfigShape(qC, numLoss);
    bfpValidateSumHeadroom(qC, batch, "linearCalcBiasGradsBfpRaw");
    int32_t expBias = bfpExponentBias(qC);
    const int32_t *codes = (const int32_t *)loss->data;
    int32_t *acc = (int32_t *)biasGrad->data;

    /* Pre-pass in storage order (one cursor, no division): E_acc per feature,
     * from its nonzero codes only (a zero code never sets it). */
    int eAccOf[numFeatures];
    for (size_t f = 0; f < numFeatures; f++) {
        eAccOf[f] = batch > 0 ? BFP_NO_LIVE_EXP : 0; /* a feature without codes keeps 0 */
    }
    if (batch > 0) {
        int guard = bfpSumFoldGuardBits(batch, qC->mantissaBits, "linearCalcBiasGradsBfpRaw");
        bfpRowCursor_t c = {.group = 0, .groupEnd = qC->groupSize};
        for (size_t n = 0; n < batch; n++) {
            for (size_t f = 0; f < numFeatures; f++) {
                int e = bfpRowMaxExponent(&c, qC, expBias, codes, n * numFeatures + f, 1);
                eAccOf[f] = e > eAccOf[f] ? e : eAccOf[f];
            }
        }
        for (size_t f = 0; f < numFeatures; f++) {
            eAccOf[f] = bfpLiveTopOrZero(eAccOf[f]) + guard;
        }
    }

    bfpGroupRange_t qRange = {0};
    for (size_t f = 0; f < numFeatures; f++) {
        int32_t sum = 0;
        int32_t partial = 0;
        size_t currentGroup = 0;
        for (size_t n = 0; n < batch; n++) {
            size_t idx = n * numFeatures + f;
            size_t g = bfpGroupAt(&qRange, qC, idx);
            if (n == 0) {
                currentGroup = g;
            } else if (g != currentGroup) {
                sum += bfpAlignPartial(partial,
                                       eAccOf[f] - ((int)qC->exponents[currentGroup] - expBias));
                partial = 0;
                currentGroup = g;
            }
            partial += codes[idx];
        }
        if (batch > 0) {
            sum +=
                bfpAlignPartial(partial, eAccOf[f] - ((int)qC->exponents[currentGroup] - expBias));
        }
        acc[f] = sum;
    }
    bfpRawEmitFromRows(acc, numFeatures, eAccOf, 1, biasGrad->quantization->qConfig,
                       "linearCalcBiasGradsBfpRaw");
}

/* executeOp kernel adapters — ops convention: weight-grad {loss, fwdIn},
 * bias-grad {loss}, propLoss {loss, weightsParam}. auxOut unused; ctx unused
 * EXCEPT propLossKernelSym, which mirrors linearForwardKernelSym: ctx carries
 * the stored weight's symGroupedWeight_t* (non-NULL iff grouped SYM or ASYM,
 * linearBackward sets ctx + groupedSymOperandPos together) and routes to the
 * grouped matmul entry. */
static void weightGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsFloat32(ops[1], ops[0], rawOut);
}
static void weightGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsSymInt32(ops[0], ops[1], rawOut);
}
static void biasGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsFloat32(ops[0], rawOut);
}
static void biasGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsSymInt32(rawOut, ops[0]);
}
static void propLossKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcPropLossFloat32(ops[0], ops[1], rawOut);
}
static void propLossKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const symGroupedWeight_t *groupedWeight = (const symGroupedWeight_t *)ctx;
    if (groupedWeight != NULL) {
        linearCalcPropLossSymInt32Grouped(ops[1], ops[0], rawOut, groupedWeight, rs);
    } else {
        linearCalcPropLossSymInt32(ops[1], ops[0], rawOut);
    }
}
static void weightGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsBfp(ops[0], ops[1], rawOut);
}
static void biasGradKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsBfp(ops[0], rawOut);
}
static void weightGradKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsBfpRaw(ops[0], ops[1], rawOut);
}
static void biasGradKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                 const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsBfpRaw(ops[0], rawOut);
}
static void propLossKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcPropLossBfp(ops[0], ops[1], rawOut);
}

/* The integer grad path (BFP_RAW_BACKWARD_ACC): an ARITH_BFP slot into a BFP
 * grad under OUT_ACC_DYNAMIC_RESCALE. FLOAT32 grads and the fixed grid keep
 * the float bridge (BFP_RAW_NONE). */
static bool linearGradIntAcc(arithmetic_t math, const tensor_t *grad, outputMode_t mode) {
    return math.type == ARITH_BFP && grad->quantization->type == BFP &&
           mode == OUT_ACC_DYNAMIC_RESCALE;
}

/* The factory and the deserializer admit only these three arithmetic types. */
static opKernelFn_t linearBackwardKernelForArithmetic(arithmetic_t math, opKernelFn_t floatKernel,
                                                      opKernelFn_t symKernel,
                                                      opKernelFn_t bfpKernel) {
    if (math.type == ARITH_SYM_INT32) {
        return symKernel;
    }
    return math.type == ARITH_BFP ? bfpKernel : floatKernel;
}

void linearBackward(layer_t *linearLayer, tensor_t *forwardInput, tensor_t *loss,
                    tensor_t *propLoss, const layerExec_t *exec) {
    rngKey_t opKey;
    linearConfig_t *cfg = linearLayer->config->linear;
    tensor_t *weights = getParamFromParameter(cfg->weights);

    size_t lossDims[2];
    size_t lossOrder[2];
    shape_t lossExpected = {.dimensions = lossDims, .orderOfDimensions = lossOrder};
    linearCalcOutputShape(linearLayer, forwardInput->shape, &lossExpected);
    /* Dense, not logical: Matmul (weight-grad, dx) would honour a transposed
     * loss view, but the bias-grad kernels sum it by raw flat index. */
    layerRequireDenseWire(loss, &lossExpected, "Linear backward", "loss");
    if (propLoss != NULL) {
        size_t inDims[2] = {getDimensionsByIndex(forwardInput, 0),
                            getDimensionsByIndex(forwardInput, 1)};
        size_t inOrder[2] = {0, 1};
        shape_t inExpected = {
            .numberOfDimensions = 2, .dimensions = inDims, .orderOfDimensions = inOrder};
        layerRequireDenseWire(propLoss, &inExpected, "Linear backward", "propLoss");
    }

    /* BFP-stored weights anchor FLOAT32 operand staging widths; only ops that run need them.
     * The stack template lives through all three executeOp calls; executeOp owns exponents
     * and applies each operation's rounding. Zero-init keeps it inert when no running op is
     * ARITH_BFP, since the .bfpStage entries below point at it unconditionally. */
    bool runsGrads = !cfg->frozen;
    bool anyBfpBackward = (runsGrads && cfg->weightGradMath.type == ARITH_BFP) ||
                          (runsGrads && cfg->bias != NULL && cfg->biasGradMath.type == ARITH_BFP) ||
                          (propLoss != NULL && cfg->propLossMath.type == ARITH_BFP);
    bfpQConfig_t stage = {0};
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

    if (runsGrads) {
        tensor_t *weightGrad = getGradFromParameter(cfg->weights);
        bool wIntAcc = linearGradIntAcc(cfg->weightGradMath, weightGrad, cfg->weightGradAccMode);
        executeOp(
            &(opSpec_t){
                .kernel = linearBackwardKernelForArithmetic(
                    cfg->weightGradMath, weightGradKernelFloat, weightGradKernelSym,
                    wIntAcc ? weightGradKernelBfpRaw : weightGradKernelBfp),
                .inputs = (tensor_t *[]){loss, forwardInput},
                .nInputs = 2,
                .arithmetic = cfg->weightGradMath,
                .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, LINEAR_OP_WEIGHT_GRAD, &opKey),
                .mode = cfg->weightGradAccMode,
                .bfpStage = {loss->quantization->type == FLOAT32 ? &stage : NULL,
                             forwardInput->quantization->type == FLOAT32 ? &stage : NULL, NULL},
                .bfpRawOut = wIntAcc ? BFP_RAW_BACKWARD_ACC : BFP_RAW_NONE,
            },
            weightGrad);

        if (cfg->bias != NULL) {
            tensor_t *biasGrad = getGradFromParameter(cfg->bias);
            bool bIntAcc = linearGradIntAcc(cfg->biasGradMath, biasGrad, cfg->biasGradAccMode);
            executeOp(
                &(opSpec_t){
                    .kernel = linearBackwardKernelForArithmetic(
                        cfg->biasGradMath, biasGradKernelFloat, biasGradKernelSym,
                        bIntAcc ? biasGradKernelBfpRaw : biasGradKernelBfp),
                    .inputs = (tensor_t *[]){loss},
                    .nInputs = 1,
                    .arithmetic = cfg->biasGradMath,
                    .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, LINEAR_OP_BIAS_GRAD, &opKey),
                    .mode = cfg->biasGradAccMode,
                    .bfpStage = {loss->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
                    .bfpRawOut = bIntAcc ? BFP_RAW_BACKWARD_ACC : BFP_RAW_NONE,
                },
                biasGrad);
        }
    }

    /* NULL propLoss requests parameter gradients only. */
    if (propLoss != NULL) {
        /* Grouped weights require both grouped-kernel routing and prologue unpacking. The
         * weight is operand 1, encoded as groupedSymOperandPos = 2. */
        symQConfig_t asymWeightView; /* lifetime: this frame (SymGroupedWeight.h view doc) */
        symGroupedWeight_t groupedWeightCtx;
        const symGroupedWeight_t *groupedWeight =
            symGroupedWeightOrNull(weights, &asymWeightView, &groupedWeightCtx);
        bool grouped = groupedWeight != NULL;

        executeOp(
            &(opSpec_t){
                .kernel = linearBackwardKernelForArithmetic(cfg->propLossMath, propLossKernelFloat,
                                                            propLossKernelSym, propLossKernelBfp),
                .ctx = groupedWeight,
                .inputs = (tensor_t *[]){loss, weights},
                .nInputs = 2,
                .arithmetic = cfg->propLossMath,
                .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, LINEAR_OP_PROP_LOSS, &opKey),
                .mode = OUT_WRITE,
                .groupedSymOperandPos = grouped ? 2 : 0,
                /* Under ARITH_BFP the weights are BFP-stored (gate above): borrowed. */
                .bfpStage = {loss->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
                /* ARITH_BFP dx is integer end to end: the BFP_RAW, re-blocked
                 * into a BFP dx wire (or dequantized into a FLOAT32 one). */
                .bfpRawOut =
                    cfg->propLossMath.type == ARITH_BFP ? BFP_RAW_BACKWARD_DX : BFP_RAW_NONE,
            },
            propLoss);
    }
}

void linearCalcOutputShape(layer_t *linearLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 2) {
        PRINT_ERROR("Linear layer expects 2D input, got %zuD", inputShape->numberOfDimensions);
        exit(1);
    }

    /* Logical, not physical: Matmul reads the input order-aware, and the
     * backward propLoss gate takes the input's logical dims too. */
    tensor_t inputView = {.shape = inputShape};
    size_t batchSize = getDimensionsByIndex(&inputView, 0);

    linearConfig_t *cfg = linearLayer->config->linear;
    shape_t *weightShape = cfg->weights->param->shape;
    size_t outFeatures = weightShape->dimensions[0];
    size_t inFeatures = weightShape->dimensions[1];
    size_t gotFeatures = getDimensionsByIndex(&inputView, 1);
    if (gotFeatures != inFeatures) {
        PRINT_ERROR("Linear: input must have %zu features (logical dim 1), got %zu", inFeatures,
                    gotFeatures);
        exit(1);
    }

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outFeatures;

    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
