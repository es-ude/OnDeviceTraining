#define SOURCE_FILE "SOFTMAX"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "BfpSoftmaxExp.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Softmax.h"
#include "TensorConversion.h"

void softmaxInitConfig(softmaxConfig_t *softmaxConfig, quantization_t *forwardQ,
                       quantization_t *backwardQ) {
    softmaxConfig->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    softmaxConfig->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    softmaxConfig->outputQ = forwardQ;
    softmaxConfig->propLossQ = backwardQ;
    softmaxConfig->bfpExpShiftRounding = BFP_SHIFT_TRUNC;
}

void softmaxInitLayer(layerConfig_t *softmaxConfig, layer_t *softmaxLayer) {
    softmaxLayer->type = SOFTMAX;
    softmaxLayer->config = softmaxConfig;
}

/* BFP epic PR6 Task 2: extracted verbatim from the forward kernel's body
 * (behavior-identical refactor) so the backward arms can recompute the
 * softmax OUTPUT from the layer INPUT they are actually handed (P6-1) --
 * Task 4's float kernel and Task 5's SYM arm reuse this too. Max by strict
 * `>` (first-wins on ties), expf, float sum in index order, divide. */
static void softmaxValuesFloat(const float *x, float *s, size_t n) {
    // 1. find max
    float max = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    // 2. exp and sum
    float sum = 0.f;
    for (size_t i = 0; i < n; i++) {
        float e = expf(x[i] - max);
        s[i] = e;
        sum += e;
    }

    // 3. normalize
    for (size_t i = 0; i < n; i++) {
        s[i] /= sum;
    }
}

/* Softmax's real compute is always float (numerically stable max-shifted exp);
 * SYM_INT32 forwardMath only ever meant "convert in, compute in float, convert
 * out" (never native SYM arithmetic like Linear/Conv), so the funnel's
 * prologue/epilogue perform that conversion automatically. Arithmetic is
 * hardcoded ARITH_FLOAT32 here on purpose — forwardMath no longer selects a
 * compute path, it only declares the layer's storage dtype. */
static void softmaxForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                 const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *input = ops[0];
    size_t count = calcNumberOfElementsByTensor(input);

    float *x = (float *)input->data;
    float *y = (float *)rawOut->data;

    softmaxValuesFloat(x, y, count);
}

/* BFP epic PR6 Task 4 (P6-2..P6-5): the native ARITH_BFP softmax -- numerics
 * spec .superpowers/sdd/2026-09-08-bfp-pr6-softmax/numerics-spec.md steps 1-5
 * (normative; the goldgen mirrors this function statement for statement).
 * Operands arrive in the funnel's unpacked-BFP scratch form (int32 mantissa
 * codes + live bfpQConfig_t). `mode` is the layer's bfpExpShiftRounding knob:
 * it governs ONLY the integer right-shift sites here (the alignment shift and
 * the >>z inside bfpIExpQ) -- staging and the OUT_WRITE pack keep the normal
 * roundingMode_t machinery. Raw out is FLOAT32 (D7). Task 5's backward
 * recompute path calls this helper verbatim. */
static void softmaxValuesBfp(const tensor_t *input, bfpShiftRounding_t mode, float *sOut) {
    /* n == 0 handled by caller. */
    const bfpQConfig_t *qC = input->quantization->qConfig;
    const int32_t bias = bfpExponentBias(qC);
    const int32_t *m = (const int32_t *)input->data;
    /* calcNumberOfElementsByTensor takes a non-const tensor_t* but only
     * reads; the cast keeps this helper's const-view contract for Task 5's
     * backward recompute caller. */
    const size_t n = calcNumberOfElementsByTensor((tensor_t *)input);

    /* (1) Max (P6-4): ldexpf compare, strict > keeps the FIRST max; remember
     * the winner's mantissa and UNBIASED exponent. */
    int32_t mMax = m[0];
    int32_t eMax = (int32_t)qC->exponents[bfpGroupOf(qC, 0)] - bias;
    float xMax = ldexpf((float)m[0], (int)eMax);
    for (size_t i = 1; i < n; i++) {
        const int32_t ei = (int32_t)qC->exponents[bfpGroupOf(qC, i)] - bias;
        const float xi = ldexpf((float)m[i], (int)ei);
        if (xi > xMax) {
            xMax = xi;
            mMax = m[i];
            eMax = ei;
        }
    }

    /* (2) Align onto the work-grid-or-coarser common grid (THE lossy site --
     * at most ONE rounded shift per element). When the storage grid is FINER
     * than the work grid (sigma < 0), `down` folds the extra descent into
     * that same shift -- never two chained roundings. The net shift si MAY be
     * NEGATIVE (spec amended 2026-09-08, fix round 1): eMax follows the
     * SIGNED argmax while block exponents follow the block ABSMAX, so a
     * negative-dominated block is legitimately coarser than the argmax block
     * (ei > eMax) -- such elements take an EXACT saturating LEFT shift. */
    const int32_t sigma = eMax + BFP_SOFTMAX_EXP_FRAC_BITS;
    const uint32_t down = (sigma < 0) ? (uint32_t)(-sigma) : 0u;
    const int32_t mMaxW = bfpShiftRightRounded(mMax, down, mode);
    float sum = 0.f;
    for (size_t i = 0; i < n; i++) {
        const int32_t ei = (int32_t)qC->exponents[bfpGroupOf(qC, i)] - bias;
        const int32_t si = (eMax - ei) + (int32_t)down;
        int32_t aligned;
        if (si >= 0) {
            aligned = bfpShiftRightRounded(m[i], (uint32_t)si, mode);
        } else {
            const uint32_t up = (uint32_t)(-si);
            if (m[i] < 0 && (up >= 31u || m[i] < -(INT32_MAX >> up))) {
                /* Saturate: the element's true value sits below the
                 * exp-underflow floor up to a residual <= exp(-16); the
                 * INT32_MIN/2 sentinel rides the qT clamp + thr/QFLOOR
                 * handling below (headroom: |mMaxW| <= 2^30 keeps the qT
                 * subtraction inside int32). The m[i] < -(INT32_MAX >> up)
                 * form avoids negating INT32_MIN. */
                aligned = INT32_MIN / 2;
            } else {
                /* Exact left shift, unsigned image (no signed-shift UB). For
                 * m[i] > 0 it provably cannot overflow: x_i <= x_max gives
                 * m_i * 2^(ei - eMax) <= mMax, and -si <= ei - eMax, so
                 * aligned <= mMax -- with m_i >= 1 that also bounds
                 * ei - eMax <= 30, i.e. up stays a valid shift count
                 * (comment, not assert -- spec step 2). */
                aligned = (int32_t)((uint32_t)m[i] << up);
            }
        }
        int32_t qT = aligned - mMaxW;
        if (qT > 0) {
            /* min(qT, 0): load-bearing under SR -- deterministic modes are
             * monotone (qT <= 0 by construction), but SR jitter can yield +1,
             * which would reach bfpIExpQ as qW > 0 and make z negative; such
             * jitter maps to exp(0) instead. */
            qT = 0;
        }
        /* (3) Work-grid promotion (exact). sigma >= 31: any nonzero deficit
         * at scale >= 2^17 is >> 22.18 -> exact underflow. sigma >= 0:
         * saturating exact left shift -- the thr guard keeps it inside int32
         * (|qT| <= ceil(363392/2^sigma) => |qW| < 363392 + 2^sigma); the
         * unsigned re-shift is the two's-complement image of qT * 2^sigma
         * without the signed-left-shift UB (the BfpSoftmaxExp.c SR idiom).
         * sigma < 0: qT is already ON the work grid from step (2). */
        int32_t qW;
        if (sigma >= 31) {
            qW = (qT < 0) ? BFP_SOFTMAX_QFLOOR : 0;
        } else if (sigma >= 0) {
            const int32_t thr = -((363392 + (1 << sigma) - 1) >> sigma);
            qW = (qT <= thr) ? BFP_SOFTMAX_QFLOOR : (int32_t)((uint32_t)qT << (uint32_t)sigma);
        } else {
            qW = qT;
        }
        /* (4) Integer core (lossy site 3 is the >>z inside); (5) float
         * boundary (P6-5): S_out = A * 2^-28, float sum in index order. */
        const int32_t qe = bfpIExpQ(qW, mode);
        const float e = ldexpf((float)qe * BFP_SOFTMAX_A, -28);
        sOut[i] = e;
        sum += e;
    }
    for (size_t i = 0; i < n; i++) {
        sOut[i] /= sum;
    }
}

static void softmaxForwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                    const void *ctx) {
    (void)n;
    (void)auxOut;
    const softmaxConfig_t *cfg = ctx;
    tensor_t *input = ops[0];
    size_t count = calcNumberOfElementsByTensor(input);
    if (count == 0) {
        return;
    }
    validateBfpQConfigShape(input->quantization->qConfig, count);
    softmaxValuesBfp(input, cfg->bfpExpShiftRounding, (float *)rawOut->data);
}

void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output) {
    softmaxConfig_t *cfg = softmaxLayer->config->softmax;
    switch (cfg->forwardMath.type) {
    case ARITH_BFP: {
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->outputQ, "Softmax forward");
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->forwardMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        executeOp(&(opSpec_t){.kernel = softmaxForwardKernelBfp,
                              .ctx = cfg,
                              .inputs = (tensor_t *[]){input},
                              .nInputs = 1,
                              .arithmetic = cfg->forwardMath,
                              .mode = OUT_WRITE,
                              .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL}},
                  output);
        return;
    }
    case ARITH_FLOAT32:
    case ARITH_SYM_INT32:
        /* P6-7: declared SYM math keeps its documented fake-quant meaning --
         * compute in float, funnel prologue/epilogue convert. The opSpec
         * arithmetic stays HARDCODED ARITH_FLOAT32 (NOT cfg->forwardMath):
         * ARITH_SYM_INT32 would make the prologue unpack into int32 scratch
         * the float kernel misreads through a float* cast. */
        executeOp(&(opSpec_t){.kernel = softmaxForwardKernel,
                              .inputs = (tensor_t *[]){input},
                              .nInputs = 1,
                              .arithmetic =
                                  (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                              .mode = OUT_WRITE},
                  output);
        return;
    default:
        PRINT_ERROR("Softmax forward: declared forwardMath %d not implemented",
                    (int)cfg->forwardMath.type);
        exit(1);
    }
}

/* BFP epic PR2 Task 8: softmaxBackward is the fourth outside-funnel site (after
 * Relu/Dropout/Flatten). Its ARITH_FLOAT32 arm raw-casts all three wires to
 * float* with no dtype check at all, and it is selected by the layer's DECLARED
 * propLossMath -- so a BFP dx wire whose math slot is pinned to (or, before the
 * Task 9 flip, derived as) ARITH_FLOAT32 lands straight in the raw casts: a 4x
 * heap over-read on input/loss and an over-write into the packed propLoss buffer.
 * That became reachable only with this task's initGradTensor BFP arm (before it,
 * a BFP propLossQ died in the allocator's default arm).
 *
 * Forward needs no guard: it runs inside executeOp, whose prologue/epilogue
 * convert both ways. PR6, not PR4 — softmax BFP semantics belong to research
 * package II. */

static void softmaxBackwardFloat(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    size_t n = calcNumberOfElementsByTensor(input);

    float *x = (float *)input->data;
    float *dLds = (float *)loss->data;
    float *dLdx = (float *)propLoss->data;

    /* P6-1 root fix: the training loop hands every backward the layer INPUT
     * (logits), not the softmax OUTPUT the Jacobian needs -- recompute it. */
    float s[n];
    softmaxValuesFloat(x, s, n);

    float dot = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot += s[i] * dLds[i];
    }

    for (size_t i = 0; i < n; i++) {
        dLdx[i] = s[i] * (dLds[i] - dot);
    }
}

static void softmaxBackwardSymInt32(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    size_t inputSize = calcNumberOfElementsByTensor(input);

    tensor_t inputFloat;
    quantization_t inputFloatQ;
    initFloat32Quantization(&inputFloatQ);
    uint8_t inputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(inputFloatData, &inputFloatQ, input, &inputFloat);
    convertTensor(input, &inputFloat);

    tensor_t lossFloat;
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);
    uint8_t lossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(lossFloatData, &lossFloatQ, loss, &lossFloat);
    convertTensor(loss, &lossFloat);

    tensor_t propLossFloat;
    quantization_t propLossFloatQ;
    initFloat32Quantization(&propLossFloatQ);
    uint8_t propLossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(propLossFloatData, &propLossFloatQ, propLoss, &propLossFloat);
    convertTensor(propLoss, &propLossFloat);

    float *dLds = (float *)lossFloat.data;
    float *dLdx = (float *)propLossFloat.data;

    /* P6-1 root fix: same recompute as the float arm -- the dequantized
     * inputFloat is the layer INPUT (logits), not the softmax OUTPUT. */
    float s[inputSize];
    softmaxValuesFloat((float *)inputFloat.data, s, inputSize);

    float dot = 0.0f;
    for (size_t i = 0; i < inputSize; i++) {
        dot += s[i] * dLds[i];
    }

    for (size_t i = 0; i < inputSize; i++) {
        dLdx[i] = s[i] * (dLds[i] - dot);
    }

    convertTensor(&propLossFloat, propLoss);
}

void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    /* Before the dispatch (the Relu placement): all three wires are dereferenced
     * by whichever arm runs, and the check is on STORAGE dtype, not the declared
     * arithmetic that selects the arm. */
    bfpRequireNoBfpWire(input, "Softmax backward (input)");
    bfpRequireNoBfpWire(loss, "Softmax backward (loss)");
    bfpRequireNoBfpWire(propLoss, "Softmax backward (propLoss)");

    switch (softmaxLayer->config->softmax->propLossMath.type) {
    case ARITH_FLOAT32:
        softmaxBackwardFloat(input, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        softmaxBackwardSymInt32(input, loss, propLoss);
        break;
    default:
        PRINT_ERROR("Softmax backward: declared propLossMath %d not implemented "
                    "(FLOAT32/SYM_INT32 only) -- native BFP softmax arrives with epic PR6",
                    (int)softmaxLayer->config->softmax->propLossMath.type);
        exit(1);
    }
}

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape) {
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
