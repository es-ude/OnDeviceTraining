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
#include "LayerExec.h"
#include "Softmax.h"
#include "TensorConversion.h"
#include "TensorRows.h"

void softmaxInitConfig(softmaxConfig_t *softmaxConfig, quantization_t *forwardQ,
                       quantization_t *backwardQ) {
    softmaxConfig->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    softmaxConfig->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    softmaxConfig->outputQ = forwardQ;
    softmaxConfig->propLossQ = backwardQ;
    softmaxConfig->ownsQuantizations = false;
    softmaxConfig->bfpExpShiftRounding = BFP_SHIFT_TRUNC;
}

void softmaxInitLayer(layerConfig_t *softmaxConfig, layer_t *softmaxLayer) {
    softmaxLayer->type = SOFTMAX;
    softmaxLayer->config = softmaxConfig;
}

static bool softmaxIsArithmetic(arithmeticType_t type) {
    return type == ARITH_FLOAT32 || type == ARITH_SYM_INT32 || type == ARITH_BFP;
}

void softmaxValidateConfig(const softmaxConfig_t *cfg, const char *who) {
    if (!softmaxIsArithmetic(cfg->forwardMath.type) ||
        !softmaxIsArithmetic(cfg->propLossMath.type)) {
        PRINT_ERROR("%s: forwardMath and propLossMath must be ARITH_FLOAT32, ARITH_SYM_INT32 or "
                    "ARITH_BFP (got %d, %d)",
                    who, (int)cfg->forwardMath.type, (int)cfg->propLossMath.type);
        exit(1);
    }
    /* The ARITH_BFP arms take their staging widths from the produced-wire config. */
    if (cfg->forwardMath.type == ARITH_BFP && cfg->outputQ->type != BFP) {
        PRINT_ERROR("%s: ARITH_BFP forwardMath requires a BFP-typed outputQ", who);
        exit(1);
    }
    if (cfg->propLossMath.type == ARITH_BFP && cfg->propLossQ->type != BFP) {
        PRINT_ERROR("%s: ARITH_BFP propLossMath requires a BFP-typed propLossQ", who);
        exit(1);
    }
}

/* Shared by the forward kernel and both non-BFP backward arms, which recompute s from the
 * logits. Max by strict `>` (first wins on ties), expf, float sum in index order, divide;
 * softmaxValuesBfp is the ARITH_BFP twin of this contract. */
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

/* Float compute for both ARITH_FLOAT32 and ARITH_SYM_INT32 forwardMath: the funnel's prologue and
 * epilogue convert a SYM wire. */
static void softmaxForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                 const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *input = ops[0];
    size_t rows;
    size_t rowLen;
    tensorShapeRows(input->shape, &rows, &rowLen);
    if (rowLen == 0) {
        return; /* empty rows: no max to read */
    }

    float *x = (float *)input->data;
    float *y = (float *)rawOut->data;

    for (size_t r = 0; r < rows; r++) {
        softmaxValuesFloat(x + r * rowLen, y + r * rowLen, rowLen);
    }
}

/* The native ARITH_BFP softmax's integer steps 1-4
 * (docs/conventions/arithmetic-bfp.md §5.9; the goldgen mirrors this function
 * statement for statement). Writes
 * the i-exp code qe_i (scale A * 2^-28, >= 0) of every element to `qe`; the
 * forward normalises them in integers (softmaxForwardBfpRaw), the backward's
 * recompute in float32 (softmaxValuesBfp), so both see the same codes.
 * Operands arrive in the funnel's unpacked-BFP scratch form (int32 mantissa
 * codes + live bfpQConfig_t). `mode` is the layer's bfpExpShiftRounding knob:
 * it governs ONLY the integer right-shift sites here (the alignment shift and
 * the >>z inside bfpIExpQ) -- staging and the OUT_WRITE pack keep the normal
 * roundingMode_t machinery. */
static void softmaxExpCodesBfp(const tensor_t *input, bfpShiftRounding_t mode, int32_t *qe,
                               rngStream_t *shift) {
    /* n == 0 handled by caller. */
    const bfpQConfig_t *qC = input->quantization->qConfig;
    if (qC->mantissaBits > 16) {
        PRINT_ERROR("Softmax BFP: mantissaBits %u > 16 -- the exact max compare needs "
                    "|code| <= 2^15",
                    (unsigned)qC->mantissaBits);
        exit(1);
    }
    const int32_t bias = bfpExponentBias(qC);
    bfpGroupRange_t qRange = {0};
    const int32_t *m = (const int32_t *)input->data;
    /* calcNumberOfElementsByTensor takes a non-const tensor_t* but only
     * reads; the cast keeps this helper's const-view contract. */
    const size_t n = calcNumberOfElementsByTensor((tensor_t *)input);

    /* (1) Max: the exact (code, E) compare, strict > keeps the FIRST
     * max; remember the winner's mantissa and UNBIASED exponent -- the
     * alignment grid below. A code of an ALL-ZERO group never decides it
     * (§13's zero-group rule): such a group stores the zero state (E = 0),
     * an exponent unrelated to the data, and a 0 there that beats every live
     * (negative) value would align the live logits onto 2^0. When that
     * happens the max is 0 on the TOP live group's grid (every live element
     * then shifts right, as under a live max in that group). All codes zero:
     * the first element's grid, as before (every aligned value is 0). */
    int32_t mMax = 0;
    int32_t eMax = (int32_t)qC->exponents[bfpGroupAt(&qRange, qC, 0)] - bias;
    bool haveMax = false;
    bool deadZero = false;
    int32_t liveTop = INT32_MIN;
    size_t curGroup = SIZE_MAX;
    bool curLive = false;
    for (size_t i = 0; i < n; i++) {
        const size_t g = bfpGroupAt(&qRange, qC, i);
        const int32_t ei = (int32_t)qC->exponents[g] - bias;
        if (g != curGroup) {
            curGroup = g;
            size_t start = qC->groupSize == 0 ? 0 : g * qC->groupSize;
            size_t end = qC->groupSize == 0 ? n : start + qC->groupSize;
            end = end < n ? end : n;
            curLive = false;
            for (size_t j = start; j < end && !curLive; j++) {
                curLive = m[j] != 0;
            }
            if (curLive && ei > liveTop) {
                liveTop = ei;
            }
        }
        if (!curLive) {
            deadZero = true;
            continue;
        }
        if (!haveMax || bfpValueGreater(m[i], (int)ei, mMax, (int)eMax)) {
            mMax = m[i];
            eMax = ei;
            haveMax = true;
        }
    }
    if (haveMax && deadZero && mMax < 0) {
        mMax = 0;
        eMax = liveTop;
    }

    /* (2) Combine alignment and work-grid descent into one rounded shift; negative shifts are
     * exact unless saturated. A shift is negative when a negative-dominated block is coarser
     * than the signed argmax's block (ei > eMax). */
    const int32_t sigma = eMax + BFP_SOFTMAX_EXP_FRAC_BITS;
    const uint32_t down = (sigma < 0) ? (uint32_t)(-sigma) : 0u;
    const int32_t mMaxW = bfpShiftRightRounded(mMax, down, mode, shift);
    for (size_t i = 0; i < n; i++) {
        const int32_t ei = (int32_t)qC->exponents[bfpGroupAt(&qRange, qC, i)] - bias;
        const int32_t si = (eMax - ei) + (int32_t)down;
        int32_t aligned;
        if (si >= 0) {
            aligned = bfpShiftRightRounded(m[i], (uint32_t)si, mode, shift);
        } else {
            const uint32_t up = (uint32_t)(-si);
            if (m[i] < 0 && (up >= 31u || m[i] < -(INT32_MAX >> up))) {
                /* Saturate negative overflow: the value sits below the exp-underflow floor.
                 * |mMaxW| <= 2^30 keeps qT inside int32; the form avoids negating INT32_MIN. */
                aligned = INT32_MIN / 2;
            } else {
                /* Exact unsigned left shift. m[i] > 0 cannot overflow (x_i <= x_max bounds
                 * aligned <= mMax); m[i] == 0 is unbounded in `up`, so handle zero separately
                 * when the shift count exceeds the word width. */
                aligned = (up >= 31u) ? 0 : (int32_t)((uint32_t)m[i] << up);
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
        /* (3) Promote exactly to the work grid, saturating deficits below the exponential floor
         * (thr keeps the unsigned left shift inside int32). sigma < 0: step (2) already landed
         * on the work grid. */
        int32_t qW;
        if (sigma >= 31) {
            qW = (qT < 0) ? BFP_SOFTMAX_QFLOOR : 0;
        } else if (sigma >= 0) {
            const int32_t thr = -((-BFP_SOFTMAX_QFLOOR + (1 << sigma) - 1) >> sigma);
            qW = (qT <= thr) ? BFP_SOFTMAX_QFLOOR : (int32_t)((uint32_t)qT << (uint32_t)sigma);
        } else {
            qW = qT;
        }
        /* (4) Integer core (lossy site 3 is the >>z inside). */
        qe[i] = bfpIExpQ(qW, mode, shift);
    }
}

/* Step 5 of the integer forward, in place: q_i (the i-exp codes, >= 0) ->
 * round(q_i / sum q * 2^F), F = BFP_SOFTMAX_RAW_FRAC_BITS. The common scale
 * A * 2^-28 cancels, so no constant multiplies.
 *  - Sum: the true sum T reaches n * QE_MAX > 2^31 for n >= 3, so it is kept
 *    as T = hi * 2^p + lo (lo < 2^p, one carry per term) with p the smallest
 *    p >= 0 such that n * QE_MAX <= INT32_MAX * 2^p (so hi fits); the divisor
 *    is S = round_half_up(T / 2^p), exact up to that one rounding.
 *  - Divide: mant_i = round_half_up(q_i * 2^(F-p) / S), a restoring division
 *    of 32 quotient bits (F + 1 plus the half bit), no hardware divide. It
 *    starts from r = q_i >> (p + 1), which must be < S for the quotient to
 *    fit 32 bits: S >= T / 2^p - 1/2 and T > 2^p (the argmax's qe is ~QE_MAX)
 *    give it, and it is CHECKED once per call (qMax). r < S <= INT32_MAX
 *    keeps 2r + 1 inside uint32. |mant| <= 2^F (1 + 2^(p-1)/T) + 1 < 2^31.
 * Rounding is HALF_AWAY (= half up, all values >= 0), never SR and never the
 * knob's: the re-block owns the op's rounding mode. */
/* The largest n with n * QE_MAX <= INT32_MAX * 2^p, per p (compile-time
 * constants, so choosing p costs no 64-bit arithmetic at run time). */
#define SM_MAX_TERMS(p) ((uint32_t)(((uint64_t)INT32_MAX << (p)) / BFP_SOFTMAX_QE_MAX))
static const uint32_t kSoftmaxMaxTerms[29] = {
    SM_MAX_TERMS(0),  SM_MAX_TERMS(1),  SM_MAX_TERMS(2),  SM_MAX_TERMS(3),  SM_MAX_TERMS(4),
    SM_MAX_TERMS(5),  SM_MAX_TERMS(6),  SM_MAX_TERMS(7),  SM_MAX_TERMS(8),  SM_MAX_TERMS(9),
    SM_MAX_TERMS(10), SM_MAX_TERMS(11), SM_MAX_TERMS(12), SM_MAX_TERMS(13), SM_MAX_TERMS(14),
    SM_MAX_TERMS(15), SM_MAX_TERMS(16), SM_MAX_TERMS(17), SM_MAX_TERMS(18), SM_MAX_TERMS(19),
    SM_MAX_TERMS(20), SM_MAX_TERMS(21), SM_MAX_TERMS(22), SM_MAX_TERMS(23), SM_MAX_TERMS(24),
    SM_MAX_TERMS(25), SM_MAX_TERMS(26), SM_MAX_TERMS(27), SM_MAX_TERMS(28)};
#undef SM_MAX_TERMS

static void softmaxNormalizeRaw(int32_t *q, size_t n) {
    unsigned p = 0;
    while (p <= 28u && n > kSoftmaxMaxTerms[p]) {
        p++;
    }
    if (p > 28u) {
        PRINT_ERROR("Softmax BFP: %zu elements are beyond the int32 normalisation", n);
        exit(1);
    }
    const uint32_t mask = (1u << p) - 1u;
    uint32_t hi = 0;
    uint32_t lo = 0;
    uint32_t qMax = 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t v = (uint32_t)q[i];
        hi += v >> p;
        lo += v & mask;
        if (lo > mask) {
            hi++;
            lo -= mask + 1u;
        }
        if (v > qMax) {
            qMax = v;
        }
    }
    const uint32_t s = hi + ((p > 0u && lo >= (1u << (p - 1u))) ? 1u : 0u);
    if ((qMax >> (p + 1u)) >= s) {
        PRINT_ERROR("Softmax BFP: normalisation divisor %u below the largest code %u >> %u "
                    "(no i-exp code near exp(0))",
                    (unsigned)s, (unsigned)qMax, p + 1u);
        exit(1);
    }
    const unsigned zeroBits = BFP_SOFTMAX_RAW_FRAC_BITS - p + 1u;
    for (size_t i = 0; i < n; i++) {
        const uint32_t v = (uint32_t)q[i];
        uint32_t r = v >> (p + 1u);
        uint32_t quot = 0;
        for (unsigned b = p + 1u; b-- > 0u;) {
            r = (r << 1) | ((v >> b) & 1u);
            quot <<= 1;
            if (r >= s) {
                r -= s;
                quot |= 1u;
            }
        }
        for (unsigned b = 0; b < zeroBits; b++) {
            r <<= 1;
            quot <<= 1;
            if (r >= s) {
                r -= s;
                quot |= 1u;
            }
        }
        q[i] = (int32_t)((quot >> 1) + (quot & 1u));
    }
}

void softmaxForwardBfpRaw(const tensor_t *input, bfpShiftRounding_t mode, rngStream_t *shift,
                          tensor_t *rawOut) {
    const size_t n = calcNumberOfElementsByTensor((tensor_t *)input);
    bfpRawQConfig_t *rawQC = rawOut->quantization->qConfig;
    bfpRawRequireOutput(rawQC, n, "softmaxForwardBfpRaw");
    if (n == 0) {
        return;
    }
    int32_t *mant = (int32_t *)rawOut->data;
    softmaxExpCodesBfp(input, mode, mant, shift);
    softmaxNormalizeRaw(mant, n);
    for (size_t t = 0; t < rawQC->numGroups; t++) {
        rawQC->exponents[t] = -BFP_SOFTMAX_RAW_FRAC_BITS;
    }
}

/* The BFP kernels' ctx: the layer config plus the stream of the shift sites
 * in softmaxValuesBfp. Both phases hand the FORWARD key's shift stream, so the
 * backward's recompute of s replays the forward's draws. */
typedef struct {
    const softmaxConfig_t *cfg;
    rngStream_t *shift;
} softmaxKernelCtx_t;

/* A BFP_SHIFT_SR softmax draws from the forward-phase shift stream; a key-less call has none. */
static void softmaxRequireShiftStream(const softmaxConfig_t *cfg, const rngStream_t *shift) {
    if (cfg->bfpExpShiftRounding == BFP_SHIFT_SR && shift == NULL) {
        PRINT_ERROR("Softmax: BFP_SHIFT_SR needs a key (run it through inferenceWithLoss with "
                    "call->key or a training step)");
        exit(1);
    }
}

static void softmaxForwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                    const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    const softmaxKernelCtx_t *kctx = ctx;
    softmaxForwardBfpRaw(ops[0], kctx->cfg->bfpExpShiftRounding, kctx->shift, rawOut);
}

/* The native BFP pipeline runs one alignment grid and one partition sum over the whole wire. */
static void softmaxBfpRequireSingleRow(const tensor_t *t, const char *where) {
    size_t rows;
    size_t rowLen;
    tensorShapeRows(t->shape, &rows, &rowLen);
    if (rows > 1) {
        PRINT_ERROR("%s: ARITH_BFP softmax takes a single row, got %zu rows of %zu elements", where,
                    rows, rowLen);
        exit(1);
    }
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    SOFTMAX_OP_FORWARD = 0,
    SOFTMAX_OP_PROP_LOSS = 1,
};

/* For a wire walked in flat storage order at the input's offsets: it needs the input's raw dims
 * and, when multi-row, identity order. */
static void softmaxRequireWire(const tensor_t *input, const tensor_t *t, const char *where,
                               const char *what) {
    const shape_t *a = input->shape;
    const shape_t *b = t->shape;
    bool same = a->numberOfDimensions == b->numberOfDimensions;
    for (size_t d = 0; same && d < a->numberOfDimensions; d++) {
        same = a->dimensions[d] == b->dimensions[d];
    }
    if (!same) {
        PRINT_ERROR("%s: %s must have the input's rank and dims (input rank %zu, %s rank %zu)",
                    where, what, a->numberOfDimensions, what, b->numberOfDimensions);
        exit(1);
    }
    tensorRequireRowOrder(b, where);
}

/* Admits the input; the caller admits the other wires against it. */
static void softmaxRequireInput(const tensor_t *input) {
    tensorRequireRowOrder(input->shape, "Softmax");
}

void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output,
                    const layerExec_t *exec) {
    rngKey_t opKey;
    softmaxConfig_t *cfg = softmaxLayer->config->softmax;
    softmaxRequireInput(input);
    softmaxRequireWire(input, output, "Softmax forward", "output");
    switch (cfg->forwardMath.type) {
    case ARITH_BFP: {
        softmaxBfpRequireSingleRow(input, "Softmax forward");
        const bfpQConfig_t *anchor = cfg->outputQ->qConfig;
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->forwardMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        rngStream_t shiftStorage;
        softmaxKernelCtx_t kctx = {.cfg = cfg,
                                   .shift = layerExecPurposeStream(exec, RNG_TAG_FORWARD,
                                                                   RNG_TAG_SOFTMAX_SHIFT,
                                                                   &shiftStorage)};
        softmaxRequireShiftStream(cfg, kctx.shift);
        executeOp(
            &(opSpec_t){.kernel = softmaxForwardKernelBfp,
                        .ctx = &kctx,
                        .inputs = (tensor_t *[]){input},
                        .nInputs = 1,
                        .arithmetic = cfg->forwardMath,
                        .key = layerExecOpKey(exec, RNG_TAG_FORWARD, SOFTMAX_OP_FORWARD, &opKey),
                        .mode = OUT_WRITE,
                        .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL},
                        /* Forward raw contract: the backward op never sets it. */
                        .bfpRawOut = BFP_RAW_FORWARD},
            output);
        return;
    }
    case ARITH_FLOAT32:
    case ARITH_SYM_INT32:
        /* Fake-quant SYM: the op stays ARITH_FLOAT32, since ARITH_SYM_INT32 would unpack int32
         * scratch the float kernel misreads. */
        executeOp(
            &(opSpec_t){.kernel = softmaxForwardKernel,
                        .key = layerExecOpKey(exec, RNG_TAG_FORWARD, SOFTMAX_OP_FORWARD, &opKey),
                        .inputs = (tensor_t *[]){input},
                        .nInputs = 1,
                        .arithmetic =
                            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                        .mode = OUT_WRITE},
            output);
        return;
    }
}

/* The Jacobian-vector product s * (dLds - <s, dLds>) per row: the Jacobian is block-diagonal
 * over rows, so each row recomputes its own s from its own logits. */
static void softmaxVjpRows(const float *x, const float *dLds, float *dLdx, size_t rows,
                           size_t rowLen) {
    if (rowLen == 0) {
        return; /* empty rows: no max to read, no zero-length VLA */
    }
    float s[rowLen];
    for (size_t r = 0; r < rows; r++) {
        const size_t off = r * rowLen;
        softmaxValuesFloat(x + off, s, rowLen);

        float dot = 0.0f;
        for (size_t i = 0; i < rowLen; i++) {
            dot += s[i] * dLds[off + i];
        }
        for (size_t i = 0; i < rowLen; i++) {
            dLdx[off + i] = s[i] * (dLds[off + i] - dot);
        }
    }
}

/* The non-BFP arms run outside the funnel: FLOAT32 raw-casts all three wires, SYM converts
 * through input-sized scratch. Legal BFP-storage arithmetics are FLOAT32 fake-quant (funnel only)
 * and native BFP, so both arms deny BFP wires. */
static void softmaxRequireNonBfpGradWires(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    bfpRequireNoBfpWire(input, "Softmax backward (input)");
    bfpRequireNoBfpWire(loss, "Softmax backward (loss)");
    bfpRequireNoBfpWire(propLoss, "Softmax backward (propLoss)");
    softmaxRequireWire(input, loss, "Softmax backward", "loss");
    softmaxRequireWire(input, propLoss, "Softmax backward", "propLoss");
}

/* The training loop hands the backward the logits, not s; softmaxVjpRows recomputes s. */
static void softmaxBackwardFloat(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    size_t rows;
    size_t rowLen;
    tensorShapeRows(input->shape, &rows, &rowLen);
    softmaxVjpRows((const float *)input->data, (const float *)loss->data, (float *)propLoss->data,
                   rows, rowLen);
}

static void softmaxBackwardSymInt32(tensor_t *input, tensor_t *loss, tensor_t *propLoss,
                                    rngStream_t *convert) {
    size_t rows;
    size_t rowLen;
    tensorShapeRows(input->shape, &rows, &rowLen);
    size_t inputSize = calcNumberOfElementsByTensor(input);
    if (inputSize == 0) {
        return; /* no zero-length scratch VLAs */
    }

    tensor_t inputFloat;
    quantization_t inputFloatQ;
    initFloat32Quantization(&inputFloatQ);
    uint8_t inputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(inputFloatData, &inputFloatQ, input, &inputFloat);
    convertTensor(input, &inputFloat, NULL);

    tensor_t lossFloat;
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);
    uint8_t lossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(lossFloatData, &lossFloatQ, loss, &lossFloat);
    convertTensor(loss, &lossFloat, NULL);

    tensor_t propLossFloat;
    quantization_t propLossFloatQ;
    initFloat32Quantization(&propLossFloatQ);
    uint8_t propLossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(propLossFloatData, &propLossFloatQ, propLoss, &propLossFloat);

    softmaxVjpRows((const float *)inputFloat.data, (const float *)lossFloat.data,
                   (float *)propLossFloat.data, rows, rowLen);

    convertTensor(&propLossFloat, propLoss, convert);
}

/* The integer VJP of the native ARITH_BFP softmax backward (arithmetic-bfp.md
 * section 14): ONE funnel op, OUT_WRITE of the BFP_RAW dx
 * (BFP_RAW_BACKWARD_DX) anchored at propLossQ. The backward differentiates
 * the UNPACKED forward values: p is recomputed from the LOGITS by the
 * integer forward (softmaxExpCodesBfp + softmaxNormalizeRaw: Q30 codes,
 * replaying the forward's shift draws), then dx_i = p_i (g_i - d), d =
 * sum_j p_j g_j, the linear VJP in the FLOAT32 arm's order:
 *  - p'_j = p_j >> 15 half away (<= 2^15 + 1): the explicit precision cut
 *    before the dot's products, |p'_j g_j| <= 2^15 (2^15 + 1) < 2^31.
 *  - the dot on E_d = E_top - 15 + G, E_top = the max exponent over g's
 *    NONZERO codes, G = bfpFoldGuardBits(n + 1, 17, m_g): each product
 *    enters by bfpAlignPartial (left exact, right half away), index order.
 *  - D_i = g_i aligned to E_d, minus d: the n + 1 terms of G's proof bound
 *    every |D_i| below INT32_MAX.
 *  - per raw target group: both factors cut to 15 bits relative to the
 *    group's max (s_p, s_D = bitlen(max) - 15, half away), so the product
 *    fits int32 and keeps 15 bits of each factor at the group's top; raw
 *    exponent E_d + s_D + s_p - 30. An all-zero g (or group) is dx = 0.
 * Error: at most half a unit of each 15-bit cut, the dot's alignments
 * (n/2 units of 2^E_d) and p's Q30 rounding -- about 2^-15 of the group's
 * largest |dx|, below an m <= 15 wire's step (about one step at m = 16).
 * No float, no division. The caller guarantees one row and equal x/loss
 * counts. */
#define SOFTMAX_VJP_BITS 15
static void softmaxBackwardKernelBfp(tensor_t **ops, size_t nOps, tensor_t *rawOut,
                                     tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)nOps;
    (void)auxOut;
    (void)rs;
    const softmaxKernelCtx_t *kctx = ctx;
    const softmaxConfig_t *cfg = kctx->cfg;
    tensor_t *x = ops[0];
    tensor_t *dLdsT = ops[1];
    size_t n = calcNumberOfElementsByTensor(x);
    bfpRawQConfig_t *rawQC = rawOut->quantization->qConfig;
    size_t gsz = bfpRawRequireOutput(rawQC, n, "Softmax backward BFP");
    if (n == 0) {
        return;
    }
    const bfpQConfig_t *lQC = dLdsT->quantization->qConfig;

    int32_t *mant = (int32_t *)rawOut->data;
    softmaxExpCodesBfp(x, cfg->bfpExpShiftRounding, mant, kctx->shift);
    softmaxNormalizeRaw(mant, n); /* p, Q30 */

    const int32_t lBias = bfpExponentBias(lQC);
    const int32_t *gArr = (const int32_t *)dLdsT->data;
    int16_t *rawE = rawQC->exponents;
    bfpRowCursor_t cursor = {.group = 0, .groupEnd = lQC->groupSize};
    int top = bfpRowMaxExponent(&cursor, lQC, lBias, gArr, 0, n);
    if (top == BFP_NO_LIVE_EXP) {
        memset(mant, 0, n * sizeof(int32_t));
        for (size_t t = 0; t < rawQC->numGroups; t++) {
            rawE[t] = 0;
        }
        return;
    }
    const int eD = top - SOFTMAX_VJP_BITS +
                   bfpFoldGuardBits(n + 1, 17, lQC->mantissaBits, 0, "Softmax backward BFP");

    bfpGroupRange_t lRange = {0};
    int32_t dot = 0;
    for (size_t i = 0; i < n; i++) {
        const int eg = (int)lQC->exponents[bfpGroupAt(&lRange, lQC, i)] - lBias;
        const int32_t pc = bfpShrHalfAway(mant[i], SOFTMAX_VJP_BITS);
        dot += bfpAlignPartial(pc * gArr[i], eD - (eg - SOFTMAX_VJP_BITS));
    }

    size_t t = 0;
    for (size_t start = 0; start < n; start += gsz, t++) {
        const size_t end = start + gsz;
        uint32_t maxP = 0;
        uint32_t maxD = 0;
        for (size_t i = start; i < end; i++) {
            const int eg = (int)lQC->exponents[bfpGroupAt(&lRange, lQC, i)] - lBias;
            const int32_t d = bfpAlignPartial(gArr[i], eD - eg) - dot;
            const uint32_t mag = d < 0 ? 0u - (uint32_t)d : (uint32_t)d;
            maxD = mag > maxD ? mag : maxD;
            maxP = (uint32_t)mant[i] > maxP ? (uint32_t)mant[i] : maxP;
        }
        if (maxP == 0 || maxD == 0) {
            memset(mant + start, 0, gsz * sizeof(int32_t));
            rawE[t] = 0;
            continue;
        }
        const unsigned bp = bfpBitLength32(maxP);
        const unsigned bd = bfpBitLength32(maxD);
        const unsigned sp = bp > SOFTMAX_VJP_BITS ? bp - SOFTMAX_VJP_BITS : 0u;
        const unsigned sd = bd > SOFTMAX_VJP_BITS ? bd - SOFTMAX_VJP_BITS : 0u;
        for (size_t i = start; i < end; i++) {
            const int eg = (int)lQC->exponents[bfpGroupAt(&lRange, lQC, i)] - lBias;
            const int32_t d = bfpAlignPartial(gArr[i], eD - eg) - dot;
            mant[i] = bfpShrHalfAway(mant[i], sp) * bfpShrHalfAway(d, sd);
        }
        rawE[t] = (int16_t)(eD + (int)sd + (int)sp - 30);
    }
}

/* The BFP arm's walks run the input's flat count over loss and the raw out. */
static void softmaxBfpRequireCount(tensor_t *t, size_t n, const char *what) {
    size_t got = calcNumberOfElementsByTensor(t);
    if (got != n) {
        PRINT_ERROR("Softmax backward BFP: %s holds %zu elements, the input %zu", what, got, n);
        exit(1);
    }
}

void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss,
                     const layerExec_t *exec) {
    rngKey_t opKey;
    softmaxConfig_t *cfg = softmaxLayer->config->softmax;
    softmaxRequireInput(input);
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        softmaxRequireNonBfpGradWires(input, loss, propLoss);
        softmaxBackwardFloat(input, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        softmaxRequireNonBfpGradWires(input, loss, propLoss);
        rngStream_t convertStorage;
        softmaxBackwardSymInt32(
            input, loss, propLoss,
            layerExecPurposeStream(exec, RNG_TAG_BACKWARD, RNG_TAG_CONVERT, &convertStorage));
        break;
    case ARITH_BFP: {
        if (propLoss == NULL) {
            return; /* no param grads, no ops to run */
        }
        softmaxBfpRequireSingleRow(input, "Softmax backward");
        size_t n = calcNumberOfElementsByTensor(input);
        softmaxBfpRequireCount(loss, n, "loss");
        softmaxBfpRequireCount(propLoss, n, "propLoss");
        const bfpQConfig_t *anchor = cfg->propLossQ->qConfig;
        /* Known deviation, kept on purpose: a FLOAT32-stored input is
         * re-staged here with propLossMath's rounding and the BACKWARD op's
         * stage stream, while the forward staged it with forwardMath's and the
         * FORWARD op's. The recompute replays the forward's shift draws (the
         * kctx below), but s matches the forward's only when both stagings
         * agree: always for a BFP-stored input, and for a FLOAT32 one only
         * when the two modes are the same deterministic mode. */
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->propLossMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        rngStream_t shiftStorage;
        softmaxKernelCtx_t kctx = {.cfg = cfg,
                                   .shift = layerExecPurposeStream(exec, RNG_TAG_FORWARD,
                                                                   RNG_TAG_SOFTMAX_SHIFT,
                                                                   &shiftStorage)};
        softmaxRequireShiftStream(cfg, kctx.shift);
        executeOp(
            &(opSpec_t){.kernel = softmaxBackwardKernelBfp,
                        .ctx = &kctx,
                        .inputs = (tensor_t *[]){input, loss},
                        .nInputs = 2,
                        .arithmetic = cfg->propLossMath,
                        .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, SOFTMAX_OP_PROP_LOSS, &opKey),
                        .mode = OUT_WRITE,
                        .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL,
                                     loss->quantization->type == FLOAT32 ? &stage : NULL},
                        .bfpRawOut = BFP_RAW_BACKWARD_DX}, /* the integer VJP raw */
            propLoss);
        break;
    }
    }
}

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape) {
    (void)softmaxLayer;
    tensorRequireRowOrder(inputShape, "Softmax");
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
