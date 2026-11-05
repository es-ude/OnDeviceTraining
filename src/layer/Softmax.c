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

/* BFP epic PR6 Task 2: the float softmax, shared by the FLOAT32/SYM forward
 * kernel and by BOTH non-BFP backward arms, which recompute the softmax
 * OUTPUT from the layer INPUT they are actually handed (P6-1). Max by strict
 * `>` (first-wins on ties), expf, float sum in index order, divide;
 * softmaxValuesBfp below is the ARITH_BFP twin of exactly this contract. */
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

/* #152: row = axis 0 -- every row normalizes over all elements after axis 0
 * (crossEntropyForwardFloat's MEAN rule); a rank-1 tensor is one row. The row
 * count is the STORAGE dimensions[0], the same field that rule reads
 * (transposeTensor permutes orderOfDimensions only, never dimensions). rowLen
 * is the product of the trailing dims, not count / rows, so an empty leading
 * axis cannot divide by zero. The row walk runs over physical storage, so
 * more than one storage row requires identity order (GroupNorm.c:67's rule):
 * a permuted view would silently normalize a partition that is not its
 * logical rows. One storage row is the whole tensor for any order, as before
 * #152 -- so a logical [1, N] stored as [N, 1] + transpose fails fast. */
static void softmaxRowGeometry(const tensor_t *t, const char *where, size_t *rows, size_t *rowLen) {
    const shape_t *s = t->shape;
    size_t first = (s->numberOfDimensions >= 2) ? 1 : 0;
    *rows = (first == 1) ? s->dimensions[0] : 1;
    size_t len = 1;
    for (size_t d = first; d < s->numberOfDimensions; d++) {
        len *= s->dimensions[d];
    }
    *rowLen = len;
    if (*rows > 1) {
        for (size_t d = 0; d < s->numberOfDimensions; d++) {
            if (s->orderOfDimensions[d] != d) {
                PRINT_ERROR("%s: a wire with %zu storage rows must be identity-order (dim %zu "
                            "is order %zu)",
                            where, *rows, d, s->orderOfDimensions[d]);
                exit(1);
            }
        }
    }
}

/* Softmax's real compute is always float (numerically stable max-shifted exp);
 * SYM_INT32 forwardMath only ever meant "convert in, compute in float, convert
 * out" (never native SYM arithmetic like Linear/Conv), so the funnel's
 * prologue/epilogue perform that conversion automatically. Arithmetic is
 * hardcoded ARITH_FLOAT32 here on purpose — forwardMath no longer selects a
 * compute path, it only declares the layer's storage dtype. */
static void softmaxForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                 const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *input = ops[0];
    size_t rows;
    size_t rowLen;
    softmaxRowGeometry(input, "Softmax forward", &rows, &rowLen);
    if (rowLen == 0) {
        return; /* empty rows: no max to read */
    }

    float *x = (float *)input->data;
    float *y = (float *)rawOut->data;

    for (size_t r = 0; r < rows; r++) {
        softmaxValuesFloat(x + r * rowLen, y + r * rowLen, rowLen);
    }
}

/* BFP epic PR6 Task 4 (P6-2..P6-5): the native ARITH_BFP softmax's integer
 * steps 1-4 -- pipeline contract docs/conventions/arithmetic-bfp.md §5.9
 * (R-S2); the goldgen mirrors this function statement for statement. Writes
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

    /* (1) Max (P6-4): the exact (code, E) compare, strict > keeps the FIRST
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

    /* (2) Align onto the work-grid-or-coarser common grid (THE lossy site --
     * at most ONE rounded shift per element). When the storage grid is FINER
     * than the work grid (sigma < 0), `down` folds the extra descent into
     * that same shift -- never two chained roundings. The net shift si MAY be
     * NEGATIVE: eMax follows the
     * SIGNED argmax while block exponents follow the block ABSMAX, so a
     * negative-dominated block is legitimately coarser than the argmax block
     * (ei > eMax) -- such elements take an EXACT saturating LEFT shift. */
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
                 * (comment, not assert). m[i] == 0 is the one
                 * non-negative code that proof does NOT bound: x_i = 0 puts
                 * no cap on ei - eMax, so up can reach >= 32 -- a shift-count
                 * UB (C11 6.5.7p3) even though the value is 0 either way.
                 * The up >= 31 clause returns 0's exact shift (0) explicitly;
                 * negatives with up >= 31 already saturated above (fix
                 * round 2). */
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
    if (input->quantization->type != BFP) {
        PRINT_ERROR("softmaxForwardBfpRaw: input must be BFP (unpacked scratch form)");
        exit(1);
    }
    if (rawOut->quantization->type != BFP_RAW) {
        PRINT_ERROR("softmaxForwardBfpRaw: output must be BFP_RAW (the integer forward raw)");
        exit(1);
    }
    const size_t n = calcNumberOfElementsByTensor((tensor_t *)input);
    if (calcNumberOfElementsByTensor(rawOut) != n) {
        PRINT_ERROR("softmaxForwardBfpRaw: output holds %zu elements, the input %zu",
                    calcNumberOfElementsByTensor(rawOut), n);
        exit(1);
    }
    validateBfpQConfigShape(input->quantization->qConfig, n);
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

static void softmaxForwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                    const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    const softmaxKernelCtx_t *kctx = ctx;
    softmaxForwardBfpRaw(ops[0], kctx->cfg->bfpExpShiftRounding, kctx->shift, rawOut);
}

/* #152: the native BFP pipeline runs ONE max/alignment grid and
 * ONE partition sum over the whole wire. Per-row BFP softmax is out of scope,
 * so a multi-row call fails fast instead of normalizing across rows; a single
 * row ([N] or [1, N]) runs unchanged. */
static void softmaxBfpRequireSingleRow(const tensor_t *t, const char *where) {
    size_t rows;
    size_t rowLen;
    softmaxRowGeometry(t, where, &rows, &rowLen);
    if (rows > 1) {
        PRINT_ERROR("%s: ARITH_BFP softmax takes a single row, got %zu rows of %zu elements "
                    "(per-row BFP softmax is out of scope, #152)",
                    where, rows, rowLen);
        exit(1);
    }
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    SOFTMAX_OP_FORWARD = 0,
    SOFTMAX_OP_PROP_LOSS = 1,
};

static void softmaxRequireInputDims(const tensor_t *input, const tensor_t *t, const char *where,
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
}

void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output,
                    const layerExec_t *exec) {
    rngKey_t opKey;
    softmaxConfig_t *cfg = softmaxLayer->config->softmax;
    /* #436: every arm writes y flat at the input's row offsets into a rawOut
     * shaped like output -- a wider output is left partly unwritten, a
     * multi-row permuted one silently transposed (softmaxBackwardRequireWires'
     * twin). */
    size_t outRows;
    size_t outRowLen;
    softmaxRowGeometry(output, "Softmax forward (output)", &outRows, &outRowLen);
    softmaxRequireInputDims(input, output, "Softmax forward", "output");
    switch (cfg->forwardMath.type) {
    case ARITH_BFP: {
        softmaxBfpRequireSingleRow(input, "Softmax forward");
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->outputQ, "Softmax forward");
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
        /* P6-7: declared SYM math keeps its documented fake-quant meaning --
         * compute in float, funnel prologue/epilogue convert. The opSpec
         * arithmetic stays HARDCODED ARITH_FLOAT32 (NOT cfg->forwardMath):
         * ARITH_SYM_INT32 would make the prologue unpack into int32 scratch
         * the float kernel misreads through a float* cast. */
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
    default:
        PRINT_ERROR("Softmax forward: declared forwardMath %d not implemented",
                    (int)cfg->forwardMath.type);
        exit(1);
    }
}

/* softmaxBackward's FLOAT32/SYM arms run OUTSIDE the funnel (PR2 Task 8: the
 * FLOAT32 arm raw-casts all three wires to float*, the SYM arm convertTensors
 * through them), so each arm guards its own wires against BFP storage. PR6
 * Task 5 retired the pre-dispatch BLANKET guard: BFP wires are now legal on
 * the backward -- the ARITH_BFP arm below routes them through the funnel. */

/* The Jacobian-vector product s * (dLds - <s, dLds>) per row: the softmax
 * Jacobian is block-diagonal over rows (#152), so each row recomputes its own
 * s from its own logits (P6-1) and takes its own dot. `s` is one row of
 * scratch, reused across rows. */
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

/* #457: both non-BFP arms walk loss/propLoss in flat storage order at the
 * input's offsets (raw float* casts, or scratch sized from the input count),
 * so each wire needs the input's dims -- else an OOB read/write, a
 * stack-scratch overflow, or (count-equal reshape) rows that no longer line
 * up -- and a multi-row wire must be identity-order, else a silent
 * mis-placement (the BatchNorm1d wire contract). Runs before any conversion;
 * returns the input count. */
static size_t softmaxBackwardRequireWires(tensor_t *input, tensor_t *loss, tensor_t *propLoss,
                                          size_t *rows, size_t *rowLen) {
    size_t wireRows;
    size_t wireRowLen;
    softmaxRowGeometry(input, "Softmax backward", rows, rowLen);
    softmaxRowGeometry(loss, "Softmax backward (loss)", &wireRows, &wireRowLen);
    softmaxRowGeometry(propLoss, "Softmax backward (propLoss)", &wireRows, &wireRowLen);
    softmaxRequireInputDims(input, loss, "Softmax backward", "loss");
    softmaxRequireInputDims(input, propLoss, "Softmax backward", "propLoss");
    return calcNumberOfElementsByTensor(input);
}

/* P6-1 root fix: the training loop hands every backward the layer INPUT
 * (logits), not the softmax OUTPUT the Jacobian needs -- softmaxVjpRows
 * recomputes it. */
static void softmaxBackwardFloat(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    size_t rows;
    size_t rowLen;
    softmaxBackwardRequireWires(input, loss, propLoss, &rows, &rowLen);
    softmaxVjpRows((const float *)input->data, (const float *)loss->data, (float *)propLoss->data,
                   rows, rowLen);
}

static void softmaxBackwardSymInt32(tensor_t *input, tensor_t *loss, tensor_t *propLoss,
                                    rngStream_t *convert) {
    size_t rows;
    size_t rowLen;
    size_t inputSize = softmaxBackwardRequireWires(input, loss, propLoss, &rows, &rowLen);
    if (inputSize == 0) {
        return; /* #457: no zero-length scratch VLAs */
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
    convertTensor(propLoss, &propLossFloat, NULL);

    /* P6-1 root fix: same recompute as the float arm -- the dequantized
     * inputFloat is the layer INPUT (logits), not the softmax OUTPUT. */
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
 * No float, no division. The loss-count gate is load-bearing (the #436 OOB
 * class): a shorter per-tensor {1, 0} loss passes every grid check. The arm
 * is single-row (softmaxBfpRequireSingleRow, #152). */
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
    if (rawOut->quantization->type != BFP_RAW) {
        PRINT_ERROR("Softmax backward BFP: output must be BFP_RAW (opSpec_t.bfpRawOut = "
                    "BFP_RAW_BACKWARD_DX)");
        exit(1);
    }
    bfpRawQConfig_t *rawQC = rawOut->quantization->qConfig;
    size_t gsz = bfpRawRequireOutput(rawQC, n, "Softmax backward BFP");
    if (n == 0) {
        return;
    }
    const bfpQConfig_t *xQC = x->quantization->qConfig;
    const bfpQConfig_t *lQC = dLdsT->quantization->qConfig;
    validateBfpQConfigShape(xQC, n);
    bfpRequireElementCount(lQC, calcNumberOfElementsByTensor(dLdsT), n,
                           "Softmax backward BFP (loss)");

    int32_t *mant = (int32_t *)rawOut->data;
    softmaxExpCodesBfp(x, cfg->bfpExpShiftRounding, mant, kctx->shift); /* P6-6: R-N4 recompute */
    softmaxNormalizeRaw(mant, n);                                       /* p, Q30 */

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

void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss,
                     const layerExec_t *exec) {
    rngKey_t opKey;
    softmaxConfig_t *cfg = softmaxLayer->config->softmax;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        bfpRequireNoBfpWire(input, "Softmax backward (input)");
        bfpRequireNoBfpWire(loss, "Softmax backward (loss)");
        bfpRequireNoBfpWire(propLoss, "Softmax backward (propLoss)");
        softmaxBackwardFloat(input, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        /* BFP wires under DECLARED SYM math stay denied. Not because
         * convertTensor would fail (the BFP->FLOAT32 cell exists and this arm
         * would silently "work") but as the outside-funnel twin of the funnel's
         * PR2-Decision-11 deny: legal BFP-storage arithmetics are FLOAT32
         * (fake-quant) and BFP (native) only. */
        bfpRequireNoBfpWire(input, "Softmax backward (input)");
        bfpRequireNoBfpWire(loss, "Softmax backward (loss)");
        bfpRequireNoBfpWire(propLoss, "Softmax backward (propLoss)");
        rngStream_t convertStorage;
        softmaxBackwardSymInt32(
            input, loss, propLoss,
            layerExecPurposeStream(exec, RNG_TAG_BACKWARD, RNG_TAG_CONVERT, &convertStorage));
        break;
    case ARITH_BFP: {
        if (propLoss == NULL) {
            return; /* P6-6: no param grads, no ops to run */
        }
        softmaxBfpRequireSingleRow(input, "Softmax backward");
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->propLossQ, "Softmax backward");
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
    default:
        PRINT_ERROR("Softmax backward: declared propLossMath %d not implemented",
                    (int)cfg->propLossMath.type);
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
