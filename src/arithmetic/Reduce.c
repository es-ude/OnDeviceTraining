#define SOURCE_FILE "REDUCE"

#include "Reduce.h"

#include "Add.h"
#include "Arithmetic.h" // getDimensionsByIndex, calcElementIndexByIndices
#include "Common.h"
#include "Div.h"
#include "MinMax.h"
#include "Mul.h"
#include "Quantization.h"
#include "Rounding.h"
#include "Square.h"
#include "Sub.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Value-sum bound for the SYM_INT32 reductions: int16-range mantissas sum in an
 * int32 accumulator without overflow for N < 2^(32-qMaxBits) (>= 65536 terms at
 * 16). Deliberately a NEW, looser bound than the int12 operand contract
 * (ODT_SYM_OPERAND_QMAXBITS): the reduction here is a VALUE-sum, not a product
 * accumulator, so no int12 product-headroom argument applies. Do NOT alias
 * ODT_SYM_GRAD_QMAXBITS (same numeric value, different contract). */
#define REDUCE_SYM_VALUESUM_QMAXBITS 16

/* Width bound for the SYM_INT32 REQUANT TARGET (rsqrtSymInt32's `out`), NOT an
 * operand-guard alias. Numerically identical to REDUCE_SYM_VALUESUM_QMAXBITS
 * (16) but for a wholly different reason -- do NOT fold the two guards:
 *   - The operand guard (reduceValidateSymOperand) bounds an int32
 *     ACCUMULATOR reading pre-existing mantissas (overflow argument).
 *   - This bound polices a WRITE target instead: rsqrtSymInt32 derives out's
 *     scale dynamically every call (scale = absMax/qMax) and requantizes into
 *     it. Widths above 16 bits are, by design, reserved for scale=1
 *     raw-integer semantics (see the wide-SYM design intent in
 *     docs/conventions -- qMaxBits > 16 only makes sense at scale=1). A
 *     dynamic per-call scale on a >16-bit target would silently violate that
 *     contract, and a scale=1 rsqrt output is meaningless anyway (rsqrt
 *     results are fractional, not raw integers). So <=16 here is a
 *     DELIBERATE contract for requant targets, not a leftover overflow bound
 *     -- it would stay 16 even if the operand guard's bound ever changed. */
#define REDUCE_SYM_REQUANT_QMAXBITS 16

/* K = product of the leading (rank-k) logical dims (block count); N = product of
 * the trailing k logical dims (per-block element count). Logical sizes honor
 * orderOfDimensions via getDimensionsByIndex. k == rank -> K = 1 (whole-tensor
 * reduction). */
static void blockGeom(tensor_t *in, size_t k, size_t *K, size_t *N) {
    size_t rank = in->shape->numberOfDimensions;
    if (k > rank) {
        /* k > rank underflows `rank - k` (size_t) in reducePhysOffset's block-
         * decompose loop. That loop would then call getDimensionsByIndex with a
         * near-SIZE_MAX index, which itself fail-fasts (exit(1)) BEFORE the
         * would-be OOB idx[] write — so this is not a silent-corruption bug.
         * We still guard at blockGeom, the single chokepoint both public
         * reductions call, to fail fast with a clear "k exceeds rank" diagnostic
         * instead of a cryptic "Tensor doesn't have <huge> dimensions!". */
        PRINT_ERROR("Reduce: k (%zu) exceeds input rank (%zu)", k, rank);
        exit(1);
    }
    size_t nn = 1;
    size_t kk = 1;
    for (size_t d = 0; d < rank; d++) {
        size_t e = getDimensionsByIndex(in, d);
        if (d >= rank - k) {
            nn *= e;
        } else {
            kk *= e;
        }
    }
    *K = kk;
    *N = nn;
}

/* Physical flat offset of the logical element (block b, inner j): decompose j
 * over the trailing k logical dims and b over the leading rank-k dims, then map
 * through calcElementIndexByIndices (which applies orderOfDimensions).
 * Generalizes layerNormPhysOffset (LayerNorm.c). Permutation-awareness is
 * v1-mandatory: a contiguous b*N + j offset would silently read wrong elements
 * on a transposed/viewed input. */
static size_t reducePhysOffset(tensor_t *t, size_t k, size_t b, size_t j) {
    size_t rank = t->shape->numberOfDimensions;
    size_t idx[rank];

    size_t rem = j;
    for (size_t d = rank; d-- > rank - k;) {
        size_t dimSize = getDimensionsByIndex(t, d);
        idx[d] = rem % dimSize;
        rem /= dimSize;
    }
    rem = b;
    for (size_t d = rank - k; d-- > 0;) {
        size_t dimSize = getDimensionsByIndex(t, d);
        idx[d] = rem % dimSize;
        rem /= dimSize;
    }
    return calcElementIndexByIndices(rank, t->shape->dimensions, idx, t->shape->orderOfDimensions);
}

void meanOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *meanOut) {
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    float *x = (float *)in->data;
    float *m = (float *)meanOut->data;
    for (size_t b = 0; b < K; b++) {
        float acc = 0.0f;
        for (size_t j = 0; j < N; j++) {
            acc = addFloat32s(acc, x[reducePhysOffset(in, k, b, j)]);
        }
        m[b] = divFloat32s(acc, (float)N);
    }
}

void varianceBiasedOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *meanIn,
                                           tensor_t *varOut) {
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    float *x = (float *)in->data;
    float *m = (float *)meanIn->data;
    float *v = (float *)varOut->data;
    for (size_t b = 0; b < K; b++) {
        float acc = 0.0f;
        for (size_t j = 0; j < N; j++) {
            float d = subFloat32s(x[reducePhysOffset(in, k, b, j)], m[b]);
            acc = addFloat32s(acc, squareFloat32(d));
        }
        v[b] = divFloat32s(acc, (float)N); /* BIASED -- divide by N, not N-1 */
    }
}

float rsqrtFloat32(float x, float eps) {
    return divFloat32s(1.0f, sqrtf(addFloat32s(x, eps))); /* eps INSIDE sqrt guards x == 0 */
}

/* A FLOAT32 buffer read as int32 mantissas is silent garbage, so fail fast on a
 * non-SYM operand. The qMaxBits <= 16 bound is a deliberately NEW, looser bound
 * than the int12 operand contract (ODT_SYM_OPERAND_QMAXBITS = 12): the mantissa
 * reduction here is a VALUE-sum, not a product accumulator, so int16-range
 * mantissas sum in an int32 accumulator without overflow for N < 2^(32-qMaxBits)
 * (>= 65536 terms at qMaxBits = 16) -- no int12 product-headroom argument
 * applies. (mirrors layerNormValidateSymTensor, whose value-sum path is likewise
 * sound at qMaxBits <= 16.) Also rejects qMaxBits == 0: a zero-width SYM operand
 * is degenerate/invalid, and excluding it guarantees qMaxBits in [1,16] for every
 * downstream user -- notably the N-guard's `32 - qMaxBits` shift in
 * meanOverTrailingAxesSymInt32, which would otherwise shift by 32 and be UB on
 * 32-bit size_t targets. */
static void reduceValidateSymOperand(tensor_t *t, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("Reduce SYM_INT32: %s must be SYM_INT32", what);
        exit(1);
    }
    symInt32QConfig_t *qc = t->quantization->qConfig;
    if (qc->qMaxBits == 0) {
        PRINT_ERROR("Reduce SYM_INT32: %s qMaxBits is 0 -- a zero-width SYM operand is "
                    "degenerate/invalid",
                    what);
        exit(1);
    }
    if (qc->qMaxBits > REDUCE_SYM_VALUESUM_QMAXBITS) {
        PRINT_ERROR("Reduce SYM_INT32: %s qMaxBits (%u) exceeds the value-sum bound (%u)", what,
                    (unsigned)qc->qMaxBits, (unsigned)REDUCE_SYM_VALUESUM_QMAXBITS);
        exit(1);
    }
}

/* Guard for a SYM_INT32 REQUANT TARGET (a tensor rsqrtSymInt32 writes into,
 * not reads from) -- distinct contract from reduceValidateSymOperand above.
 * A FLOAT32 buffer written with int32 mantissas is silent garbage, so the
 * dtype check is shared. The width bound is NOT shared: see
 * REDUCE_SYM_REQUANT_QMAXBITS for why <=16 here is a requant-target contract
 * (dynamic-scale write validity), not a value-sum accumulator bound. Use this
 * guard ONLY on tensors being requantized into, never on read-only operands
 * (those use reduceValidateSymOperand). */
static void reduceValidateSymRequantTarget(tensor_t *t, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("Reduce SYM_INT32: %s must be SYM_INT32", what);
        exit(1);
    }
    symInt32QConfig_t *qc = t->quantization->qConfig;
    if (qc->qMaxBits > REDUCE_SYM_REQUANT_QMAXBITS) {
        PRINT_ERROR("Reduce SYM_INT32: %s qMaxBits (%u) exceeds the requant-target bound (%u)",
                    what, (unsigned)qc->qMaxBits, (unsigned)REDUCE_SYM_REQUANT_QMAXBITS);
        exit(1);
    }
}

void meanOverTrailingAxesSymInt32(tensor_t *in, size_t k, tensor_t *meanOut) {
    reduceValidateSymOperand(in, "input");
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    uint8_t qMaxBits = ((symInt32QConfig_t *)in->quantization->qConfig)->qMaxBits;
    /* Shift safety: reduceValidateSymOperand now guarantees qMaxBits in [1,16], so
     * the shift amount (32 - qMaxBits) is in [16,31] -- the relevant fact is the
     * upper bound (<= 31, no UB on a 32-bit size_t), not the lower one (>= 16 was
     * never what made this safe). Conservative bound -- the worst-case value-sum
     * reaches exactly INT32_MIN at N = 2^(32-qMaxBits); the guard rejects from
     * that boundary on. */
    size_t bound = (size_t)1 << (32u - qMaxBits);
    if (N >= bound) {
        PRINT_ERROR("meanOverTrailingAxesSymInt32: N (%zu) exceeds the value-sum bound for "
                    "qMaxBits (%u) -- must be < 2^(32-qMaxBits) (%zu)",
                    N, (unsigned)qMaxBits, bound);
        exit(1);
    }
    int32_t *q = (int32_t *)in->data;
    float *m = (float *)meanOut->data;
    float s = ((symInt32QConfig_t *)in->quantization->qConfig)->scale;
    for (size_t b = 0; b < K; b++) {
        int32_t sumQ = 0; /* value-sum, sound for N < 2^(32-qMaxBits) */
        for (size_t j = 0; j < N; j++) {
            sumQ = addInt32s(sumQ, q[reducePhysOffset(in, k, b, j)]);
        }
        m[b] = mulFloat32s(s, divFloat32s((float)sumQ, (float)N));
    }
}

void varianceBiasedOverTrailingAxesSymInt32(tensor_t *in, size_t k, tensor_t *meanIn,
                                            tensor_t *varOut) {
    reduceValidateSymOperand(in, "input");
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    int32_t *q = (int32_t *)in->data;
    float *m = (float *)meanIn->data;
    float *v = (float *)varOut->data;
    float s = ((symInt32QConfig_t *)in->quantization->qConfig)->scale;
    for (size_t b = 0; b < K; b++) {
        float acc = 0.0f;
        for (size_t j = 0; j < N; j++) {
            /* Dequant-center-square in float (mantissa*scale, not an int product). */
            float d = subFloat32s(mulFloat32s((float)q[reducePhysOffset(in, k, b, j)], s), m[b]);
            acc = addFloat32s(acc, squareFloat32(d));
        }
        v[b] = divFloat32s(acc, (float)N); /* BIASED -- divide by N, not N-1 */
    }
}

void rsqrtSymInt32(tensor_t *in, float eps, tensor_t *out) {
    reduceValidateSymOperand(in, "input");
    reduceValidateSymRequantTarget(out, "output");
    size_t n = calcNumberOfElementsByTensor(in);
    int32_t *qIn = (int32_t *)in->data;
    int32_t *qOut = (int32_t *)out->data;
    float sIn = ((symInt32QConfig_t *)in->quantization->qConfig)->scale;
    symInt32QConfig_t *outQC = out->quantization->qConfig;
    const float qMax = powf(2, (float)outQC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qMaxBits - 1);

    /* Dynamic absmax of the dequant->rsqrt result, recomputed (no scratch buffer:
     * alloc-locality forbids reserveMemory in arithmetic/). rsqrt of any finite
     * input is strictly positive, so absMax == 0 only when n == 0. */
    float absMax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float r = rsqrtFloat32(mulFloat32s((float)qIn[i], sIn), eps);
        absMax = maxFloat32s(absMax, absFloat32(r));
    }

    if (absMax == 0.0f) { /* n == 0: nothing to write, neutral scale (cf. #160) */
        outQC->scale = 1.0f;
        return;
    }

    /* Requant into out via the convertFloatTensorToSymInt32Tensor idiom
     * (scale = absMax/qMax, round-clamp), refreshing out's scale on every call. */
    float scale = divFloat32s(absMax, qMax);
    outQC->scale = scale;
    for (size_t i = 0; i < n; i++) {
        float r = rsqrtFloat32(mulFloat32s((float)qIn[i], sIn), eps);
        qOut[i] = clampInt32(roundByMode(divFloat32s(r, scale), outQC->roundingMode), (int32_t)qMin,
                             (int32_t)qMax);
    }
}
