#define SOURCE_FILE "REDUCE"

#include "Reduce.h"

#include "Add.h"
#include "Arithmetic.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "Div.h"
#include "Mul.h"
#include "ScaleCanon.h"
#include "Square.h"
#include "Sub.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* int16-range mantissas sum in an int32 accumulator without overflow for
 * N < 2^(32-qMaxBits). A value-sum bound, looser than the int12 product-operand
 * contract (ODT_SYM_OPERAND_QMAXBITS); do not alias the two. */
#define REDUCE_SYM_VALUESUM_QMAXBITS 16

/* K = product of the leading rank-k logical dims, N = product of the trailing k. */
static void blockGeom(tensor_t *in, size_t k, size_t *K, size_t *N) {
    size_t rank = in->shape->numberOfDimensions;
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

/* Physical offset of logical element (block b, inner j). A contiguous b*N + j
 * would read wrong elements on a permuted view. */
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
        v[b] = divFloat32s(acc, (float)N);
    }
}

void sumSquaresOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *ssqOut) {
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    float *x = (float *)in->data;
    float *s = (float *)ssqOut->data;
    for (size_t b = 0; b < K; b++) {
        float acc = 0.0f;
        for (size_t j = 0; j < N; j++) {
            float v = x[reducePhysOffset(in, k, b, j)];
            acc = addFloat32s(acc, squareFloat32(v));
        }
        s[b] = acc;
    }
}

float rsqrtFloat32(float x, float eps) {
    return divFloat32s(1.0f, sqrtf(addFloat32s(x, eps)));
}

float sqrtFloat32(float x) {
    return sqrtf(x);
}

/* 1 <= qMaxBits <= 16 bounds the int32 value-sum; the lower bound also keeps the
 * mean's N-bound shift below 32 on 32-bit size_t targets. No entry rejects a
 * zero-width SYM quantization, so the check stays here. */
static void reduceValidateSymWidth(tensor_t *t) {
    uint8_t qMaxBits = ((symInt32QConfig_t *)t->quantization->qConfig)->qMaxBits;
    if (qMaxBits == 0 || qMaxBits > REDUCE_SYM_VALUESUM_QMAXBITS) {
        PRINT_ERROR("Reduce SYM_INT32: qMaxBits (%u) outside [1, %u]", (unsigned)qMaxBits,
                    (unsigned)REDUCE_SYM_VALUESUM_QMAXBITS);
        exit(1);
    }
}

void meanOverTrailingAxesSymInt32(tensor_t *in, size_t k, tensor_t *meanOut) {
    reduceValidateSymWidth(in);
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    uint8_t qMaxBits = ((symInt32QConfig_t *)in->quantization->qConfig)->qMaxBits;
    /* Worst-case sum reaches INT32_MIN at N = 2^(32-qMaxBits). */
    size_t bound = (size_t)1 << (32u - qMaxBits);
    if (N >= bound) {
        PRINT_ERROR("meanOverTrailingAxesSymInt32: N (%zu) exceeds the value-sum bound for "
                    "qMaxBits (%u) -- must be < 2^(32-qMaxBits) (%zu)",
                    N, (unsigned)qMaxBits, bound);
        exit(1);
    }
    int32_t *q = (int32_t *)in->data;
    float *m = (float *)meanOut->data;
    float s = loadCanonScale(&((symInt32QConfig_t *)in->quantization->qConfig)->scale);
    for (size_t b = 0; b < K; b++) {
        int32_t sumQ = 0;
        for (size_t j = 0; j < N; j++) {
            sumQ = addInt32s(sumQ, q[reducePhysOffset(in, k, b, j)]);
        }
        m[b] = mulFloat32s(s, divFloat32s((float)sumQ, (float)N));
    }
}

void varianceBiasedOverTrailingAxesSymInt32(tensor_t *in, size_t k, tensor_t *meanIn,
                                            tensor_t *varOut) {
    reduceValidateSymWidth(in);
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    int32_t *q = (int32_t *)in->data;
    float *m = (float *)meanIn->data;
    float *v = (float *)varOut->data;
    float s = loadCanonScale(&((symInt32QConfig_t *)in->quantization->qConfig)->scale);
    for (size_t b = 0; b < K; b++) {
        float acc = 0.0f;
        for (size_t j = 0; j < N; j++) {
            float d = subFloat32s(mulFloat32s((float)q[reducePhysOffset(in, k, b, j)], s), m[b]);
            acc = addFloat32s(acc, squareFloat32(d));
        }
        v[b] = divFloat32s(acc, (float)N);
    }
}

void meanOverTrailingAxesBfp(tensor_t *in, size_t k, tensor_t *meanOut) {
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    const bfpQConfig_t *qC = in->quantization->qConfig;
    bfpValidateSumHeadroom(qC, N, "meanOverTrailingAxesBfp");
    const int32_t expBias = bfpExponentBias(qC);
    int32_t const *q = (int32_t const *)in->data;
    float *m = (float *)meanOut->data;
    for (size_t b = 0; b < K; b++) {
        float acc = 0.0f;
        int32_t partial = 0;
        size_t currentGroup = 0;
        for (size_t j = 0; j < N; j++) {
            size_t off = reducePhysOffset(in, k, b, j);
            size_t g = bfpGroupOf(qC, off);
            if (j == 0) {
                currentGroup = g;
            } else if (g != currentGroup) {
                acc = addFloat32s(
                    acc, ldexpf((float)partial, (int)qC->exponents[currentGroup] - expBias));
                partial = 0;
                currentGroup = g;
            }
            partial = addInt32s(partial, q[off]);
        }
        acc = addFloat32s(acc, ldexpf((float)partial, (int)qC->exponents[currentGroup] - expBias));
        m[b] = divFloat32s(acc, (float)N);
    }
}

void varianceBiasedOverTrailingAxesBfp(tensor_t *in, size_t k, tensor_t *meanIn, tensor_t *varOut) {
    size_t K;
    size_t N;
    blockGeom(in, k, &K, &N);
    const bfpQConfig_t *qC = in->quantization->qConfig;
    const int32_t expBias = bfpExponentBias(qC);
    int32_t const *q = (int32_t const *)in->data;
    float *m = (float *)meanIn->data;
    float *v = (float *)varOut->data;
    for (size_t b = 0; b < K; b++) {
        float acc = 0.0f;
        for (size_t j = 0; j < N; j++) {
            size_t off = reducePhysOffset(in, k, b, j);
            float x = ldexpf((float)q[off], (int)qC->exponents[bfpGroupOf(qC, off)] - expBias);
            float d = subFloat32s(x, m[b]);
            acc = addFloat32s(acc, squareFloat32(d));
        }
        v[b] = divFloat32s(acc, (float)N);
    }
}
