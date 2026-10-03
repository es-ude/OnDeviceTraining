#define SOURCE_FILE "PPCA_REPLAY"

#include <stdlib.h>

#include "Axpby.h"
#include "Common.h"
#include "Distributions.h"
#include "ExecuteOp.h"
#include "JacobiEig.h"
#include "Matmul.h"
#include "MinMax.h"
#include "Mul.h"
#include "PpcaReplay.h"
#include "Reduce.h"
#include "RowBroadcast.h"
#include "Tensor.h"
#include "TensorConversion.h"

/* Shared v1 arithmetic guard (Sgd.c/#310 wording pattern; re-checked at
 * every step because the kernels raw-cast operand data to float*). */
void ppcaValidateFloatArith(arithmetic_t math, const char *context) {
    if (math.type != ARITH_FLOAT32) {
        PRINT_ERROR("%s: only ARITH_FLOAT32 is implemented (integer-arithmetic PPCA "
                    "numerics are literature-first future work, see #326 tracking issue)",
                    context);
        exit(1);
    }
}

size_t ppcaWorkspaceBytes(size_t dim, size_t rank, size_t maxSessionSamples) {
    size_t p = rank + maxSessionSamples + 1;
    size_t floats = p * dim /* bT */
                    + p * p /* gram */
                    + p * p /* eigvecs */
                    + p     /* theta */
                    + rank  /* lambdaOut */
                    + p     /* rowScales */
                    + dim   /* meanBatch */
                    + dim   /* muOld */
                    + dim;  /* u */
    return floats * sizeof(float);
}

size_t ppcaReplayBytes(const ppcaReplay_t *g) {
    size_t bytes = sizeof(ppcaReplay_t);
    bytes += calcNumberOfBytesForData(g->mean->quantization, g->dim);
    bytes += calcNumberOfBytesForData(g->basis->quantization, g->rank * g->dim);
    bytes += calcNumberOfBytesForData(g->eigvals->quantization, g->rank);
    return bytes;
}

size_t ppcaReplayIsoExemplarCount(const ppcaReplay_t *g, size_t bytesPerExemplar) {
    if (bytesPerExemplar == 0) {
        PRINT_ERROR("ppcaReplayIsoExemplarCount: bytesPerExemplar must be > 0");
        exit(1);
    }
    return ppcaReplayBytes(g) / bytesPerExemplar;
}

typedef struct {
    float sigma2;
    size_t kEff;
    rng32_t *rng;
} ppcaSampleCtx_t;

/* operands {mean, basis, eigvals} (ARITH_FLOAT32 views via the funnel
 * prologue) -> rawOut = mu + sum_i coeff_i * basisRow_i + sigma*eps.
 * ALWAYS draws rank + dim variates: z beyond kEff is drawn and
 * discarded so the stream position never depends on kEff. */
static void ppcaSampleKernelFloat(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                                  const void *ctxv) {
    (void)n;
    (void)aux;
    const ppcaSampleCtx_t *ctx = ctxv;
    const float *mu = (const float *)op[0]->data;
    const float *basis = (const float *)op[1]->data;
    const float *lam = (const float *)op[2]->data;
    float *out = (float *)rawOut->data;
    size_t d = calcNumberOfElementsByTensor(op[0]);
    size_t k = calcNumberOfElementsByTensor(op[2]);
    float sigma = sqrtFloat32(maxFloat32s(ctx->sigma2, 0.0f));

    for (size_t j = 0; j < d; j++) {
        out[j] = mu[j];
    }
    for (size_t i = 0; i < k; i++) {
        float z = randomNormalCtx(ctx->rng, 0.0f, 1.0f); /* always drawn */
        if (i < ctx->kEff) {
            float coeff = sqrtFloat32(maxFloat32s(lam[i] - ctx->sigma2, 0.0f)) * z;
            for (size_t j = 0; j < d; j++) {
                out[j] += coeff * basis[i * d + j];
            }
        }
    }
    for (size_t j = 0; j < d; j++) {
        out[j] += sigma * randomNormalCtx(ctx->rng, 0.0f, 1.0f);
    }
}

void ppcaReplaySample(const ppcaReplay_t *g, rng32_t *rng, tensor_t *out) {
    ppcaValidateFloatArith(g->sampleMath, "ppcaReplaySample sampleMath");
    if (g->count == 0) {
        PRINT_ERROR("ppcaReplaySample: generator has absorbed no data (count == 0)");
        exit(1);
    }
    if (calcNumberOfElementsByTensor(out) != g->dim) {
        PRINT_ERROR("ppcaReplaySample: out must have exactly dim=%zu elements", g->dim);
        exit(1);
    }
    if (out->quantization->type != FLOAT32 && out->quantization->type != SYM &&
        out->quantization->type != ASYM) {
        PRINT_ERROR("ppcaReplaySample: out storage must be FLOAT32/SYM/ASYM");
        exit(1);
    }
    size_t kEff = g->rank;
    if ((size_t)(g->count - 1) < kEff) {
        kEff = (size_t)(g->count - 1);
    }
    ppcaSampleCtx_t ctx = {.sigma2 = g->sigma2, .kEff = kEff, .rng = rng};
    executeOp(
        &(opSpec_t){
            .kernel = ppcaSampleKernelFloat,
            .ctx = &ctx,
            /* funnel's sources-never-mutated contract: executeOp treats
             * `inputs` as read-only despite the non-const tensor_t** param,
             * so stripping g's const here is safe. */
            .inputs =
                (tensor_t *[]){(tensor_t *)g->mean, (tensor_t *)g->basis, (tensor_t *)g->eigvals},
            .nInputs = 3,
            .arithmetic = g->sampleMath,
            .mode = OUT_WRITE,
            .auxOut = NULL,
            .writesInPlaceSafe = false,
        },
        out);
}

/* Mean-replay baseline (#326 fair comparison): out <- mu. A pure storage
 * conversion through the conversionMatrix — no RNG, no arithmetic, so
 * sampleMath is deliberately NOT consulted (centroid replay stays available
 * where an arithmetic arm would be rejected). Same out contract as
 * ppcaReplaySample: any element-count-dim FLOAT32/SYM/ASYM tensor. */
void ppcaReplayMean(const ppcaReplay_t *g, tensor_t *out) {
    if (g->count == 0) {
        PRINT_ERROR("ppcaReplayMean: generator has absorbed no data (count == 0)");
        exit(1);
    }
    if (calcNumberOfElementsByTensor(out) != g->dim) {
        PRINT_ERROR("ppcaReplayMean: out must have exactly dim=%zu elements", g->dim);
        exit(1);
    }
    if (out->quantization->type != FLOAT32 && out->quantization->type != SYM &&
        out->quantization->type != ASYM) {
        PRINT_ERROR("ppcaReplayMean: out storage must be FLOAT32/SYM/ASYM");
        exit(1);
    }
    executeConvert(g->mean, out);
}

/* Stack-bound FLOAT32 view over a workspace buffer (no allocation; the
 * view lives exactly as long as the enclosing call). */
typedef struct {
    tensor_t tensor;
    shape_t shape;
    quantization_t quant;
    size_t dims[2];
    size_t order[2];
} ppcaFloatView_t;

static tensor_t *bindFloatView(ppcaFloatView_t *v, float *data, size_t rank, size_t d0, size_t d1) {
    v->dims[0] = d0;
    v->dims[1] = d1;
    v->order[0] = 0;
    v->order[1] = 1;
    v->shape.numberOfDimensions = rank;
    v->shape.dimensions = v->dims;
    v->shape.orderOfDimensions = v->order;
    initFloat32Quantization(&v->quant);
    v->tensor.data = (uint8_t *)data;
    v->tensor.shape = &v->shape;
    v->tensor.quantization = &v->quant;
    v->tensor.sparsity = NULL;
    return &v->tensor;
}

typedef struct {
    float a, b;
} ppcaMeanMergeCtx_t;

/* operands {muOld, meanBatch} (FLOAT32 scratch) -> rawOut = a*op0 + b*op1. */
static void ppcaMeanMergeKernelFloat(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                                     const void *ctxv) {
    (void)n;
    (void)aux;
    const ppcaMeanMergeCtx_t *ctx = ctxv;
    axpbyFloat32Tensors(ctx->a, op[0], ctx->b, op[1], rawOut);
}

typedef struct {
    ppcaWorkspace_t *ws;
    uint32_t nOld;
    size_t m;
    float totalVarOld;
    float sigma2Floor;
    float shrinkageGamma;
} ppcaMergeCtx_t;

#define PPCA_JACOBI_MAX_SWEEPS 30
#define PPCA_JACOBI_TOL 1e-6f
#define PPCA_THETA_EPS 1e-6f

/* operands {basis, eigvals}; muOld/meanBatch and all scratch via ctx->ws;
 * the ingested session rows already sit (UNcentered) in ws->bT rows k..k+m.
 * rawOut = new basis [k,d]. writesInPlaceSafe=true is sound: op[0] is fully
 * consumed into bT (step 1) before the first rawOut write (step 8). */
static void ppcaMergeKernelFloat(tensor_t **op, size_t nOps, tensor_t *rawOut, tensor_t *aux,
                                 const void *ctxv) {
    (void)nOps;
    (void)aux;
    const ppcaMergeCtx_t *ctx = ctxv;
    ppcaWorkspace_t *ws = ctx->ws;
    size_t d = ws->dim;
    size_t k = ws->rank;
    size_t m = ctx->m;
    size_t p = k + m + 1;
    float n = (float)ctx->nOld;
    float fm = (float)m;
    const float *lamOld = (const float *)op[1]->data;

    ppcaFloatView_t vA, vB, vC, vD, vE;

    /* 1. top rows: bT[0..k) = diag(sqrt(n*lam)) * basis. */
    for (size_t i = 0; i < k; i++) {
        ws->rowScales[i] = sqrtFloat32(n * maxFloat32s(lamOld[i], 0.0f));
    }
    scaleRowsFloat32(op[0], bindFloatView(&vA, ws->rowScales, 1, k, 0),
                     bindFloatView(&vB, ws->bT, 2, k, d));

    /* 2. center session rows in place: bT[k..k+m) -= meanBatch. */
    tensor_t *sampleRegion = bindFloatView(&vA, ws->bT + k * d, 2, m, d);
    tensor_t *meanBatchView = bindFloatView(&vB, ws->meanBatch, 1, d, 0);
    subRowBroadcastFloat32(sampleRegion, meanBatchView, sampleRegion);

    /* 3. correction row k+m: sqrt(n*m/(n+m)) * (muOld - meanBatch). */
    tensor_t *corrRow = bindFloatView(&vC, ws->bT + (k + m) * d, 1, d, 0);
    axpbyFloat32Tensors(1.0f, bindFloatView(&vD, ws->muOld, 1, d, 0), -1.0f, meanBatchView,
                        corrRow);
    float corrFactor = (ctx->nOld == 0) ? 0.0f : sqrtFloat32((n * fm) / (n + fm));
    mulFloat32ElementWithFloat32TensorInplace(corrRow, corrFactor);

    /* 4. totalVar' = totalVarOld + ssq(centered rows) + ssq(corr row)
     *    (the corr row's factor^2 IS the n*m/(n+m) between-means term). */
    float ssqSamples = 0.0f, ssqCorr = 0.0f;
    sumSquaresOverTrailingAxesFloat32(sampleRegion, 2, bindFloatView(&vD, &ssqSamples, 1, 1, 0));
    sumSquaresOverTrailingAxesFloat32(corrRow, 1, bindFloatView(&vD, &ssqCorr, 1, 1, 0));
    float totalVarNew = ctx->totalVarOld + ssqSamples + ssqCorr;

    /* 5. Gram = bT * bT^T via a zero-copy transpose view (Linear.c idiom). */
    tensor_t *bTView = bindFloatView(&vA, ws->bT, 2, p, d);
    tensor_t *bTTView = bindFloatView(&vB, ws->bT, 2, p, d);
    transposeTensor(bTTView, 0, 1);
    tensor_t *gramView = bindFloatView(&vC, ws->gram, 2, p, p);
    matmulFloat32Tensors(bTView, bTTView, gramView);

    /* 6. eigendecompose G (destroyed). */
    tensor_t *thetaView = bindFloatView(&vD, ws->theta, 1, p, 0);
    tensor_t *eigvecsView = bindFloatView(&vE, ws->eigvecs, 2, p, p);
    jacobiEigSymFloat32(gramView, thetaView, eigvecsView, PPCA_JACOBI_MAX_SWEEPS, PPCA_JACOBI_TOL);

    /* 7. numerical-rank guard (theta is descending). */
    float thetaMax = maxFloat32s(ws->theta[0], 0.0f);
    size_t rankNum = 0;
    while (rankNum < p && ws->theta[rankNum] > PPCA_THETA_EPS * thetaMax &&
           ws->theta[rankNum] > 0.0f) {
        rankNum++;
    }
    size_t rKeep = (rankNum < k) ? rankNum : k;

    /* 8. rotate back only rKeep rows; zero the rest. */
    float *outData = (float *)rawOut->data;
    for (size_t i = 0; i < k * d; i++) {
        outData[i] = 0.0f;
    }
    if (rKeep > 0) {
        for (size_t i = 0; i < rKeep; i++) {
            ws->rowScales[i] = rsqrtFloat32(ws->theta[i], 0.0f);
        }
        tensor_t *topEig = bindFloatView(&vA, ws->eigvecs, 2, rKeep, p);
        scaleRowsFloat32(topEig, bindFloatView(&vB, ws->rowScales, 1, rKeep, 0), topEig);
        tensor_t *bTFull = bindFloatView(&vB, ws->bT, 2, p, d);
        tensor_t *outTop = bindFloatView(&vC, outData, 2, rKeep, d);
        matmulFloat32Tensors(topEig, bTFull, outTop);
        /* renormalize kept rows (requant-after-normalize) */
        sumSquaresOverTrailingAxesFloat32(outTop, 1,
                                          bindFloatView(&vD, ws->rowScales, 1, rKeep, 0));
        for (size_t i = 0; i < rKeep; i++) {
            ws->rowScales[i] = rsqrtFloat32(ws->rowScales[i], 1e-12f);
        }
        scaleRowsFloat32(outTop, bindFloatView(&vD, ws->rowScales, 1, rKeep, 0), outTop);
    }

    /* 9. spectrum + optional shrinkage. */
    float denom = n + fm;
    for (size_t i = 0; i < k; i++) {
        ws->lambdaOut[i] = (i < rKeep) ? ws->theta[i] / denom : 0.0f;
    }
    if (ctx->shrinkageGamma > 0.0f && rKeep > 0) {
        float lamBar = 0.0f;
        for (size_t i = 0; i < rKeep; i++) {
            lamBar += ws->lambdaOut[i];
        }
        lamBar /= (float)rKeep;
        for (size_t i = 0; i < rKeep; i++) {
            ws->lambdaOut[i] =
                (1.0f - ctx->shrinkageGamma) * ws->lambdaOut[i] + ctx->shrinkageGamma * lamBar;
        }
    }

    /* 10. sigma2' with floor. */
    float kept = 0.0f;
    for (size_t i = 0; i < rKeep; i++) {
        kept += ws->lambdaOut[i];
    }
    float sigma2New = (totalVarNew / denom - kept) / (float)(d - rKeep);
    if (!(sigma2New > ctx->sigma2Floor)) { /* NaN-robust floor */
        sigma2New = ctx->sigma2Floor;
    }
    ws->sigma2Out = sigma2New;
    ws->totalVarOut = totalVarNew;
}

void ppcaReplayUpdate(ppcaReplay_t *g, const tensor_t *samples, ppcaWorkspace_t *ws) {
    ppcaValidateFloatArith(g->mergeMath, "ppcaReplayUpdate mergeMath");
    if (ws->dim != g->dim || ws->rank != g->rank) {
        PRINT_ERROR("ppcaReplayUpdate: workspace built for dim=%zu rank=%zu", ws->dim, ws->rank);
        exit(1);
    }
    if (samples->quantization->type == BOOL) {
        PRINT_ERROR("ppcaReplayUpdate: BOOL samples are not ingestible (no conversion cell)");
        exit(1);
    }
    if (samples->shape->numberOfDimensions != 2 || samples->shape->dimensions[1] != g->dim) {
        PRINT_ERROR("ppcaReplayUpdate: samples must be [m,%zu]", g->dim);
        exit(1);
    }
    size_t m = samples->shape->dimensions[0];
    if (m == 0 || m > ws->maxSessionSamples) {
        PRINT_ERROR("ppcaReplayUpdate: need 1 <= m <= maxSessionSamples=%zu (got %zu)",
                    ws->maxSessionSamples, m);
        exit(1);
    }

    size_t d = g->dim;
    size_t k = g->rank;
    ppcaFloatView_t vIngest, vTrans, vMean, vMuOld, vLambda;

    /* Step 0 — ingest (any dtype except BOOL) into the bT session region:
     * executeConvert = the funnel's kernel-less form; heap target, zero
     * stack. Sources are never mutated. */
    tensor_t *sampleRegion = bindFloatView(&vIngest, ws->bT + k * d, 2, m, d);
    executeConvert((tensor_t *)samples, sampleRegion);

    /* Step 1 — batch mean over the ingested region via a [d,m] transpose
     * view (Reduce is trailing-axes-only; reads are permutation-aware). */
    tensor_t *transView = bindFloatView(&vTrans, ws->bT + k * d, 2, m, d);
    transposeTensor(transView, 0, 1);
    tensor_t *meanBatchView = bindFloatView(&vMean, ws->meanBatch, 1, d, 0);
    meanOverTrailingAxesFloat32(transView, 1, meanBatchView);

    /* Step 2 — stage muOld (dequants packed mean ONCE; preserves it against
     * step 5's overwrite). */
    tensor_t *muOldView = bindFloatView(&vMuOld, ws->muOld, 1, d, 0);
    executeConvert(g->mean, muOldView);

    /* Step 3 — the merge op (target: basis state). */
    ppcaMergeCtx_t ctx = {
        .ws = ws,
        .nOld = g->count,
        .m = m,
        .totalVarOld = g->totalVar,
        .sigma2Floor = g->sigma2Floor,
        .shrinkageGamma = g->shrinkageGamma,
    };
    executeOp(
        &(opSpec_t){
            .kernel = ppcaMergeKernelFloat,
            .ctx = &ctx,
            .inputs = (tensor_t *[]){g->basis, g->eigvals},
            .nInputs = 2,
            .arithmetic = g->mergeMath,
            .mode = OUT_WRITE,
            .auxOut = NULL,
            .writesInPlaceSafe = true, /* basis fully consumed before rawOut writes */
        },
        g->basis);

    /* Step 4 — eigvals via identity kernel (NOT auxOut: auxOut is never
     * funnel-converted and would pin eigvals storage to FLOAT32). */
    tensor_t *lambdaView = bindFloatView(&vLambda, ws->lambdaOut, 1, k, 0);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .ctx = NULL,
            .inputs = (tensor_t *[]){lambdaView},
            .nInputs = 1,
            .arithmetic = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
            .auxOut = NULL,
            .writesInPlaceSafe = false,
        },
        g->eigvals);

    /* Step 5 — mean merge LAST (muOld already staged). */
    float nf = (float)g->count;
    float mf = (float)m;
    ppcaMeanMergeCtx_t mctx = {.a = nf / (nf + mf), .b = mf / (nf + mf)};
    executeOp(
        &(opSpec_t){
            .kernel = ppcaMeanMergeKernelFloat,
            .ctx = &mctx,
            .inputs = (tensor_t *[]){muOldView, meanBatchView},
            .nInputs = 2,
            .arithmetic = g->mergeMath,
            .mode = OUT_WRITE,
            .auxOut = NULL,
            .writesInPlaceSafe = true,
        },
        g->mean);

    g->sigma2 = ws->sigma2Out;
    g->totalVar = ws->totalVarOut;
    g->count += (uint32_t)m;
}

typedef struct {
    ppcaWorkspace_t *ws;
    uint32_t nOld;
} ppcaCcipcaCtx_t;

/* operands {basis, eigvals}; the ingested sample sits in ws->meanBatch
 * (reused as xFloat), muOld staged in ws->muOld. rawOut = updated basis.
 * CCIPCA (Weng et al. 2003), amnesic-free: component i lives once
 * count > i+1; bootstrap of component (nOld-1) from the deflated residual.
 * lambda' -> ws->lambdaOut, Welford totalVar/sigma2 -> ws fields. */
static void ppcaCcipcaStepKernelFloat(tensor_t **op, size_t nOps, tensor_t *rawOut, tensor_t *aux,
                                      const void *ctxv) {
    (void)nOps;
    (void)aux;
    const ppcaCcipcaCtx_t *ctx = ctxv;
    ppcaWorkspace_t *ws = ctx->ws;
    size_t d = ws->dim;
    size_t k = ws->rank;
    const float *basis = (const float *)op[0]->data;
    const float *lamOld = (const float *)op[1]->data;
    float *out = (float *)rawOut->data;
    const float *x = ws->meanBatch; /* xFloat */
    const float *muOld = ws->muOld;
    float nNew = (float)(ctx->nOld + 1);

    /* Welford: totalVar += (x-muOld)·(x-muNew); u starts as x - muNew. */
    float welford = 0.0f;
    for (size_t j = 0; j < d; j++) {
        float xc = x[j] - muOld[j];
        float muNewJ = muOld[j] + xc / nNew;
        float uj = x[j] - muNewJ;
        ws->u[j] = uj;
        welford += xc * uj;
    }
    ws->totalVarOut += welford; /* orchestrator pre-seeds totalVarOut = g->totalVar */

    size_t liveBefore = (ctx->nOld >= 1) ? ((ctx->nOld - 1 < k) ? ctx->nOld - 1 : k) : 0;

    for (size_t i = 0; i < k; i++) {
        if (i < liveBefore) {
            /* v = lam*u_hat; v' = ((n-1)/n) v + (1/n)(u . u_hat) u */
            float dot = 0.0f;
            for (size_t j = 0; j < d; j++) {
                dot += ws->u[j] * basis[i * d + j];
            }
            float w1 = (nNew - 1.0f) / nNew;
            float w2 = dot / nNew;
            float norm = 0.0f;
            for (size_t j = 0; j < d; j++) {
                float vj = w1 * lamOld[i] * basis[i * d + j] + w2 * ws->u[j];
                out[i * d + j] = vj;
                norm += vj * vj;
            }
            norm = sqrtFloat32(norm);
            ws->lambdaOut[i] = norm;
            if (norm > 0.0f) {
                float inv = 1.0f / norm;
                for (size_t j = 0; j < d; j++) {
                    out[i * d + j] *= inv;
                }
            }
            /* deflate with the UPDATED direction */
            float dotU = 0.0f;
            for (size_t j = 0; j < d; j++) {
                dotU += ws->u[j] * out[i * d + j];
            }
            for (size_t j = 0; j < d; j++) {
                ws->u[j] -= dotU * out[i * d + j];
            }
        } else if (i == liveBefore && ctx->nOld >= 1) {
            /* bootstrap from the residual */
            float norm = 0.0f;
            for (size_t j = 0; j < d; j++) {
                norm += ws->u[j] * ws->u[j];
            }
            norm = sqrtFloat32(norm);
            ws->lambdaOut[i] = norm;
            if (norm > 0.0f) {
                float inv = 1.0f / norm;
                for (size_t j = 0; j < d; j++) {
                    out[i * d + j] = ws->u[j] * inv;
                    ws->u[j] = 0.0f;
                }
            } else {
                for (size_t j = 0; j < d; j++) {
                    out[i * d + j] = 0.0f;
                }
            }
        } else {
            for (size_t j = 0; j < d; j++) {
                out[i * d + j] = 0.0f;
            }
            ws->lambdaOut[i] = 0.0f;
        }
    }

    /* sigma2 = mean residual variance over the d - live discarded dims:
     * (totalVar/n - sum of the live eigenvalues) / (d - live). */
    size_t live = (ctx->nOld < k) ? ctx->nOld : k;
    float kept = 0.0f;
    for (size_t i = 0; i < live; i++) {
        kept += ws->lambdaOut[i];
    }
    float sigma2New = (ws->totalVarOut / nNew - kept) / (float)(d - live);
    ws->sigma2Out = sigma2New;
}

void ppcaReplayUpdateStreaming(ppcaReplay_t *g, const tensor_t *x, ppcaWorkspace_t *ws) {
    ppcaValidateFloatArith(g->streamMath, "ppcaReplayUpdateStreaming streamMath");
    if (ws->dim != g->dim || ws->rank != g->rank) {
        PRINT_ERROR("ppcaReplayUpdateStreaming: workspace built for dim=%zu rank=%zu", ws->dim,
                    ws->rank);
        exit(1);
    }
    if (x->quantization->type == BOOL) {
        PRINT_ERROR("ppcaReplayUpdateStreaming: BOOL input is not ingestible");
        exit(1);
    }
    if (calcNumberOfElementsByTensor((tensor_t *)x) != g->dim) {
        PRINT_ERROR("ppcaReplayUpdateStreaming: x must have dim=%zu elements", g->dim);
        exit(1);
    }

    size_t d = g->dim;
    size_t k = g->rank;
    ppcaFloatView_t vX, vMuOld, vLambda;

    /* Step 0 — ingest x (any dtype) + stage muOld. */
    tensor_t *xView = bindFloatView(&vX, ws->meanBatch, 1, d, 0);
    executeConvert((tensor_t *)x, xView);
    tensor_t *muOldView = bindFloatView(&vMuOld, ws->muOld, 1, d, 0);
    executeConvert(g->mean, muOldView);

    ws->totalVarOut = g->totalVar;
    ppcaCcipcaCtx_t ctx = {.ws = ws, .nOld = g->count};
    executeOp(
        &(opSpec_t){
            .kernel = ppcaCcipcaStepKernelFloat,
            .ctx = &ctx,
            .inputs = (tensor_t *[]){g->basis, g->eigvals},
            .nInputs = 2,
            .arithmetic = g->streamMath,
            .mode = OUT_WRITE,
            .auxOut = NULL,
            .writesInPlaceSafe = false, /* v1 conservative */
        },
        g->basis);

    tensor_t *lambdaView = bindFloatView(&vLambda, ws->lambdaOut, 1, k, 0);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .ctx = NULL,
            .inputs = (tensor_t *[]){lambdaView},
            .nInputs = 1,
            .arithmetic = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
            .auxOut = NULL,
            .writesInPlaceSafe = false,
        },
        g->eigvals);

    float nf = (float)g->count;
    ppcaMeanMergeCtx_t mctx = {.a = nf / (nf + 1.0f), .b = 1.0f / (nf + 1.0f)};
    executeOp(
        &(opSpec_t){
            .kernel = ppcaMeanMergeKernelFloat,
            .ctx = &mctx,
            .inputs = (tensor_t *[]){muOldView, xView},
            .nInputs = 2,
            .arithmetic = g->streamMath,
            .mode = OUT_WRITE,
            .auxOut = NULL,
            .writesInPlaceSafe = true,
        },
        g->mean);

    g->totalVar = ws->totalVarOut;
    if (ws->sigma2Out > g->sigma2Floor) {
        g->sigma2 = ws->sigma2Out;
    } else {
        g->sigma2 = g->sigma2Floor;
    }
    g->count += 1;
}
