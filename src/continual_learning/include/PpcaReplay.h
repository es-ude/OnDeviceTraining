#ifndef ODT_PPCA_REPLAY_H
#define ODT_PPCA_REPLAY_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ArithmeticType.h"
#include "RNG.h"
#include "Tensor.h"

/* Per-class PPCA generator (#326; overview in docs/CONTINUAL_LEARNING.md).
 * State tensors accept FLOAT32/SYM/ASYM
 * storage; SYM_INT32 (#261), INT32 (silent value-cast) and BOOL (no
 * conversion cell) are rejected at create. v1 arithmetic: ARITH_FLOAT32
 * only (integer arms are the tracking-issue follow-up; no-int64 rule and
 * the d > 511 Gram bound are the quantified blockers). */
typedef struct {
    size_t dim, rank;       /* d, k */
    tensor_t *mean;         /* [d] */
    tensor_t *basis;        /* [k,d]; rows orthonormal; beyond-rank rows zeroed */
    tensor_t *eigvals;      /* [k]; descending; beyond-rank entries 0 */
    float sigma2, totalVar; /* totalVar = running total scatter */
    uint32_t count;         /* n */
    arithmetic_t mergeMath, streamMath, sampleMath;
    float sigma2Floor, shrinkageGamma;
} ppcaReplay_t;

typedef struct {
    size_t dim, rank, maxSessionSamples;
    arithmetic_t mergeMath, streamMath, sampleMath;
    quantization_t *meanQ, *basisQ, *eigvalsQ; /* borrowed; cloned via getQLike */
    float sigma2Floor;                         /* e.g. 1e-6f; also the initial sigma2 */
    float shrinkageGamma; /* lambda_i <- (1-g)*lambda_i + g*mean(top-kept lambda); 0 = off */
} ppcaReplayConfig_t;

/* Merge/streaming workspace: reserveMemory-backed flat float buffers,
 * created once (ppcaWorkspaceCreate), reused across updates and classes.
 * p_max = rank + maxSessionSamples + 1. ppcaWorkspaceBytes returns EXACTLY
 * the sum of the buffers below. */
typedef struct {
    size_t dim, rank, maxSessionSamples;
    float *bT;        /* [(p_max) * dim]  augmented matrix, stored transposed */
    float *gram;      /* [p_max * p_max] */
    float *eigvecs;   /* [p_max * p_max] */
    float *theta;     /* [p_max] */
    float *lambdaOut; /* [rank] */
    float *rowScales; /* [p_max] */
    float *meanBatch; /* [dim]  (path B reuses as xFloat) */
    float *muOld;     /* [dim] */
    float *u;         /* [dim]  path-B deflation buffer */
    float sigma2Out;
    float totalVarOut;
} ppcaWorkspace_t;

typedef struct {
    size_t numClasses;
    ppcaReplay_t **generators;
    ppcaWorkspace_t *workspace;
} ppcaReplaySet_t;

/* Math API (no allocation in steady state; all funnel-routed). */
void ppcaReplayUpdate(ppcaReplay_t *g, const tensor_t *samples, ppcaWorkspace_t *ws);
void ppcaReplayUpdateStreaming(ppcaReplay_t *g, const tensor_t *x, ppcaWorkspace_t *ws);
void ppcaReplaySample(const ppcaReplay_t *g, rng32_t *rng, tensor_t *out);
void ppcaReplayMean(const ppcaReplay_t *g, tensor_t *out);

size_t ppcaReplayBytes(const ppcaReplay_t *g);
size_t ppcaWorkspaceBytes(size_t dim, size_t rank, size_t maxSessionSamples);
size_t ppcaReplayIsoExemplarCount(const ppcaReplay_t *g, size_t bytesPerExemplar);

/* Shared v1 fail-fast: PPCA math knobs accept only ARITH_FLOAT32 (#310
 * pattern). Called at create AND at every step entry. */
void ppcaValidateFloatArith(arithmetic_t math, const char *context);

#endif // ODT_PPCA_REPLAY_H
