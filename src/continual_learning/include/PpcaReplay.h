#ifndef ODT_PPCA_REPLAY_H
#define ODT_PPCA_REPLAY_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ArithmeticType.h"
#include "RNG.h"
#include "Tensor.h"

/* Per-class PPCA generator (overview in docs/CONTINUAL_LEARNING.md). State
 * storage supports FLOAT32/SYM/ASYM/BFP; BFP templates are per-tensor.
 * Arithmetic must be ARITH_FLOAT32; ppcaReplayCreate checks both. */
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

/* Reusable workspace shared across updates/classes; byte estimate excludes
 * the workspace struct. p_max = rank + maxSessionSamples + 1. */
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
/* One synthetic sample under slotKey: every draw comes from the KERNEL stream
 * of the key, so the sample is a pure function of the generator and the key. A
 * NULL key stops the program. */
void ppcaReplaySample(const ppcaReplay_t *g, const rngKey_t *slotKey, tensor_t *out);
void ppcaReplayMean(const ppcaReplay_t *g, tensor_t *out);

/* Buffer sizes of a ppcaWorkspace_t, in floats; bytes is the sum of all
 * buffers. Exits when any size overflows size_t, so every size it returns is
 * safe to allocate. Shared by ppcaWorkspaceBytes and ppcaWorkspaceCreate. */
typedef struct {
    size_t rows;       /* p_max */
    size_t bTFloats;   /* p_max * dim */
    size_t gramFloats; /* p_max * p_max (gram, eigvecs) */
    size_t bytes;
} ppcaWorkspaceLayout_t;

ppcaWorkspaceLayout_t ppcaWorkspaceLayout(size_t dim, size_t rank, size_t maxSessionSamples);

size_t ppcaReplayBytes(const ppcaReplay_t *g);
size_t ppcaWorkspaceBytes(size_t dim, size_t rank, size_t maxSessionSamples);
size_t ppcaReplayIsoExemplarCount(const ppcaReplay_t *g, size_t bytesPerExemplar);

#endif // ODT_PPCA_REPLAY_H
