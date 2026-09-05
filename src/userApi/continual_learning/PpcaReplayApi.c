#define SOURCE_FILE "PPCA_REPLAY_API"

#include <stdlib.h>

#include "Common.h"
#include "PpcaReplay.h"
#include "PpcaReplayApi.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

static void validateStateStorage(const quantization_t *q, const char *field) {
    if (q == NULL) {
        PRINT_ERROR("ppcaReplayCreate: %s storage config is NULL", field);
        exit(1);
    }
    if (q->type != FLOAT32 && q->type != SYM && q->type != ASYM && q->type != BFP) {
        /* SYM_INT32 is compute-format-not-storage (#261); INT32 would be a
         * silent value-cast through the conversion matrix; BOOL has no cell.
         * BFP epic PR3 Task 6: BFP joins the whitelist -- the gate lift
         * gradInit/optimizer-state already went through. Every still-
         * unsupported dtype is rejected HERE, never left to the matrix. */
        PRINT_ERROR("ppcaReplayCreate: %s storage must be FLOAT32/SYM/ASYM/BFP", field);
        exit(1);
    }
    /* BFP epic PR3 Task 7: BFP state storage is per-tensor-only, mirroring
     * the optimizer-state gates (SgdApi/AdamWApi) -- a grouped template
     * would flow through getQLike into a grouped BFP state tensor with no
     * spec mandate (#300 axis). SYM/ASYM templates share this gap but are
     * deliberately left as-is here (pre-existing behavior, out of scope). */
    if (q->type == BFP) {
        const bfpQConfig_t *bfpQC = q->qConfig;
        if (bfpQC->numGroups > 1) {
            PRINT_ERROR("ppcaReplayCreate: %s grouped BFP state templates are unsupported -- "
                        "per-tensor only (#300 axis)",
                        field);
            exit(1);
        }
    }
}

static shape_t *buildOwnedShape(const size_t *srcDims, size_t numberOfDims) {
    size_t *dims = reserveMemory(numberOfDims * sizeof(size_t));
    size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
    for (size_t i = 0; i < numberOfDims; i++) {
        dims[i] = srcDims[i];
        order[i] = i;
    }
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = numberOfDims;
    return shape;
}

ppcaReplay_t *ppcaReplayCreate(const ppcaReplayConfig_t *cfg) {
    if (cfg->dim == 0 || cfg->rank == 0 || cfg->rank >= cfg->dim) {
        PRINT_ERROR("ppcaReplayCreate: need 0 < rank < dim (dim=%zu rank=%zu)", cfg->dim,
                    cfg->rank);
        exit(1);
    }
    /* NaN-robust (adamWInit betas idiom): sigma2Floor seeds sigma2 AND floors
     * every merge -- a NaN or non-positive floor silently collapses generated
     * samples toward the class mean (the sigma-noise term dies), poisoning
     * replay results with no crash. */
    if (!(cfg->sigma2Floor > 0.0f)) {
        PRINT_ERROR("ppcaReplayCreate: sigma2Floor must be > 0 (got %g)", (double)cfg->sigma2Floor);
        exit(1);
    }
    validateStateStorage(cfg->meanQ, "meanQ");
    validateStateStorage(cfg->basisQ, "basisQ");
    validateStateStorage(cfg->eigvalsQ, "eigvalsQ");
    ppcaValidateFloatArith(cfg->mergeMath, "ppcaReplayCreate mergeMath");
    ppcaValidateFloatArith(cfg->streamMath, "ppcaReplayCreate streamMath");
    ppcaValidateFloatArith(cfg->sampleMath, "ppcaReplayCreate sampleMath");

    ppcaReplay_t *g = reserveMemory(sizeof(ppcaReplay_t));
    g->dim = cfg->dim;
    g->rank = cfg->rank;
    g->mean = initTensor(buildOwnedShape((size_t[]){cfg->dim}, 1), getQLike(cfg->meanQ), NULL);
    g->basis = initTensor(buildOwnedShape((size_t[]){cfg->rank, cfg->dim}, 2),
                          getQLike(cfg->basisQ), NULL);
    g->eigvals =
        initTensor(buildOwnedShape((size_t[]){cfg->rank}, 1), getQLike(cfg->eigvalsQ), NULL);
    g->sigma2 = cfg->sigma2Floor;
    g->totalVar = 0.0f;
    g->count = 0;
    g->mergeMath = cfg->mergeMath;
    g->streamMath = cfg->streamMath;
    g->sampleMath = cfg->sampleMath;
    g->sigma2Floor = cfg->sigma2Floor;
    g->shrinkageGamma = cfg->shrinkageGamma;
    return g;
}

void freePpcaReplay(ppcaReplay_t *g) {
    if (g == NULL) {
        return;
    }
    freeTensor(g->eigvals);
    freeTensor(g->basis);
    freeTensor(g->mean);
    freeReservedMemory(g);
}

ppcaWorkspace_t *ppcaWorkspaceCreate(size_t dim, size_t rank, size_t maxSessionSamples) {
    if (dim == 0 || rank == 0 || maxSessionSamples == 0) {
        PRINT_ERROR("ppcaWorkspaceCreate: dim/rank/maxSessionSamples must be > 0");
        exit(1);
    }
    size_t p = rank + maxSessionSamples + 1;
    ppcaWorkspace_t *ws = reserveMemory(sizeof(ppcaWorkspace_t));
    ws->dim = dim;
    ws->rank = rank;
    ws->maxSessionSamples = maxSessionSamples;
    ws->bT = reserveMemory(p * dim * sizeof(float));
    ws->gram = reserveMemory(p * p * sizeof(float));
    ws->eigvecs = reserveMemory(p * p * sizeof(float));
    ws->theta = reserveMemory(p * sizeof(float));
    ws->lambdaOut = reserveMemory(rank * sizeof(float));
    ws->rowScales = reserveMemory(p * sizeof(float));
    ws->meanBatch = reserveMemory(dim * sizeof(float));
    ws->muOld = reserveMemory(dim * sizeof(float));
    ws->u = reserveMemory(dim * sizeof(float));
    ws->sigma2Out = 0.0f;
    ws->totalVarOut = 0.0f;
    return ws;
}

void freePpcaWorkspace(ppcaWorkspace_t *ws) {
    if (ws == NULL) {
        return;
    }
    freeReservedMemory(ws->u);
    freeReservedMemory(ws->muOld);
    freeReservedMemory(ws->meanBatch);
    freeReservedMemory(ws->rowScales);
    freeReservedMemory(ws->lambdaOut);
    freeReservedMemory(ws->theta);
    freeReservedMemory(ws->eigvecs);
    freeReservedMemory(ws->gram);
    freeReservedMemory(ws->bT);
    freeReservedMemory(ws);
}

ppcaReplaySet_t *ppcaReplaySetCreate(size_t numClasses, const ppcaReplayConfig_t *cfg) {
    if (numClasses == 0) {
        PRINT_ERROR("ppcaReplaySetCreate: numClasses must be > 0");
        exit(1);
    }
    ppcaReplaySet_t *set = reserveMemory(sizeof(ppcaReplaySet_t));
    set->numClasses = numClasses;
    set->generators = reserveMemory(numClasses * sizeof(ppcaReplay_t *));
    for (size_t c = 0; c < numClasses; c++) {
        set->generators[c] = ppcaReplayCreate(cfg);
    }
    set->workspace = ppcaWorkspaceCreate(cfg->dim, cfg->rank, cfg->maxSessionSamples);
    return set;
}

void freePpcaReplaySet(ppcaReplaySet_t *set) {
    if (set == NULL) {
        return;
    }
    freePpcaWorkspace(set->workspace);
    for (size_t c = 0; c < set->numClasses; c++) {
        freePpcaReplay(set->generators[c]);
    }
    freeReservedMemory(set->generators);
    freeReservedMemory(set);
}
