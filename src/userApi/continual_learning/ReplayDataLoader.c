#define SOURCE_FILE "REPLAY_DATA_LOADER"

#include <stdlib.h>

#include "Common.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "ExemplarBuffer.h"
#include "PpcaReplay.h"
#include "PpcaReplayApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "TensorApi.h"

typedef struct {
    dataLoader_t pub; /* MUST stay first: container-of cast in replayGetBatch */
    dataLoader_t *base;
    replayLoaderConfig_t cfg;
    bool poolsReady;
    tensor_t **itemPool;  /* [numClasses * samplesPerClass], lazily built */
    tensor_t **labelPool; /* [numClasses] shared one-hot labels */
} replayLoader_t;

static size_t loaderNumClasses(const replayLoader_t *rl) {
    return rl->cfg.mode == REPLAY_MODE_EXEMPLAR ? rl->cfg.exemplars->numClasses
                                                : rl->cfg.set->numClasses;
}

static void initPoolsFromFirstSample(replayLoader_t *rl, const sample_t *first) {
    size_t numClasses = loaderNumClasses(rl);
    size_t r = rl->cfg.samplesPerClass;
    if (first->label->quantization->type != FLOAT32 ||
        calcNumberOfElementsByTensor(first->label) != numClasses) {
        PRINT_ERROR("replayDataLoader: labels must be FLOAT32 one-hot [numClasses=%zu]",
                    numClasses);
        exit(1);
    }
    if (rl->cfg.mode == REPLAY_MODE_EXEMPLAR) {
        /* exemplars are lent out as-is — the element check runs against the
         * first stored one (if the buffer is still empty, nothing will be
         * appended anyway); no item pool: replay is zero-copy */
        exemplarBuffer_t *buf = rl->cfg.exemplars;
        for (size_t c = 0; c < numClasses; c++) {
            if (buf->counts[c] > 0) {
                size_t stored = calcNumberOfElementsByTensor(buf->items[c * buf->capacity]);
                if (calcNumberOfElementsByTensor(first->item) != stored) {
                    PRINT_ERROR("replayDataLoader: item element count %zu != exemplar %zu",
                                calcNumberOfElementsByTensor(first->item), stored);
                    exit(1);
                }
                break;
            }
        }
    } else {
        size_t dim = rl->cfg.set->generators[0]->dim;
        if (calcNumberOfElementsByTensor(first->item) != dim) {
            PRINT_ERROR("replayDataLoader: item element count %zu != generator dim %zu",
                        calcNumberOfElementsByTensor(first->item), dim);
            exit(1);
        }
        rl->itemPool = reserveMemory(numClasses * r * sizeof(tensor_t *));
        for (size_t c = 0; c < numClasses; c++) {
            for (size_t i = 0; i < r; i++) {
                rl->itemPool[c * r + i] = initTensor(getShapeLike(first->item->shape),
                                                     getQLike(first->item->quantization), NULL);
            }
        }
    }
    rl->labelPool = reserveMemory(numClasses * sizeof(tensor_t *));
    for (size_t c = 0; c < numClasses; c++) {
        rl->labelPool[c] = initTensor(getShapeLike(first->label->shape),
                                      getQLike(first->label->quantization), NULL);
        float *lab = (float *)rl->labelPool[c]->data;
        for (size_t j = 0; j < numClasses; j++) {
            lab[j] = (j == c) ? 1.0f : 0.0f;
        }
    }
    rl->poolsReady = true;
}

static bool classEligible(const replayLoader_t *rl, size_t c) {
    if (rl->cfg.mode == REPLAY_MODE_EXEMPLAR) {
        return rl->cfg.exemplars->counts[c] > 0;
    }
    return rl->cfg.set->generators[c]->count >= rl->cfg.minCount;
}

static batch_t *replayGetBatch(dataLoader_t *dl, size_t index) {
    replayLoader_t *rl = (replayLoader_t *)dl;
    batch_t *baseBatch = rl->base->getBatch(rl->base, index);
    if (!rl->poolsReady) {
        initPoolsFromFirstSample(rl, baseBatch->samples[0]);
    }

    size_t numClasses = loaderNumClasses(rl);
    size_t r = rl->cfg.samplesPerClass;
    size_t eligible = 0;
    for (size_t c = 0; c < numClasses; c++) {
        if (classEligible(rl, c)) {
            eligible++;
        }
    }
    if (eligible == 0) {
        return baseBatch; /* untouched pass-through */
    }

    size_t newSize = baseBatch->size + eligible * r;
    sample_t **samples = reserveMemory(newSize * sizeof(sample_t *));
    for (size_t i = 0; i < baseBatch->size; i++) {
        samples[i] = baseBatch->samples[i]; /* REAL FIRST (labelRef contract) */
    }
    size_t w = baseBatch->size;
    for (size_t c = 0; c < numClasses; c++) {
        if (!classEligible(rl, c)) {
            continue;
        }
        for (size_t i = 0; i < r; i++) {
            tensor_t *item;
            if (rl->cfg.mode == REPLAY_MODE_EXEMPLAR) {
                /* uniform pick WITH replacement (mirrors PPCA's independent
                 * per-slot draws); zero-copy: the buffer lends the tensor */
                exemplarBuffer_t *buf = rl->cfg.exemplars;
                size_t pick = (size_t)(rngNextFloatCtx(rl->cfg.stream) * (float)buf->counts[c]);
                if (pick >= buf->counts[c]) {
                    pick = buf->counts[c] - 1;
                }
                item = buf->items[c * buf->capacity + pick];
            } else {
                ppcaReplay_t *g = rl->cfg.set->generators[c];
                item = rl->itemPool[c * r + i]; /* pool-owned, reused every batch */
                if (rl->cfg.mode == REPLAY_MODE_CLASS_MEAN) {
                    ppcaReplayMean(g, item);
                } else {
                    ppcaReplaySample(g, rl->cfg.stream, item);
                }
            }
            sample_t *s = reserveMemory(sizeof(sample_t));
            s->item = item;
            s->label = rl->labelPool[c]; /* shared one-hot, never freed by loop */
            samples[w++] = s;
        }
    }

    batch_t *batch = reserveMemory(sizeof(batch_t));
    batch->samples = samples;
    batch->size = newSize;
    /* Hand back the base containers (array + struct, NOT the sample_t's we
     * adopted — freeBatch never touches those). */
    freeBatch(baseBatch);
    return batch;
}

dataLoader_t *replayDataLoaderWrap(dataLoader_t *base, const replayLoaderConfig_t *cfg) {
    if (base == NULL || cfg == NULL || cfg->samplesPerClass == 0 ||
        (cfg->mode == REPLAY_MODE_EXEMPLAR ? cfg->exemplars == NULL : cfg->set == NULL) ||
        (cfg->stream == NULL && cfg->mode != REPLAY_MODE_CLASS_MEAN)) {
        PRINT_ERROR("replayDataLoaderWrap: base required, samplesPerClass >= 1, source "
                    "required (set, or exemplars for REPLAY_MODE_EXEMPLAR), stream "
                    "required unless REPLAY_MODE_CLASS_MEAN");
        exit(1);
    }
    replayLoader_t *rl = reserveMemory(sizeof(replayLoader_t));
    rl->pub = *base; /* borrow every base field (indices, sizes, fn ptrs) */
    rl->pub.getBatch = replayGetBatch;
    rl->base = base;
    rl->cfg = *cfg;
    rl->poolsReady = false;
    rl->itemPool = NULL;
    rl->labelPool = NULL;
    return &rl->pub;
}

void freeReplayDataLoader(dataLoader_t *wrapped) {
    if (wrapped == NULL) {
        return;
    }
    replayLoader_t *rl = (replayLoader_t *)wrapped;
    if (rl->poolsReady) {
        size_t numClasses = loaderNumClasses(rl);
        size_t r = rl->cfg.samplesPerClass;
        for (size_t c = 0; c < numClasses; c++) {
            if (rl->itemPool != NULL) { /* EXEMPLAR mode lends buffer tensors */
                for (size_t i = 0; i < r; i++) {
                    freeTensor(rl->itemPool[c * r + i]);
                }
            }
            freeTensor(rl->labelPool[c]);
        }
        if (rl->itemPool != NULL) {
            freeReservedMemory(rl->itemPool);
        }
        freeReservedMemory(rl->labelPool);
    }
    freeReservedMemory(rl); /* base loader + its indices stay caller-owned */
}
