#define SOURCE_FILE "REPLAY_DATA_LOADER"

#include <stdlib.h>

#include "CheckedSize.h"
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

static void *reserveOrExit(size_t bytes, const char *what) {
    void *p = reserveMemory(bytes);
    if (p == NULL) {
        PRINT_ERROR("replayDataLoader: cannot reserve %zu bytes for the %s", bytes, what);
        exit(1);
    }
    return p;
}

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
        /* the wrap-time batch-array bound also bounds this product */
        rl->itemPool = reserveOrExit(numClasses * r * sizeof(tensor_t *), "replay item pool");
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

static batch_t *replayGetBatch(dataLoader_t *dl, size_t index, const rngKey_t *stepKey) {
    replayLoader_t *rl = (replayLoader_t *)dl;
    /* The wrapper copied the base's fields when it wrapped it, but both share
     * one order array: an order set through the wrapper is the base's order
     * too. Hand that on before the base's own fetch checks it, and never clear
     * it: the base may have been shuffled itself after wrapping. */
    if (rl->pub.shuffled) {
        rl->base->shuffled = true;
    }
    batch_t *baseBatch = rl->base->getBatch(rl->base, index, stepKey);
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
    /* One key per replayed slot, under the update's key: a slot's draw depends
     * on (update, class, slot) alone, never on the fetches before it. The class
     * means draw nothing and need no key. */
    rngKey_t replayKey = {{0u, 0u}};
    if (rl->cfg.mode != REPLAY_MODE_CLASS_MEAN) {
        if (stepKey == NULL) {
            PRINT_ERROR("replayGetBatch: replay draw without a step key");
            exit(1);
        }
        replayKey = rngFold(*stepKey, RNG_TAG_CL_REPLAY, 0u);
    }

    /* eligible * r <= numClasses * r, bounded at wrap time */
    size_t newSize;
    size_t bytes;
    if (!checkedAddSize(baseBatch->size, eligible * r, &newSize) ||
        !checkedMulSize(newSize, sizeof(sample_t *), &bytes)) {
        PRINT_ERROR("replayGetBatch: base batch of %zu samples plus %zu replayed overflows",
                    baseBatch->size, eligible * r);
        exit(1);
    }
    sample_t **samples = reserveOrExit(bytes, "replayed batch");
    for (size_t i = 0; i < baseBatch->size; i++) {
        samples[i] = baseBatch->samples[i]; /* REAL FIRST (labelRef contract) */
    }
    size_t w = baseBatch->size;
    for (size_t c = 0; c < numClasses; c++) {
        if (!classEligible(rl, c)) {
            continue;
        }
        const rngKey_t classKey = rngFold(replayKey, RNG_TAG_CLASS, (uint32_t)c);
        for (size_t i = 0; i < r; i++) {
            const rngKey_t slotKey = rngFold(classKey, RNG_TAG_SLOT, (uint32_t)i);
            tensor_t *item;
            if (rl->cfg.mode == REPLAY_MODE_EXEMPLAR) {
                /* uniform pick WITH replacement (mirrors PPCA's independent
                 * per-slot draws) by the shuffle's bounded rule; an eligible
                 * class holds at least one exemplar. Zero-copy: the buffer
                 * lends the tensor. */
                exemplarBuffer_t *buf = rl->cfg.exemplars;
                rngStream_t pickStream = rngOpen(slotKey);
                size_t pick = rngNextBounded(&pickStream, buf->counts[c]);
                item = buf->items[c * buf->capacity + pick];
            } else {
                ppcaReplay_t *g = rl->cfg.set->generators[c];
                item = rl->itemPool[c * r + i]; /* pool-owned, reused every batch */
                if (rl->cfg.mode == REPLAY_MODE_CLASS_MEAN) {
                    ppcaReplayMean(g, item);
                } else {
                    ppcaReplaySample(g, &slotKey, item);
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
        (cfg->mode == REPLAY_MODE_EXEMPLAR ? cfg->exemplars == NULL : cfg->set == NULL)) {
        PRINT_ERROR("replayDataLoaderWrap: base required, samplesPerClass >= 1, source "
                    "required (set, or exemplars for REPLAY_MODE_EXEMPLAR)");
        exit(1);
    }
    /* Slot i draws under (SLOT, i), a 32-bit fold index. */
    if (cfg->samplesPerClass > UINT32_MAX) {
        PRINT_ERROR("replayDataLoaderWrap: samplesPerClass exceeds UINT32_MAX");
        exit(1);
    }
    /* A batch holds batchSize real + numClasses * r replayed pointers at most. */
    size_t numClasses =
        cfg->mode == REPLAY_MODE_EXEMPLAR ? cfg->exemplars->numClasses : cfg->set->numClasses;
    size_t slots;
    size_t total;
    size_t bytes;
    if (!checkedMulSize(numClasses, cfg->samplesPerClass, &slots) ||
        !checkedAddSize(slots, base->batchSize, &total) ||
        !checkedMulSize(total, sizeof(sample_t *), &bytes)) {
        PRINT_ERROR("replayDataLoaderWrap: batchSize %u + numClasses %zu x samplesPerClass %zu "
                    "overflows the batch array",
                    (unsigned)base->batchSize, numClasses, cfg->samplesPerClass);
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
