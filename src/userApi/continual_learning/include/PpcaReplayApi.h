#ifndef ODT_PPCA_REPLAY_API_H
#define ODT_PPCA_REPLAY_API_H

#include "DataLoader.h"
#include "ExemplarBuffer.h"
#include "PpcaReplay.h"

ppcaReplay_t *ppcaReplayCreate(const ppcaReplayConfig_t *cfg);
void freePpcaReplay(ppcaReplay_t *g);

ppcaWorkspace_t *ppcaWorkspaceCreate(size_t dim, size_t rank, size_t maxSessionSamples);
void freePpcaWorkspace(ppcaWorkspace_t *ws);

ppcaReplaySet_t *ppcaReplaySetCreate(size_t numClasses, const ppcaReplayConfig_t *cfg);
void freePpcaReplaySet(ppcaReplaySet_t *set);

typedef enum {
    REPLAY_MODE_PPCA_SAMPLE = 0, /* draw from the generative model (default) */
    REPLAY_MODE_CLASS_MEAN,      /* replay the running class centroid */
    REPLAY_MODE_EXEMPLAR,        /* replay stored raw samples; slots are
                                    uniform draws (with replacement) from the
                                    class's buffer, handed out ZERO-COPY */
} replayMode_t;

typedef struct {
    ppcaReplaySet_t *set;        /* PPCA_SAMPLE/CLASS_MEAN; unused in EXEMPLAR */
    exemplarBuffer_t *exemplars; /* EXEMPLAR only; caller-owned, must outlive
                                    the wrapper (items are lent to batches) */
    size_t samplesPerClass;      /* r; CLASS_MEAN appends r IDENTICAL centroid
                                    copies so batch composition and the MEAN-loss
                                    weighting stay comparable across modes */
    uint32_t minCount;           /* class eligible once generator count >= minCount;
                                    0 only when every generator is populated.
                                    Ignored by EXEMPLAR, which gates on a
                                    non-empty class buffer */
    replayMode_t mode;           /* zero-init = PPCA sampling */
} replayLoaderConfig_t;

/* Training wrapper; appends r replayed samples per eligible class after the
 * real samples and borrows base's fields and the replay source. Free it only
 * with freeReplayDataLoader; base stays caller-owned. Pools inherit the first
 * real sample's shape/storage; labels must be FLOAT32 with numClasses
 * elements. Drawing modes (PPCA_SAMPLE, EXEMPLAR) need a step key whenever a
 * class is eligible; slot i of class c in update u draws under
 * STEP(u)/(CL_REPLAY, 0)/(CLASS, c)/(SLOT, i), so a replayed sample depends on
 * (update, class, slot) alone. Each resulting batch size must stay divisible
 * by the training microbatch size, which training checks only per batch.
 * samplesPerClass must fit uint32_t. */
dataLoader_t *replayDataLoaderWrap(dataLoader_t *base, const replayLoaderConfig_t *cfg);
void freeReplayDataLoader(dataLoader_t *wrapped);

#endif // ODT_PPCA_REPLAY_API_H
