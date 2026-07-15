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
    REPLAY_MODE_CLASS_MEAN,      /* replay the running class centroid — the
                                    one-vector-per-class baseline (#326) */
    REPLAY_MODE_EXEMPLAR,        /* replay stored raw samples — first-K
                                    exemplar-buffer baseline (#326); slots are
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
    uint32_t minCount;           /* class eligible once generator count >= minCount
                                    (PPCA_SAMPLE/CLASS_MEAN; EXEMPLAR gates on a
                                    non-empty class buffer instead) */
    rng32_t *stream;             /* caller-owned sampling stream; unused (NULL ok)
                                    in REPLAY_MODE_CLASS_MEAN */
    replayMode_t mode;           /* zero-init = PPCA sampling */
} replayLoaderConfig_t;

/* Wraps base: getBatch appends r synthetic samples per eligible class
 * AFTER the real samples (sample 0 stays real -> labelRef/meanScale
 * contracts hold; the optimizer's MEAN divisor uses batch->size and thus
 * scales automatically — the replay layer adds NO second divisor).
 * Pools are lazily initialized from the first real batch (item + label
 * shape/dtype/qconfig mirrored via getShapeLike/getQLike, labels must be
 * FLOAT32 one-hot [numClasses]). The wrapper BORROWS base's fields; free
 * it ONLY with freeReplayDataLoader (freeDataLoader on the wrapper would
 * free base's indices). Base stays owned by the caller. */
dataLoader_t *replayDataLoaderWrap(dataLoader_t *base, const replayLoaderConfig_t *cfg);
void freeReplayDataLoader(dataLoader_t *wrapped);

#endif // ODT_PPCA_REPLAY_API_H
