#ifndef DATALOADERAPI_H
#define DATALOADERAPI_H

#include "DataLoader.h"

/*!
 * Creates a loader over caller-owned dataset callbacks. Requires batchSize >= 1
 * and dropLast = true; the dataset size must stay stable while the loader owns its
 * indices. transform and targetTransform are stored but never called.
 *
 * Call order: a shuffle = true loader starts in the identity order and serves it
 * until dataLoaderShuffle sets its keyed order. trainingRun shuffles its loaders
 * itself; a caller that fetches from a shuffle = true loader directly (getBatch,
 * trainingEpochDefault, an evaluation entry point) calls dataLoaderShuffle first,
 * with a key from the trainingRng helpers (trainingRngShuffleKey).
 *
 * Release with freeDataLoader.
 */
dataLoader_t *dataLoaderInit(getSampleFn_t getSample, getDatasetSizeFn_t getDatasetSize,
                             uint16_t batchSize, transformFn_t transform,
                             transformFn_t targetTransform, bool shuffle, bool dropLast);

/*! Frees the sample wrapper; its tensors remain dataset-owned. */
void freeSample(sample_t *sample);

/*! Frees the batch and its pointer array. Release each sample wrapper separately
 * (freeSample); the tensors remain dataset-owned. */
void freeBatch(batch_t *batch);

/*! Releases the loader-owned indices and the loader. */
void freeDataLoader(dataLoader_t *dataloader);

#endif // DATALOADERAPI_H
