#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdbool.h>

#include "Dataset.h"
#include "Tensor.h"

typedef struct dataLoader dataLoader_t;

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngKey rngKey_t;

typedef tensor_t *(*transformFn_t)(tensor_t *tensor);
typedef sample_t *(*getSampleFn_t)(size_t id);
typedef size_t (*getDatasetSizeFn_t)();
/* stepKey: the key of the update the batch feeds, NULL outside training (an
 * evaluation fetch, a probe). A loader that draws -- the replay wrapper --
 * draws under it and hands it to the loader it wraps; a plain loader ignores
 * it. Read during the call only. */
typedef batch_t *(*getBatchFn_t)(dataLoader_t *dataLoader, size_t index, const rngKey_t *stepKey);

struct dataLoader {
    getSampleFn_t getSample;           /*!< Pointer to function to get a sample from the dataset*/
    getDatasetSizeFn_t getDatasetSize; /*!< Pointer to function to get the size of the dataset */

    uint16_t batchSize;
    getBatchFn_t getBatch; /*!< Pointer to function to get a batch with batchSize many samples form
                              the dataset */

    transformFn_t transform; /*!< Pointer to function to be used once after loading the dataset */
    transformFn_t targetTransform; /*!< Pointer to function to be used after each getBatch */

    bool shuffle;
    size_t *indices; /*!< List of indices with getDatasetSize many entries. Used for shuffling */
    bool shuffled;   /*!< True once the order of a shuffle = true loader has been set; the
                          default getBatch stops while it is false. */

    bool reshufflePerEpoch; /*!< Opt-in: trainingRun shuffles the train loader at the start of
                                every epoch under that epoch's shuffle key (true) or under
                                epoch 0's key (false), so a loader without it keeps one
                                order. Default false (zero-init; initDataLoader also sets it
                                explicitly). */

    bool dropLast;
};

/*! Initializes given data loader.
 *
 * \param dataLoader: Pointer to dataloader to initialize
 * \param getSample: Pointer to function to be to get sample from dataset
 * \param getDatasetSize: Pointer to function to get size of dataset
 * \param getBatch: Pointer to function to get batch from dataset
 * \param batchSize: Batch size
 * \param transform: Pointer to function to be used once after loading the dataset
 * \param targetTransform: Pointer to function to be used for each batch
 * \param shuffle: Use shuffle, or not. The order starts as the identity and a
 *                 shuffle = true loader serves no batch until dataLoaderShuffle
 *                 has set its order.
 * \param indices: List of indices for shuffling
 * \param dropLast: If last batch cannot be filled, drop batch
 */
void initDataLoader(dataLoader_t *dataLoader, getSampleFn_t getSample,
                    getDatasetSizeFn_t getDatasetSize, getBatchFn_t getBatch, uint16_t batchSize,
                    transformFn_t transform, transformFn_t targetTransform, bool shuffle,
                    size_t *indices, bool dropLast);

/*! Enables/disables opt-in per-epoch reshuffling (#381). Default off.
 *
 * \param dataLoader: Pointer to dataloader to configure
 * \param enable: true to give every epoch of trainingRun its own order
 */
void dataLoaderSetReshufflePerEpoch(dataLoader_t *dataLoader, bool enable);

/*! Sets the sample order of a shuffle = true loader: the identity order,
 * shuffled with Fisher-Yates (the loop and swap of rngShuffleIndices) over the
 * stream of `key`. The order depends on the key alone, never on the order
 * before the call, so one key always gives one order. A loader with
 * shuffle = false keeps its order and needs no key; fewer than two samples
 * draw nothing and need no key either. Otherwise a NULL key stops the program
 * with a named error.
 *
 * trainingRun shuffles its loaders itself (the train loader at the start of
 * every epoch, a shuffle = true eval loader once per run). A caller that drives
 * trainingEpochDefault, getBatch or an evaluation entry point directly shuffles
 * first, with a key from the trainingRng helpers (trainingRngShuffleKey).
 *
 * \param dataLoader: Pointer to dataloader to order
 * \param key: Key of this order; read during the call only
 */
void dataLoaderShuffle(dataLoader_t *dataLoader, const rngKey_t *key);

#endif // DATALOADER_H
