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

    transformFn_t transform;       /*!< Stored only; the loader never calls it */
    transformFn_t targetTransform; /*!< Stored only; the loader never calls it */

    bool shuffle;
    size_t *indices; /*!< Sample order, getDatasetSize many entries */

    bool reshufflePerEpoch; /*!< Controls per-epoch keyed reshuffling in trainingRun; default
                                false. */

    bool dropLast;
};

/*! Internal initializer; dataLoaderInit is the public entry. The caller supplies
 * valid callbacks, batchSize >= 1, dropLast = true, and indices storage for the
 * dataset size, which must stay stable while the loader uses it. The order
 * starts as the identity. */
void initDataLoader(dataLoader_t *dataLoader, getSampleFn_t getSample,
                    getDatasetSizeFn_t getDatasetSize, getBatchFn_t getBatch, uint16_t batchSize,
                    transformFn_t transform, transformFn_t targetTransform, bool shuffle,
                    size_t *indices, bool dropLast);

/*! Controls per-epoch keyed reshuffling in trainingRun; default false. */
void dataLoaderSetReshufflePerEpoch(dataLoader_t *dataLoader, bool enable);

/*! Sets the sample order of a shuffle = true loader: the identity order,
 * shuffled with Fisher-Yates (rngNextBounded, from the last slot down) over the
 * stream of `key`. The order depends on the key alone, never on the order
 * before the call, so one key always gives one order. A loader with
 * shuffle = false keeps its order and needs no key; fewer than two samples
 * draw nothing and need no key either. Otherwise a NULL key stops the program
 * with a named error. trainingRun shuffles its loaders itself; the call-order
 * contract for direct callers is in DataLoaderApi.h.
 *
 * \param dataLoader: Pointer to dataloader to order
 * \param key: Key of this order; read during the call only
 */
void dataLoaderShuffle(dataLoader_t *dataLoader, const rngKey_t *key);

#endif // DATALOADER_H
