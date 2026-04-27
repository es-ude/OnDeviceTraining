#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdbool.h>

#include "Dataset.h"
#include "Tensor.h"

typedef struct dataLoader dataLoader_t;

typedef tensor_t *(*transformFn_t)(tensor_t *tensor);
typedef sample_t *(*getSampleFn_t)(size_t id);
typedef size_t (*getDatasetSizeFn_t)();
typedef batch_t *(*getBatchFn_t)(dataLoader_t *dataLoader, size_t index);

struct dataLoader {
    getSampleFn_t getSample;           /*!< Pointer to function to get a sample from the dataset*/
    getDatasetSizeFn_t getDatasetSize; /*!< Pointer to function to get the size of the dataset */

    uint16_t batchSize;
    getBatchFn_t getBatch; /*!< Pointer to function to get a batch with batchSize many samples form
                              the dataset */

    transformFn_t transform; /*!< Pointer to function to be used once after loading the dataset */
    transformFn_t targetTransform; /*!< Pointer to function to be used after each getBatch */

    bool shuffle;
    uint64_t shuffleSeed;
    size_t *indices; /*!< List of indices with getDatasetSize many entries. Used for shuffling */

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
 * \param shuffle: Use shuffle, or not
 * \param shuffleSeed: Seed for shuffling for reproducability
 * \param indices: List of indices for shuffling
 * \param dropLast: If last batch cannot be filled, drop batch
 */
void initDataLoader(dataLoader_t *dataLoader, getSampleFn_t getSample,
                    getDatasetSizeFn_t getDatasetSize, getBatchFn_t getBatch, uint16_t batchSize,
                    transformFn_t transform, transformFn_t targetTransform, bool shuffle,
                    uint64_t shuffleSeed, size_t *indices, bool dropLast);

#endif // DATALOADER_H
