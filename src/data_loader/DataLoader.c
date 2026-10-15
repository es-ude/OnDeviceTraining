#define SOURCE_FILE "DATA_LOADER"

#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "DataLoader.h"
#include "RNG.h"

void initDataLoader(dataLoader_t *dataLoader, getSampleFn_t getSample,
                    getDatasetSizeFn_t getDatasetSize, getBatchFn_t getBatch, uint16_t batchSize,
                    transformFn_t transform, transformFn_t targetTransform, bool shuffle,
                    uint32_t shuffleSeed, size_t *indices, bool dropLast) {

    if (dropLast == false) {
        PRINT_ERROR("dropLast == false is not supported!");
        exit(1);
    }

    if (batchSize == 0) {
        /* Every consumer computes datasetSize / batchSize. */
        PRINT_ERROR("initDataLoader: batchSize must be >= 1");
        exit(1);
    }

    dataLoader->getSample = getSample;
    dataLoader->getDatasetSize = getDatasetSize;

    dataLoader->batchSize = batchSize;
    dataLoader->getBatch = getBatch;

    dataLoader->transform = transform;
    dataLoader->targetTransform = targetTransform;

    dataLoader->shuffle = shuffle;
    dataLoader->shuffleSeed = shuffleSeed;
    dataLoader->indices = indices;

    dataLoader->reshufflePerEpoch = false;

    dataLoader->dropLast = dropLast;

    size_t sizeDataset = getDatasetSize();

    for (size_t i = 0; i < sizeDataset; ++i) {
        indices[i] = i;
    }
    /* The order of a shuffle = true loader stays unset until dataLoaderShuffle. */
    dataLoader->shuffled = false;
}

void dataLoaderSetReshufflePerEpoch(dataLoader_t *dataLoader, bool enable) {
    dataLoader->reshufflePerEpoch = enable;
}

void dataLoaderShuffle(dataLoader_t *dataLoader, const rngKey_t *key) {
    if (!dataLoader->shuffle) {
        return;
    }
    size_t n = dataLoader->getDatasetSize();
    if (n >= 2 && key == NULL) {
        PRINT_ERROR("dataLoaderShuffle: shuffle without a key");
        exit(1);
    }
    /* From the identity order, so the order depends on the key alone. */
    size_t *indices = dataLoader->indices;
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }
    if (n >= 2) {
        rngStream_t stream = rngOpen(*key);
        for (size_t i = n - 1; i > 0; --i) {
            size_t j = rngNextBounded(&stream, i + 1);
            size_t swapped = indices[i];
            indices[i] = indices[j];
            indices[j] = swapped;
        }
    }
    dataLoader->shuffled = true;
}

tensor_t *transform(tensor_t *tensorList);

tensor_t *targetTransform(tensor_t *tensor);
