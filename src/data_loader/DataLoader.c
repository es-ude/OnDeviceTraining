#define SOURCE_FILE "DATA_LOADER"

#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "DataLoader.h"
#include "RNG.h"

void initDataLoader(dataLoader_t *dataLoader, getSampleFn_t getSample,
                    getDatasetSizeFn_t getDatasetSize, getBatchFn_t getBatch, uint16_t batchSize,
                    transformFn_t transform, transformFn_t targetTransform, bool shuffle,
                    uint64_t shuffleSeed, size_t *indices, bool dropLast) {

    if (dropLast == false) {
        PRINT_ERROR("dropLast == false is not supported!");
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

    dataLoader->dropLast = dropLast;

    size_t sizeDataset = getDatasetSize();

    for (size_t i = 0; i < sizeDataset; ++i) {
        indices[i] = i;
    }

    if (shuffle) {
        rngSetSeed(shuffleSeed);
        rngShuffleIndices(indices, sizeDataset);
    }
}

tensor_t *transform(tensor_t *tensorList);

tensor_t *targetTransform(tensor_t *tensor);
