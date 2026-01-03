#define SOURCE_FILE "DATA_STORAGE"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "DataStorage.h"

struct dataStorage {
    dataEntry_t *entries;
    size_t currentNumberOfEntries;
    size_t maxNumberOfEntries;

    size_t currentSizeOfUsedDataStorage;
    size_t maxSizeOfDataStorage; // numbytes
};

static dataStorage_t *storage = NULL;

/* region PUBLIC HEADER FUNCTIONS */
dataStorageErrorCode_t initDataStorage(size_t totalSizeForAllocatedMemory, uint8_t memoryFraction) {
    dataStorageErrorCode_t errorCode = DATASTORAGE_NO_ERROR;
    if (storage) {
        errorCode = DATASTORAGE_INIT_ERROR;
        return errorCode;
    }
    storage = calloc(1, sizeof(dataStorage_t));
    if (!storage) {
        errorCode = DATASTORAGE_ALLOC_ERROR;
        return errorCode;
    }
    storage->entries = calloc(1, totalSizeForAllocatedMemory);
    if (!storage->entries) {
        errorCode = DATASTORAGE_ALLOC_ERROR;
        return errorCode;
    }
    size_t sizeForEntriesStorage =
        calculateSizeForEntriesStorage(totalSizeForAllocatedMemory, memoryFraction);

    storage->maxSizeOfDataStorage = totalSizeForAllocatedMemory - sizeForEntriesStorage;
    storage->maxNumberOfEntries = sizeForEntriesStorage / sizeof(dataEntry_t);

    storage->currentSizeOfUsedDataStorage = 0;
    storage->currentNumberOfEntries = 0;
    return errorCode;
}

void deinitDataStorage(void) {
    if (!storage) {
        return;
    }
    free(storage->entries);
    free(storage);

    storage = NULL;
    return;
}
/* endregion PUBLIC HEADER FUNCTIONS */

/* region INTERNAL HEADER FUNCTIONS */
static size_t calculateSizeForEntriesStorage(size_t totalSizeForDataAndEntriesStorage,
                                             const uint8_t entryMemoryFraction) {
    size_t prealignedEntriesStorageSize =
        (size_t)(totalSizeForDataAndEntriesStorage / entryMemoryFraction);
    size_t remainder = prealignedEntriesStorageSize % sizeof(dataEntry_t);
    size_t entriesStorageSize = prealignedEntriesStorageSize - remainder;
    return entriesStorageSize;
}
/* endregion INTERNAL HEADER FUNCTIONS */
