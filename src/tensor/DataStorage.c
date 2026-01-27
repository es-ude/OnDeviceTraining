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
        free(storage);
        storage = NULL;
        return errorCode;
    }
    size_t sizeForEntriesStorage =
        calculateSizeForEntriesStorage(totalSizeForAllocatedMemory, memoryFraction);

    initFirstEntry(sizeForEntriesStorage);

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
dataStorageErrorCode_t addDataToStorage(const size_t sizeOfNewData, void **data) {
    dataStorageErrorCode_t errorCode = DATASTORAGE_NO_ERROR;
    errorCode = evaluateStorageForNewData(sizeOfNewData, 1);
    if (errorCode != DATASTORAGE_NO_ERROR) {
        return errorCode;
    }
    errorCode = createEntry(data, sizeOfNewData);
    return errorCode;
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

static dataStorageErrorCode_t evaluateStorageForNewData(size_t sizeOfNewData,
                                                        size_t amountOfEntriesToAdd) {
    dataStorageErrorCode_t errorCode = DATASTORAGE_NO_ERROR;
    if (!storage) {
        errorCode = DATASTORAGE_NOT_INITIALIZED_ERROR;
        return errorCode;
    }
    size_t remainingSizeOfDataStorage =
        storage->maxSizeOfDataStorage - storage->currentSizeOfUsedDataStorage;
    if (remainingSizeOfDataStorage < sizeOfNewData) {
        if (storage->currentNumberOfEntries >=
            (storage->maxNumberOfEntries - amountOfEntriesToAdd)) {
            errorCode = DATASTORAGE_SIZE_ERROR;
            return errorCode;
        }

        size_t remainingSizeOfEntriesForData =
            (storage->maxNumberOfEntries -
             (storage->currentNumberOfEntries + amountOfEntriesToAdd)) *
            sizeof(dataEntry_t);
        size_t remainingSizeForNewData = remainingSizeOfEntriesForData + remainingSizeOfDataStorage;
        if (remainingSizeForNewData > sizeOfNewData) {
            errorCode = DATASTORAGE_MEMORY_FRACTION_ERROR;
        } else {
            errorCode = DATASTORAGE_SIZE_ERROR;
        }
        return errorCode;
    }
    if (storage->maxNumberOfEntries <= storage->currentNumberOfEntries) {
        errorCode = DATASTORAGE_MEMORY_FRACTION_ERROR;
    }
    return errorCode;
}
static size_t getIndexOfLastEntry() {
    int indexOfLastEntry = 0;
    for (int i = 0; i < storage->maxNumberOfEntries; i++) {
        if (!(storage->entries[i].pointerToData)) {
            continue;
        }
        if (storage->entries[i].pointerToData > storage->entries[indexOfLastEntry].pointerToData) {
            indexOfLastEntry = i;
        }
    }
    return indexOfLastEntry;
}

static dataStorageErrorCode_t setDataPointerAndSizeOfEntry(const size_t sizeOfData,
                                                           const size_t indexOfEntry) {
    dataStorageErrorCode_t errorCode = DATASTORAGE_NO_ERROR;
    size_t indexOfLastEntry = getIndexOfLastEntry();
    dataEntry_t lastEntry = storage->entries[indexOfLastEntry];
    uint8_t *newDataPTR = lastEntry.pointerToData + lastEntry.sizeOfData;
    uint8_t *highestAddress =
        storage + storage->maxSizeOfDataStorage + storage->maxNumberOfEntries * sizeof(dataEntry_t);
    if ((newDataPTR + sizeOfData) > highestAddress) {
        errorCode = DATASTORAGE_FRAGMENTATION_ERROR;
        return errorCode;
    }
    storage->entries[indexOfEntry].pointerToData = newDataPTR;
    storage->entries[indexOfEntry].sizeOfData = sizeOfData;
    return errorCode;
}

static dataStorageErrorCode_t createEntry(void **data, const size_t sizeOfData) {
    dataStorageErrorCode_t errorCode = DATASTORAGE_NO_ERROR;
    size_t index = 0;
    for (int j = 0; j < storage->maxNumberOfEntries; j++) {
        if (!(storage->entries[j].pointerToData)) {
            index = j;
            break;
        }
    }
    errorCode = setDataPointerAndSizeOfEntry(sizeOfData, index);
    if (errorCode != DATASTORAGE_NO_ERROR) {
        return errorCode;
    }
    *data = storage->entries[index].pointerToData;
    (storage->currentNumberOfEntries)++;
    (storage->currentSizeOfUsedDataStorage) += sizeOfData;
    return errorCode;
}

static void initFirstEntry (size_t sizeForEntriesStorage){
    storage->entries[0].pointerToData = storage->entries + sizeForEntriesStorage;
}

/* endregion INTERNAL HEADER FUNCTIONS */
