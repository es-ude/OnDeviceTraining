#include "DataStorage.h"
#include "unity.h"

static uint16_t totalSizeForAllocatedMemory = 1000;
static uint8_t memoryFraction = 4;
static uint16_t sizeOfNewData = 0;
void **dataPTR = NULL;

/*! should work if called first (before other init tests) */
void testInitDataStorageSuccessful() {
    dataStorageErrorCode_t errorCode;
    errorCode = initDataStorage(totalSizeForAllocatedMemory, memoryFraction);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NO_ERROR, errorCode);
}
/*! should work if called afer testInitDataStorageSuccessful */
void testInitDataStorageGetInitError() {
    dataStorageErrorCode_t errorCode = DATASTORAGE_NO_ERROR;
    errorCode = initDataStorage(totalSizeForAllocatedMemory, memoryFraction);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_INIT_ERROR, errorCode);
}
/*
 * would need dummy for calloc
 * void dataStorageInitGetAllocError(){
 *  dataStorageErrorCode_t errorCode = DATASTORAGE_NO_ERROR;
 *  errorCode = initDataStorage(100,4);
 *  TEST_ASSERT_EQUAL(DATASTORAGE_ALLOC_ERROR, errorCode);
 * }
 */
void testAddDataToStorageSuccessful() {
    sizeOfNewData = 50;
    dataStorageErrorCode_t errorCode;
    errorCode = addDataToStorage(sizeOfNewData, dataPTR);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NO_ERROR, errorCode);
}

void testAddDataToStorageFirstEntryCorrectInitialized(void) {
    sizeOfNewData = 1;
    TEST_ASSERT_EQUAL(0, addDataToStorage(sizeOfNewData, dataPTR));
    uint8_t *dataInStorage = (uint8_t *)*dataPTR;
    TEST_ASSERT_NOT_NULL(dataInStorage);
    TEST_ASSERT_EQUAL_UINT8(0, dataInStorage[0]);
}

void testAddDataToStorageCanWriteCorrectValue() {
    sizeOfNewData = 100;
    addDataToStorage(sizeOfNewData, dataPTR);
    uint8_t *dataInStorage = (uint8_t *)*dataPTR;

    if (dataInStorage) {
        ((uint8_t *)(*dataPTR))[0] = 42;
        TEST_ASSERT_EQUAL_UINT8(42, ((uint8_t *)(*dataPTR))[0]);
    } else {
        TEST_FAIL_MESSAGE("dataInStorage does not exist");
    }
}

void testAddDataToStorageGetSizeError() {
    sizeOfNewData = 1000;
    dataStorageErrorCode_t errorCode;
    errorCode = addDataToStorage(sizeOfNewData, dataPTR);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_SIZE_ERROR, errorCode);
}

void testAddDataToStorageGetNotInitializedError() {
    sizeOfNewData = 50;
    dataStorageErrorCode_t errorCode;
    errorCode = addDataToStorage(sizeOfNewData, dataPTR);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NOT_INITIALIZED_ERROR, errorCode);
}

void testAddDataToStorageGetMemoryFractionErrorCaseData() {
    sizeOfNewData = totalSizeForAllocatedMemory - totalSizeForAllocatedMemory / memoryFraction +
                    sizeof(dataEntry_t);
    dataStorageErrorCode_t errorCode;
    errorCode = addDataToStorage(sizeOfNewData, dataPTR);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_MEMORY_FRACTION_ERROR, errorCode);
}

void testAddDataToStorageGetMemoryFractionErrorCaseEntries() {
    sizeOfNewData = 1;
    dataStorageErrorCode_t errorCode;
    for (int i = 0; i < totalSizeForAllocatedMemory / memoryFraction; i++) {
        errorCode = addDataToStorage(sizeOfNewData, dataPTR);
        if (errorCode != DATASTORAGE_NO_ERROR) {
            break;
        }
    }
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_MEMORY_FRACTION_ERROR, errorCode);
}

void testRemoveDataFromStorageSuccessful() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 100;
    addDataToStorage(sizeOfNewData, dataPTR);
    errorCode = removeDataFromStorage(dataPTR);
    TEST_ASSERT_EQUAL_HEX8(0, errorCode);
}

void testRemoveDataFromStorageCorrectValue() {
    sizeOfNewData = 100;
    addDataToStorage(sizeOfNewData, dataPTR);
    TEST_ASSERT_NOT_NULL((uint8_t *)*dataPTR);
    removeDataFromStorage(dataPTR);
    TEST_ASSERT_NULL((uint8_t *)*dataPTR);
}

void testRemoveDataFromStorageGeIndexError() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 100;
    addDataToStorage(sizeOfNewData, dataPTR);
    errorCode = removeDataFromStorage(dataPTR + 5);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_INDEX_ERROR, errorCode);
}

void testResizeDataInStorageSuccessful() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 10;
    addDataToStorage(sizeOfNewData, dataPTR);
    errorCode = resizeDataInStorage(dataPTR, 100);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NO_ERROR, errorCode);
}

void testResizeDataInStorageGrowEntry() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 10;
    addDataToStorage(sizeOfNewData, dataPTR);
    errorCode = resizeDataInStorage(dataPTR, 100);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NO_ERROR, errorCode);
}

void testResizeDataInStorageShrinkEntry() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 100;
    addDataToStorage(sizeOfNewData, dataPTR);
    errorCode = resizeDataInStorage(dataPTR, 10);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NO_ERROR, errorCode);
}

void testResizeDataInStorageGetNotInitializedError() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 10;
    addDataToStorage(sizeOfNewData, dataPTR);
    errorCode = resizeDataInStorage(dataPTR, 100);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NOT_INITIALIZED_ERROR, errorCode);
}

/*! 1000/4 = 250
        * totalSizeForAllocatedMemory/memoryFraction = a
        ----
       * sizeof(dataEntry_t) = 16
        * ----
       * -> 250/16 = 15.625
        * -> a/sizeof(dataEntry_t)
        * ----
        * -> storage for entries should be: 240 bytes with 15 entries
        */
void testResizeDataInStorageGetSizeError() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 15;
    for (int i = 0; i < 16; i++) {
        addDataToStorage(sizeOfNewData, dataPTR);
    }
    errorCode = resizeDataInStorage(dataPTR, 1000);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_SIZE_ERROR, errorCode);
}

void testResizeDataInStorageGetMemoryFractionError() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 750;
    errorCode = addDataToStorage(sizeOfNewData, dataPTR);
    errorCode = resizeDataInStorage(dataPTR, 100);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_MEMORY_FRACTION_ERROR, errorCode);
}

void testResizeDataInStorageGetIndexError() {
    dataStorageErrorCode_t errorCode;
    sizeOfNewData = 10;
    addDataToStorage(sizeOfNewData, dataPTR);
    errorCode = resizeDataInStorage(dataPTR + 10, 100);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_INVALID_ENTRY_ERROR, errorCode);
}

void setUp() {
    dataPTR = calloc(1, sizeof(void *));
}
void tearDown() {
    free(dataPTR);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInitDataStorageSuccessful);
    RUN_TEST(testInitDataStorageGetInitError);
    RUN_TEST(testAddDataToStorageSuccessful);
    RUN_TEST(testAddDataToStorageFirstEntryCorrectInitialized);
    RUN_TEST(testAddDataToStorageCanWriteCorrectValue);
    RUN_TEST(testAddDataToStorageGetSizeError);
    deinitDataStorage();
    RUN_TEST(testAddDataToStorageGetNotInitializedError);
    initDataStorage(totalSizeForAllocatedMemory, memoryFraction);
    RUN_TEST(testAddDataToStorageGetMemoryFractionErrorCaseData);
    RUN_TEST(testAddDataToStorageGetMemoryFractionErrorCaseEntries);
    deinitDataStorage();
    initDataStorage(totalSizeForAllocatedMemory, memoryFraction);
    RUN_TEST(testRemoveDataFromStorageSuccessful);
    RUN_TEST(testRemoveDataFromStorageCorrectValue);
    RUN_TEST(testRemoveDataFromStorageGeIndexError);
    RUN_TEST(testResizeDataInStorageSuccessful);
    RUN_TEST(testResizeDataInStorageGrowEntry);
    RUN_TEST(testResizeDataInStorageShrinkEntry);
    deinitDataStorage();
    RUN_TEST(testResizeDataInStorageGetNotInitializedError);
    initDataStorage(totalSizeForAllocatedMemory, memoryFraction);
    RUN_TEST(testResizeDataInStorageGetSizeError);
    deinitDataStorage();
    initDataStorage(totalSizeForAllocatedMemory, memoryFraction);
    RUN_TEST(testResizeDataInStorageGetMemoryFractionError);
    RUN_TEST(testResizeDataInStorageGetIndexError);

    UNITY_END();
}
