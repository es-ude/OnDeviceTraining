#include "DataStorage.h"
#include "unity.h"

static uint8_t totalSizeForAllocatedMemory = 1000;
static uint8_t memoryFraction = 4;
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
    uint8_t sizeOfNewData = 50;
    void *dataPTR = NULL;
    dataStorageErrorCode_t errorCode;
    errorCode = addDataToStorage(sizeOfNewData, &dataPTR);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NO_ERROR, errorCode);
}

void testAddDataToStorageGetSizeError() {
    uint8_t sizeOfNewData = 1000;
    void *dataPTR = NULL;
    dataStorageErrorCode_t errorCode;
    errorCode = addDataToStorage(sizeOfNewData, &dataPTR);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_SIZE_ERROR, errorCode);
}

void testAddDataToStorageGetNotInitializedError() {
    uint8_t sizeOfNewData = 50;
    void *dataPTR = NULL;
    dataStorageErrorCode_t errorCode;
    errorCode = addDataToStorage(sizeOfNewData, &dataPTR);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_NOT_INITIALIZED_ERROR, errorCode);
}

void testAddDataToStorageGetMemoryFractionErrorCaseData() {
    uint8_t sizeOfNewData = totalSizeForAllocatedMemory -
                            totalSizeForAllocatedMemory / memoryFraction + sizeof(dataEntry_t);
    void *dataPTR = NULL;
    dataStorageErrorCode_t errorCode;
    errorCode = addDataToStorage(sizeOfNewData, &dataPTR);
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_MEMORY_FRACTION_ERROR, errorCode);
}

void testAddDataToStorageGetMemoryFractionErrorCaseEntries() {
    uint8_t sizeOfNewData = 1;
    void *dataPTR = NULL;
    dataStorageErrorCode_t errorCode;
    for (int i = 0; i < totalSizeForAllocatedMemory / memoryFraction; i++) {
        errorCode = addDataToStorage(sizeOfNewData, &dataPTR);
        if (errorCode != DATASTORAGE_NO_ERROR) {
            break;
        }
    }
    TEST_ASSERT_EQUAL_HEX8(DATASTORAGE_MEMORY_FRACTION_ERROR, errorCode);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInitDataStorageSuccessful);
    RUN_TEST(testInitDataStorageGetInitError);
    RUN_TEST(testAddDataToStorageSuccessful);
    RUN_TEST(testAddDataToStorageGetSizeError);
    deinitDataStorage();
    RUN_TEST(testAddDataToStorageGetNotInitializedError);
    initDataStorage(totalSizeForAllocatedMemory, memoryFraction);
    RUN_TEST(testAddDataToStorageGetMemoryFractionErrorCaseData);
    RUN_TEST(testAddDataToStorageGetMemoryFractionErrorCaseEntries);
    UNITY_END();
}
