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
void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInitDataStorageSuccessful);
    RUN_TEST(testInitDataStorageGetInitError);
    UNITY_END();
}
