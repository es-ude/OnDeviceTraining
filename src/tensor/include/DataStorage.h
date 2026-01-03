#ifndef ENV5_RUNTIME_DATASTORAGE_H
#define ENV5_RUNTIME_DATASTORAGE_H
#include <stddef.h>
#include <stdint.h>

/* region TYPE_DEFINITIONS */
typedef struct dataEntry {
    uint8_t *pointerToData;
    size_t sizeOfData;
} dataEntry_t;

typedef struct dataStorage dataStorage_t;

typedef enum dataStorageErrorCode {
    DATASTORAGE_NO_ERROR = 0x00,
    DATASTORAGE_INIT_ERROR = 0x10,
    DATASTORAGE_FRAGMENTATION_ERROR = 0x15,
    DATASTORAGE_NOT_INITIALIZED_ERROR = 0x16,
    DATASTORAGE_MEMORY_FRACTION_ERROR = 0x17,
    DATASTORAGE_SIZE_ERROR = 0x18,
    DATASTORAGE_ALLOC_ERROR = 0x19,
    DATASTORAGE_UNDEFINED_ERROR = 0x20,

} dataStorageErrorCode_t;
/* endregion TYPE_DEFINITIONS */

/* region PUBLIC HEADER FUNCTIONS */

/*!
 * @brief Initializes the data storage object.
 *
 * This function creates a single instance of the data storage object. It allocates memory for the
 * storage of data and its management, and sets up the initial state of the storage.
 *
 * The memory is divided into two parts:
 * - The data management structure, which is allocated based on the `memoryFraction` parameter.
 * - The actual data, which is allocated from the remaining memory.
 *
 * For example, if the total memory size is 1024 bytes and the `memoryFraction` is 4, the data
 * management structure will be allocated 256 bytes (1/4 of the total memory size), leaving 768
 * bytes for the actual data.
 *
 * @param[in] totalSizeForAllocatedMemory The total size of the allocated memory, including both
 * data and data management.
 * @param[in] memoryFraction The fraction of memory allocated for data management.
 *
 * @return The error code indicating the result of the initialization:
 *         - DATASTORAGE_NO_ERROR: Initialization was successful.
 *         - DATASTORAGE_INIT_ERROR: The storage object already exists.
 *         - DATASTORAGE_ALLOC_ERROR: Memory allocation failed.
 */
dataStorageErrorCode_t initDataStorage(size_t totalSizeForAllocatedMemory, uint8_t memoryFraction);
/*!
 * @brief Deinitializes the data storage.
 *
 * This function releases all memory allocated for the data storage.
 * After deinitialization, the storage instance is set to NULL.
 *
 * Calling this function when the storage is not initialized has no effect.
 */
void deinitDataStorage(void);

uint8_t *getDataFromStorage(dataStorage_t storage, void *dataPTR);
dataEntry_t *addDataToStorage(dataStorage_t storage, void *dataPTR, size_t numberOfElements);
/* endregion PUBLIC HEADER FUNCTIONS */

/* region INTERNAL HEADER FUNCTIONS */
/*!
 * @brief Computes the aligned size of the entry storage area.
 *
 * This function calculates the number of bytes whis will be reserved for entry storage
 * (dataEntry_t), based on the specified memory fraction. The resulting size is aligned to a
 * multiple of `sizeof(dataEntry_t)` to ensure proper storage.
 *
 * @param[in] totalSizeForDataAndEntriesStorage
 *            Total allocated memory for data and entry storage.
 * @param[in] entryMemoryFraction
 *            Fraction of memory reserved for entry storage.
 *
 * @return Size in bytes which need to be allocated for entry storage, aligned to
 *         `sizeof(dataEntry_t)`.
 */
static size_t calculateSizeForEntriesStorage(size_t totalSizeForDataAndEntriesStorage,
                                             const uint8_t entryMemoryFraction);

/* endregion INTERNAL HEADER FUNCTIONS */
#endif // ENV5_RUNTIME_DATASTORAGE_H
