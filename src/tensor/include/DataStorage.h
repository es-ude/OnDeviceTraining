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

/*!
 * @brief Adds slot for writing data to the storage and creates a corresponding entry.
 * Does not write any Data.
 *
 * The function handles the storage management for the requested size of data.
 * A pointer to a pointer to the allocated data region is returned to the caller
 * via @p data.
 *
 * @param[in]  sizeOfData
 *             Size of the data (in bytes).
 * @param[out] data
 *             Pointer to a pointer to the allocated
 *             data region.
 *
 * @return The error code indicating the result of the  operation:
 *         - DATASTORAGE_NO_ERROR: storage management was successful.
 *         - DATASTORAGE_NOT_INITIALIZED_ERROR: the storage object does not exist.
 *         - DATASTORAGE_FRAGMENTATION_ERROR: the storage needs to be fragmented.
 *         - DATASTORAGE_SIZE_ERROR: the storage can't hold that much data.
 *         - DATASTORAGE_MEMORY_FRACTION_ERROR: insufficient space for data or entries. This could
 * mean many entries with small data size or few entries with huge data size.
 */
dataStorageErrorCode_t addDataToStorage(const size_t sizeOfData, void **data);
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

/*!
 * @brief Evaluates if new data and entries can be added to storage.
 *
 * @param[in] sizeOfNewData
 *            Size of all data (in bytes) regardless the amount of entries.
 * @param[in] amountOfEntriesToAdd
 *            Number of new entries to add.
 *
 * @return Error code indicating the evaluation result:
 *         - DATASTORAGE_NO_ERROR: enough space is available
 *         - DATASTORAGE_NOT_INITIALIZED_ERROR: storage is not initialized
 *         - DATASTORAGE_SIZE_ERROR: the storage can't hold that much data.
 *         - DATASTORAGE_MEMORY_FRACTION_ERROR: insufficient space for data or entries. This could
 * mean many entries with small data size or few entries with huge data size.
 */
static dataStorageErrorCode_t evaluateStorageForNewData(size_t sizeOfNewData,
                                                        size_t amountOfEntriesToAdd);
/*!
 * @brief Returns the index of the most recently added entry.
 *
 * This (internal) function returns the index of the entry that was added last
 * and therefore represents the entry with the highest address in the data array.
 *
 * @return Index of the last added entry in the data array.
 */
static size_t getIndexOfLastEntry(void);



/*!
 *@brief Assigns the data pointer and size for a storage entry.
 *
 * @param[in] sizeOfData     Size of the data to be stored (in bytes).
 * @param[in] indexOfEntry  Index of the storage entry to update.
 * @return Error code indicating the evaluation result:
 *         - DATASTORAGE_NO_ERROR: writing datapointer of entry was successful.
 *         - DATASTORAGE_FRAGMENTATION_ERROR: the storage needs to be fragmented.
 */
static dataStorageErrorCode_t setDataPointerAndSizeOfEntry(const size_t sizeOfData,
                                                           const size_t indexOfEntry);
/*!
 * @brief Creates a new storage entry.
 *
 * This (internal) function assigns a suiting data pointer and the given data size to the first free
 * storage entry (with the lowest index) and returns a pointer to the data pointer via @p data.
 * @param[out]  data        Pointer to the data pointer to be stored.
 * @param[in]  sizeOfData  Size of the data to be stored (in bytes).
 *
 * @return Error code indicating the result:
 *         - DATASTORAGE_NO_ERROR: entry created successfully.
 *         - DATASTORAGE_FRAGMENTATION_ERROR: the storage needs to be fragmented.
 */
static dataStorageErrorCode_t createEntry(void **data, const size_t sizeOfData);

static void initFirstEntry (size_t sizeForEntriesStorage);
/* endregion INTERNAL HEADER FUNCTIONS */
#endif // ENV5_RUNTIME_DATASTORAGE_H
