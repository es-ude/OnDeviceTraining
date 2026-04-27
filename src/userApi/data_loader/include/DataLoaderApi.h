#ifndef DATALOADERAPI_H
#define DATALOADERAPI_H

#include "DataLoader.h"

/*!
 * Initializes a dataloader_t.
 * Uses reserveMemory().
 *
 * \param getSample: Pointer to function to be used for getBatch()
 * \param getDatasetSize: Pointer to function to calculate dataset size
 * \param batchSize: batch size to be used
 * \param transform: Pointer to transform function to be used once after loading
 * \param targetTransform: Pointer to transform function to be used for each getBatch()
 * \param shuffle: Shuffle or not
 * \param shuffleSeed: Seed for shuffling
 * \param dropLast: If last batch can't be filled, drop or not
 * \return Pointer to initialized data loader
 */
dataLoader_t *dataLoaderInit(getSampleFn_t getSample, getDatasetSizeFn_t getDatasetSize,
                             uint16_t batchSize, transformFn_t transform,
                             transformFn_t targetTransform, bool shuffle, uint64_t shuffleSeed,
                             bool dropLast);

/*!
 * Free sample_t pointer.
 *
 * Only frees the sample_t pointer. We assume, that the contained tensors
 * are stored in the dataset and thus are not directly mutable.
 *
 * \param sample: Pointer to be freed
 * \return void
 */
void freeSample(sample_t *sample);

/*!
 * Free batch_t pointer and contained **sample_t.
 *
 * Only frees the batch_t and sample_t pointer! We assume, that the contained tensors
 * are stored in the dataset and thus are not directly mutable.
 *
 * \param batch: Pointer to be freed
 * \return void
 */
void freeBatch(batch_t *batch);

/*!
 * Free dataloader pointer and contained indices array.
 *
 * Only frees the dataloader_t and indices pointer.
 *
 * \param dataloader: Pointer to be freed
 * \return void
 */
void freeDataLoader(dataLoader_t *dataloader);

#endif // DATALOADERAPI_H
