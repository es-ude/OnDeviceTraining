#ifndef DATALOADERAPIINTERNAL_H
#define DATALOADERAPIINTERNAL_H

#include <stddef.h>

#include "DataLoader.h"

/*!
 * Returns sample by (shuffled) index.
 *
 * \param dataLoader: Pointer to data loader
 * \param index: Index of sample
 * \return Pointer to sample
 */
static sample_t *getSampleByIndex(dataLoader_t *dataLoader, size_t index);

/*!
 * Returns batch by index.
 * Uses getSample function passed to dataLoaderInit.
 *
 * \param dataLoader: Pointer to data loader
 * \param index: Index of batch
 * \param stepKey: Key of the update the batch feeds; a plain loader draws nothing
 * \return Pointer to batch
 */
static batch_t *getBatch(dataLoader_t *dataLoader, size_t index, const rngKey_t *stepKey);

#endif // DATALOADERAPIINTERNAL_H
