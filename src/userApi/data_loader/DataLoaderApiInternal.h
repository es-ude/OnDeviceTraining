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
 * \return Pointer to batch
 */
static batch_t *getBatch(dataLoader_t *dataLoader, size_t index);

#endif // DATALOADERAPIINTERNAL_H
