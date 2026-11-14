#ifndef NPYLOADERAPI_H
#define NPYLOADERAPI_H

#include "Dataset.h"

/*!
 * Loads a dataset .npy file: interprets the leading dimension as the sample count and
 * returns one tensor per sample in its remaining natural shape ([N, C, L] gives N
 * tensors of [C, L]). Rejects malformed files (see npyReadHeader), scalar datasets and
 * empty arrays with exit(1) and a message naming the path. Release with freeTensorArray.
 */
tensorArray_t *npyLoad(char *path);

/*!
 * Use for weights/parameters: returns one owned contiguous tensor preserving every
 * on-disk dimension, the leading one included. Scalars are supported; malformed files
 * and empty arrays are rejected like npyLoad. Release with freeTensor.
 */
tensor_t *npyLoadFlat(char *path);

/*! Returns an owned sample wrapper borrowing the dataset's tensors; index must exist in
 * both the item and the label array. Release with freeSample. */
sample_t *npyGetSample(dataset_t *dataset, size_t index);

/*! Destroys an npyLoad-owned array and every tensor it contains; do not pass stack or
 * static containers. */
void freeTensorArray(tensorArray_t *tensorArr);

#endif // NPYLOADERAPI_H
