#ifndef NPYLOADERAPI_H
#define NPYLOADERAPI_H

#include "Dataset.h"

/*!
 * Loads a .npy file as a tensor array, one tensor per entry of the leading dimension.
 *
 * Stops with exit(1) and a message naming the path when the file is malformed (see
 * npyReadHeader), its payload size does not match the shape, the shape is 0-d, or the
 * file holds zero samples or zero elements.
 *
 * \param path: Path to .npy file
 * \return Pointer to tensor array
 */
tensorArray_t *npyLoad(char *path);

/*!
 * Loads an entire .npy file as a SINGLE contiguous tensor, preserving the full
 * on-disk shape including the leading dimension.
 *
 * Use this for weight/parameter files, where the whole array is one tensor —
 * unlike npyLoad(), which interprets the leading dimension as a dataset's sample
 * count and returns one tensor per row (slicing dim0 away). Loading a weight of
 * shape [out, in, k] with npyLoad() yields `out` separate [in, k] tensors, so
 * only the first output channel is reachable as array[0]; npyLoadFlat() keeps
 * all `out * in * k` elements contiguous in one tensor.
 *
 * A 0-d shape loads as a single element. Stops with exit(1) like npyLoad() when the file
 * is malformed, its payload size does not match the shape, or it holds zero elements.
 *
 * The caller owns the returned tensor and must release it with freeTensor().
 *
 * \param path: Path to .npy file
 * \return Pointer to a tensor holding the file's full shape and data
 */
tensor_t *npyLoadFlat(char *path);

/*!
 * Gets a sample of given dataset.
 *
 * sample = item + label.
 *
 * \param dataset: Pointer to dataset
 * \param index: Index of sample
 * \return Pointer to sample
 */
sample_t *npyGetSample(dataset_t *dataset, size_t index);

/*!
 * Frees a tensorArray_t allocated by npyLoad: walks the inner array,
 * freeTensor()s every entry, then releases the array buffer and the
 * tensorArray_t struct itself. Stack- or static-allocated tensorArray_t
 * values must not be passed here.
 *
 * \param tensorArr: Pointer returned by npyLoad
 */
void freeTensorArray(tensorArray_t *tensorArr);

#endif // NPYLOADERAPI_H
