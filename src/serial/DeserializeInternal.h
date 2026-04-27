#ifndef DESERIALIZEINTERNAL_H
#define DESERIALIZEINTERNAL_H

#include <stdio.h>

#include "Layer.h"
#include "Tensor.h"

/*! Deserializes array of values by size from given file.
 *
 * \param values: Pointer to first element of array to deserialize into
 * \param numberOfElements: Number of values in array
 * \param sizeOfElement: Size of each element
 * \param f: Pointer of file to deserialize from
 */
static void deserialize(void *values, size_t numberOfElements, size_t sizeOfElement, FILE *f);

/*! Deserializes shape of tensor from given file.
 *
 * \param shape: Pointer to shape to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeShape(shape_t *shape, FILE *f);

/*! Deserializes quantization of tensor from given file.
 *
 * \param q: Pointer to quantization to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeQuantization(quantization_t *q, FILE *f);

/*! Deserializes quantization config of tensor from given file.
 *
 * \param q: Pointer to quantization to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeQConfig(quantization_t *q, FILE *f);

/*! Deserializes data of tensor from given file.
 *
 * \param data: Pointer to first element of data array to deserialize into
 * \param numberOfValues: Number of values
 * \param bytesPerValue: Bytes per value
 * \param f: Pointer of file to deserialize from
 */
static void deserializeData(uint8_t *data, size_t numberOfValues, size_t bytesPerValue, FILE *f);

/*! Not implemented yet!
 */
static void deserializeSparsity();

/*! Deserializes layer from given file.
 *
 * \param layer: Pointer to layer to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeLayer(layer_t *layer, FILE *f);

#endif // DESERIALIZEINTERNAL_H
