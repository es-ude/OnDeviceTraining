#ifndef SERIALINTERNAL_H
#define SERIALINTERNAL_H

#include <stdio.h>

#include "Layer.h"
#include "Tensor.h"

/*! Serializes array of values by given size to file.
 *
 * \param values: Pointer to first element of array
 * \param numberOfElements: Number of values in array
 * \param sizeOfElement: Size of each element
 * \param f: Pointer of file to serialize to
 */
static void serialize(void *values, size_t numberOfElements, size_t sizeOfElement, FILE *f);

/*! Serializes shape of tensor to given file.
 *
 * \param shape: Pointer to shape_t struct
 * \param f: Pointer of file to serialize to
 */
static void serializeShape(shape_t *shape, FILE *f);

/*! Serializes quantization of tensor to given file.
 *
 * \param q: Pointer to quantization_t
 * \param f: Pointer of file to serialize to
 */
static void serializeQuantization(quantization_t *q, FILE *f);

/*! Serializes data field of tensor to given file.
 *
 * \param data: Pointer to first element of data array
 * \param numberOfValues: Number of values in array
 * \param bytesPerValue: Nu ber of bytes per value
 * \param f: Pointer of file to serialize to
 */
static void serializeData(uint8_t *data, size_t numberOfValues, size_t bytesPerValue, FILE *f);

/*! Serializes quantization config of tensor to given file.
 *
 * \param q: Pointer to quantization_t
 * \param f: Pointer of file to serialize to
 */
static void serializeQConfig(quantization_t *q, FILE *f);

/*! Not implemented yet!
 */
static void serializeSparsity();

/*! Serializes layer of model to given file.
 *
 * \param layer: Pointer to layer
 * \param f: Pointer of file to serialize to
 */
static void serializeLayer(layer_t *layer, FILE *f);

#endif // SERIALINTERNAL_H
