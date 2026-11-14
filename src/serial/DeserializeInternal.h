#ifndef DESERIALIZEINTERNAL_H
#define DESERIALIZEINTERNAL_H

#include <stdio.h>

#include "ArithmeticType.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

/*! Fails fast if the file rank differs from the skeleton's: its dimension
 *  arrays were sized by the build-time rank. */
static void deserializeShape(shape_t *shape, FILE *f);

/*! numberOfElements is the record's element count, used to validate grouping;
 *  0 means no tensor geometry is available (layer outputQ/propLossQ wire
 *  configs). */
static void deserializeQuantization(quantization_t *q, FILE *f, size_t numberOfElements);

static void deserializeArithmetic(arithmetic_t *arithmetic, FILE *f);
static void deserializeKernel(kernel_t *kernel, FILE *f);

/*! Requires factory-compatible qConfig storage with owned group arrays (or
 *  NULL arrays with numGroups 0). Group counts may resize those arrays. Pass the record's element
 * count to validate grouping; zero means no tensor geometry is available. */
static void deserializeQConfig(quantization_t *q, FILE *f, size_t numberOfElements);

/*! Parses past one tensor record without writing any output, for a grad record
 *  a frozen skeleton has no place for. The header gets every check
 *  readTensorRecordHeader applies; the payload is skipped with fseek, so the
 *  stream must be seekable. Fails fast when the payload exceeds LONG_MAX bytes
 *  (the fseek offset cannot express it) or extends past end of file. Leaves
 *  the stream right after the record. */
static void skipSerializedTensor(FILE *f);

static void deserializeLayer(layer_t *layer, FILE *f);

#endif // DESERIALIZEINTERNAL_H
