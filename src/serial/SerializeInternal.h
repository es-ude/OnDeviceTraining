#ifndef SERIALINTERNAL_H
#define SERIALINTERNAL_H

#include <stdio.h>

#include "ArithmeticType.h"
#include "DyadicScale.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

/*! Serializes shape of tensor to given file: u32 LE rank + u32 LE
 *  dimensions[] + u32 LE orderOfDimensions[] (#370).
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

/*! Serializes a declared compute representation to given file: u8 type +
 *  u8 roundingMode.
 *
 * \param arithmetic: Pointer to arithmetic_t
 * \param f: Pointer of file to serialize to
 */
static void serializeArithmetic(arithmetic_t *arithmetic, FILE *f);

/*! Serializes kernel geometry (all kernel_t fields) to given file: u32 LE
 *  size, u8 paddingType, u32 LE stride, u32 LE dilation, u32 LE padding.
 *
 * \param kernel: Pointer to kernel_t
 * \param f: Pointer of file to serialize to
 */
static void serializeKernel(kernel_t *kernel, FILE *f);

/*! Writes one stored scale as its v9 dyadic fields: u16 LE m, i16 LE e.
 *
 * \param s: the scale, loaded from its field with odtScaleLoad; fails fast
 *           (PRINT_ERROR + exit(1)) when it is not canonical
 * \param f: Pointer of file to serialize to
 */
static void serializeScale(odtScale_t s, FILE *f);

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
