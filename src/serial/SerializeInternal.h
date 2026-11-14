#ifndef SERIALINTERNAL_H
#define SERIALINTERNAL_H

#include <stdio.h>

#include "ArithmeticType.h"
#include "DyadicScale.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

/* Record writers; the layout is documented at the top of Serialize.c. */
static void serializeShape(shape_t *shape, FILE *f);
static void serializeQuantization(quantization_t *q, FILE *f);
static void serializeArithmetic(arithmetic_t *arithmetic, FILE *f);
static void serializeKernel(kernel_t *kernel, FILE *f);
/* A scale as its v9 dyadic fields (u16 LE m, i16 LE e); a non-canonical scale fails fast. */
static void serializeScale(odtScale_t s, FILE *f);
static void serializeQConfig(quantization_t *q, FILE *f);
static void serializeLayer(layer_t *layer, FILE *f);

#endif // SERIALINTERNAL_H
