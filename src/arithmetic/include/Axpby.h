#ifndef ODT_AXPBY_H
#define ODT_AXPBY_H

#include "Tensor.h"

/* out = a*x + b*y, all FLOAT32 with the same logical dims.
 * x and y may be permuted views (reads follow orderOfDimensions); out is written
 * flat row-major, so it must have identity order. out may alias x or y only when
 * that operand has identity order. */
void axpbyFloat32Tensors(float a, tensor_t *x, float b, tensor_t *y, tensor_t *out);

#endif // ODT_AXPBY_H
