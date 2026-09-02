#ifndef IM2COL_H
#define IM2COL_H

#include "Tensor.h"
#include "SlidingWindow1d.h"

void im2col1dFloat(tensor_t const *input, windowGeometry1d_t const *geometry, tensor_t *output);

#endif //IM2COL_H
