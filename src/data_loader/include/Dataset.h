#ifndef DATASET_H
#define DATASET_H

#include "Tensor.h"

typedef enum { FLOAT_32, INT_32 } dtype_t;

typedef struct sample {
    tensor_t *item;
    tensor_t *label;
} sample_t;

typedef struct batch {
    sample_t **samples;
    size_t size;
} batch_t;

typedef struct tensorArray {
    tensor_t **array;
    size_t size;
} tensorArray_t;

typedef struct dataset {
    tensorArray_t *items;
    tensorArray_t *labels;
} dataset_t;

#endif // DATASET_H
