#ifndef ODT_ADD_LAYER_H
#define ODT_ADD_LAYER_H

#include <stdbool.h>
#include <stdint.h>

#include "ArithmeticType.h"
#include "Tensor.h"

typedef struct layer layer_t;
typedef struct layerExec layerExec_t;

/* Elementwise sum of two equally shaped wires; FLOAT32 only, no broadcasting. */
typedef struct addConfig {
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} addConfig_t;

/* in[0..2) have the shape addShape accepted; k is the op's arity (2). */
void addForward(layer_t *layer, tensor_t *const *in, uint8_t k, tensor_t *out,
                const layerExec_t *exec);

/* Exits unless both input shapes are identical (rank, dims, order). */
void addShape(layer_t *layer, const shape_t *const *in, uint8_t k, shape_t *out);

#endif
