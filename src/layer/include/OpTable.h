#ifndef ODT_OP_TABLE_H
#define ODT_OP_TABLE_H

#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "Tensor.h"

/* One row per layerType_t member: the last member + 1. */
#define OP_TABLE_ROWS ((size_t)BATCHNORM1D + 1u)

/* in[0..k) are the input shapes; out arrives with caller-owned dimension
 * storage and receives the output shape. */
typedef void (*opShapeFn_t)(layer_t *layer, const shape_t *const *in, uint8_t k, shape_t *out);

typedef struct opFunctions {
    opShapeFn_t shape;
    uint8_t arity; /* number of input buffers a node of this op reads */
} opFunctions_t;

/* Indexed by layerType_t. const so it stays flash-resident; the rows dispatch
 * through the mutable layerFunctions[] at call time. */
extern const opFunctions_t opFunctions[];

#endif
