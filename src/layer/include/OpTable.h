#ifndef ODT_OP_TABLE_H
#define ODT_OP_TABLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "Tensor.h"

/* One row per layerType_t member: the last member + 1. */
#define OP_TABLE_ROWS ((size_t)BATCHNORM1D + 1u)

/* in[0..k) are the input shapes; out arrives with caller-owned dimension
 * storage and receives the output shape. */
typedef void (*opShapeFn_t)(layer_t *layer, const shape_t *const *in, uint8_t k, shape_t *out);

/* When a node's backward needs its forward inputs kept alive (or recomputed). */
typedef enum opReads { OP_READS_NEVER = 0, OP_READS_ALWAYS, OP_READS_IF_TRAINABLE } opReads_t;

/* in[0..k) are the node's input wires; out receives its output. */
typedef void (*opForwardFn_t)(layer_t *layer, tensor_t *const *in, uint8_t k, tensor_t *out,
                              const layerExec_t *exec);
/* dx[i] receives the gradient for in[i] in mode dxMode[i]; dx[i] == NULL asks for no dx on
 * that slot (a grads-only call, valid only for param layers -- see backwardFn_t). */
typedef void (*opBackwardFn_t)(layer_t *layer, tensor_t *const *in, uint8_t k, tensor_t *dy,
                               tensor_t *const *dx, const uint8_t *dxMode, const layerExec_t *exec);

typedef struct opFunctions {
    opForwardFn_t forward;
    opBackwardFn_t backward;
    opShapeFn_t shape;
    uint8_t arity;       /* number of input buffers a node of this op reads */
    uint8_t readsInputs; /* opReads_t: whether backward reads the forward inputs */
    bool readsOutput;    /* backward reads the node's own forward output */
    bool accumulatesDx;  /* backward can add into an open dx wire (OP_MODE_ACC) */
} opFunctions_t;

/* Indexed by layerType_t. const so it stays flash-resident; the rows dispatch
 * through the mutable layerFunctions[] at call time. */
extern const opFunctions_t opFunctions[];

#endif
