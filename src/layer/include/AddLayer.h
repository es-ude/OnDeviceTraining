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

/* target += src, exactly. The one way a merge adds into an open wire, so ADD's dx accumulate and
 * a SUM step share its numerics. The caller guarantees both FLOAT32 and equally sized: the remat
 * wire table refuses a non-FLOAT32 merge or accumulate wire at init. */
void addAccumulate(tensor_t *src, tensor_t *target);

/* dx[i] = dy (OP_MODE_WRITE) or dx[i] += dy (OP_MODE_ACC), slots in input order, so ADD(y, y)
 * routed to one buffer as WRITE then ACC yields 2 dy. dx[i] == NULL skips the slot. Reads
 * neither in nor the forward output. */
void addBackward(layer_t *layer, tensor_t *const *in, uint8_t k, tensor_t *dy, tensor_t *const *dx,
                 const uint8_t *dxMode, const layerExec_t *exec);

/* Exits unless both input shapes are identical (rank, dims, order). */
void addShape(layer_t *layer, const shape_t *const *in, uint8_t k, shape_t *out);

#endif
