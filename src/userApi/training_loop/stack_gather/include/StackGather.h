#ifndef STACK_GATHER_H
#define STACK_GATHER_H

#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "Tensor.h"

/* Names the first non-FLOAT32 field of a layer, or NULL when the layer is
 * FLOAT32 for the pass being gated (training: layerNonFloat32Field,
 * evaluation: layerForwardNonFloat32Field). */
typedef const char *(*nonFloat32FieldFn_t)(layer_t *layer);

/* Gather buffer of m rows x perSampleBytes: overflow-checked multiply; NULL
 * from reserveMemory fails fast -- never a copy through NULL. `caller` prefixes
 * every error message. */
uint8_t *stackGatherReserveBuffer(const char *caller, size_t m, size_t perSampleBytes,
                                  const char *what);

/* FLOAT32 gate over the whole model, evaluated once per stacked run before any
 * buffer is reserved. Fails fast naming the first layer whose fieldFn reports a
 * non-FLOAT32 field. */
void stackGatherRequireFloat32Model(const char *caller, layer_t **model, size_t modelSize, size_t m,
                                    nonFloat32FieldFn_t fieldFn);

/* A stacked sample must be FLOAT32, carry no sparsity and match `reference` in
 * rank, dimensions and order. */
void stackGatherRequireStackable(const char *caller, tensor_t *reference, tensor_t *t,
                                 const char *what, size_t sampleIndex, size_t m);

#endif /* STACK_GATHER_H */
