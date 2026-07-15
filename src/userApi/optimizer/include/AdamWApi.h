#ifndef ADAM_W_API_H
#define ADAM_W_API_H

#include "AdamW.h"
#include "Optimizer.h"
#include "Quantization.h"

/*! torch.optim.AdamW-parity factory (#328). Every trainable parameter gets
 * TWO moment buffers (stateBuffers[0] = m, [1] = v) at the parameter's
 * shape, each carrying its own deep clone (getQLike) of the ONE
 * `momentQuant` template -- FLOAT32 via quantizationInitFloat() is the
 * supported default; quantized moment storage is a memory knob routed
 * through the executeOp funnel (no bit-parity claim, #279 dead-zone
 * semantics apply to moments exactly as to params). The template itself
 * stays caller-owned. Grad-storage dtypes are validated like every factory
 * (FLOAT32/SYM_INT32/SYM/ASYM; INT32/BOOL fail fast, #261).
 * Free with freeOptim(). */
optimizer_t *adamWCreateOptim(float learningRate, double beta1, double beta2, double eps,
                              double weightDecay, layer_t **model, size_t sizeModel,
                              quantization_t *momentQuant, arithmetic_t updateMath);

#endif // ADAM_W_API_H
