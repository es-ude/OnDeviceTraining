#ifndef ADAM_W_API_H
#define ADAM_W_API_H

#include "AdamW.h"
#include "Optimizer.h"
#include "Quantization.h"

/*! torch.optim.AdamW-parity factory. Every trainable parameter gets
 * TWO moment buffers (stateBuffers[0] = m, [1] = v) at the parameter's
 * shape, each carrying its own deep clone (getQLike) of the ONE
 * `momentQuant` template -- FLOAT32 via quantizationInitFloat() is the
 * supported default; quantized moment storage is a memory knob routed
 * through the executeOp funnel (no bit-parity claim; the write-back dead-zone
 * semantics apply to moments exactly as to params). Moment storage must be
 * per-tensor; the template stays caller-owned. Grad storage must be
 * FLOAT32/SYM_INT32/SYM/ASYM/BFP.
 * Free with freeOptim(). */
optimizer_t *adamWCreateOptim(float learningRate, double beta1, double beta2, double eps,
                              double weightDecay, const model_t *m, quantization_t *momentQuant,
                              arithmetic_t updateMath);

#endif // ADAM_W_API_H
