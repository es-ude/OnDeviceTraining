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
 * per-tensor; the template stays caller-owned and may be freed as soon as the create
 * call returns. Grad storage must be
 * FLOAT32/SYM_INT32/SYM/ASYM/BFP.
 * Free with freeOptim(). */
optimizer_t *adamWCreateOptim(float learningRate, double beta1, double beta2, double eps,
                              double weightDecay, const model_t *m, quantization_t *momentQuant,
                              arithmetic_t updateMath);

/*! Adds a parameter group: the parameters `select` picks (walked in optimizer
 * order, see ParameterWalk.h) leave group 0 and step with learning rate
 * base * lrMultiplier and this decoupled weightDecay; betas, eps and the bias
 * correction stay optimizer-wide. Returns the new group's index. Call after
 * adamWCreateOptim and before training; m must be the model the optimizer was
 * built from. Fails fast (exit 1) on a wrong model, a parameter that is
 * already grouped, an empty selection, a 257th group, or a non-finite or
 * non-positive lrMultiplier. */
size_t adamWAddGroup(optimizer_t *optim, const model_t *m, paramSelectFn_t select, void *ctx,
                     float lrMultiplier, double weightDecay);

#endif // ADAM_W_API_H
