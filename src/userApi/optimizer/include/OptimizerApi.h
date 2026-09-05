#ifndef OPTIMIZER_API_H
#define OPTIMIZER_API_H

#include "Optimizer.h"

/* No standalone `SGD` factory: sgdMCreateOptim() with momentumFactor == 0
 * IS plain SGD -- the step runs a single stateless update op per parameter
 * and the factory allocates no momentum-state buffers in this mode (#308). */

/* Scales every gradient field of every parameter tracked by the optimizer
 * by the given factor in-place.
 *
 * Iterates the same parameter list that optimizerZeroGrad uses
 * (optimizer->parameter[0..sizeStates-1]).
 *
 * Caller is responsible for not calling with factor == 1.0 (no-op);
 * factor must be positive and finite — non-positive or non-finite values
 * are logged via PRINT_ERROR (the closest existing tool; #151 will replace
 * this with a proper PRINT_WARN). For FLOAT32/SYM_INT32/SYM/ASYM grads that
 * warning is where it ends: those carriers REPRESENT NaN/inf (a float grad
 * element, a per-tensor scale) and propagate it, so the failure stays loud
 * downstream. A BFP-stored grad instead fails fast (exit(1)) inside
 * scaleBfpTensorInPlace: a (mantissa, shared exponent) grid has no
 * non-finite code, so there is nothing to propagate it into. */
void scaleOptimizerGradients(optimizer_t *optimizer, float factor);

/* #382: global-norm gradient clipping, `torch.nn.utils.clip_grad_norm_`
 * parity. Call-site contract: AFTER grad accumulation and mean-scaling
 * (computeMeanScale -> scaleOptimizerGradients) and BEFORE the optimizer
 * step -- clipping a not-yet-mean-scaled or already-stepped gradient set
 * clips the wrong quantity.
 *
 * Computes ONE joint L2 norm over every element of every parameter's grad
 * tracked by `optimizer` (as if every grad tensor were concatenated into a
 * single vector -- NOT a per-tensor norm), dtype-aware:
 *   - FLOAT32: sum of squares over elements.
 *   - SYM_INT32: scale^2 * sum(mantissa^2) per tensor (int32 mantissa reads
 *     widened to double before squaring -- NO int64, mirroring the SYM-kernel
 *     accumulator rule in spirit).
 *   - packed SYM/ASYM/BFP: unsupported in v1 -- fails fast (PRINT_ERROR + exit(1)).
 *     The O(1) scale-fold this function reuses from scaleOptimizerGradients
 *     only helps APPLYING an already-computed clip coefficient; computing the
 *     norm itself needs unpacked element values, which packed storage doesn't
 *     expose without a full unpack. Follow-up, not implemented here.
 * The running sum of squares accumulates in double across every tensor
 * (joint, not per-tensor), then one sqrt casts to float32 once.
 *
 * Torch parity: `clipCoef = maxNorm / (totalNorm + 1e-6f)`; gradients are
 * scaled (via scaleOptimizerGradients, which O(1)-folds the factor into the
 * per-tensor scale for quantized grads) only when `clipCoef < 1.0f` --
 * maxNorm >= totalNorm is a no-op, gradients are left byte-untouched (never
 * multiplied by a clamped 1.0). Returns totalNorm (PRE-clip, torch
 * convention).
 *
 * Validation: `maxNorm` must be positive and finite, else PRINT_ERROR +
 * exit(1) (hard fail -- unlike scaleOptimizerGradients's own factor check,
 * which only warns; an invalid maxNorm here means the caller's clipping
 * config itself is broken, not a transient scale value). */
float optimizerClipGradNorm(optimizer_t *optimizer, float maxNorm);

/* Fills the caller-allocated `slots` array (sized via calcTotalNumberOfStates)
 * with every trainable parameter_t* in `model`, in model order. Per-layer-type
 * switch: LINEAR/CONV1D/CONV1D_TRANSPOSED contribute weights (+ bias, if
 * present -- BIAS_FALSE layers carry none); LAYERNORM/GROUPNORM contribute
 * gamma + beta; layers with no trainable parameters are skipped; an unknown
 * layer type fails fast (PRINT_ERROR + exit(1)). Frozen layers (#380) are
 * skipped entirely -- none of their parameters land in `slots`.
 *
 * Extracted from SgdApi.c (#328 groundwork) so non-SGD factories (e.g. PR C's
 * adamWCreateOptim) can reuse the same collection logic. */
void collectTrainableParameters(layer_t **model, size_t sizeModel, parameter_t **slots);

/* #380: true iff at least one layer in `model` is frozen (layerIsFrozen).
 * Distinguishes "every parameter-bearing layer got frozen" (a mis-built,
 * nothing-to-train model -- the factories fail-fast on this) from "the model
 * simply has no parameter-bearing layer types at all" (e.g. a lone Dropout/
 * pooling layer -- a pre-existing, deliberately-supported zero-state
 * configuration, #279/#308 sibling contracts): sizeStates == 0 alone can't
 * tell the two apart, since both land on zero collected parameters. */
bool modelHasFrozenLayer(layer_t **model, size_t sizeModel);

/* #261, PR3: validates that every parameter's grad storage tracked by `optim`
 * is one of the accepted dtypes -- FLOAT32 (default), SYM_INT32 (explicit
 * low-level knob), packed SYM/ASYM, or per-tensor BFP (explicit
 * grad-storage knob, memory-constrained targets; BFP epic PR3 Task 6 --
 * grouped BFP grads are unreachable here, gradInit's own carrier gate
 * rejects them first, #300 axis). INT32/BOOL grad storage remains
 * unimplemented: fails fast (PRINT_ERROR naming `factoryName` + exit(1))
 * rather than silently misreading bytes in an unsupported layout. Frozen
 * layers (#380) are skipped before collection, so they never reach
 * `optim->parameter[]`; a NULL grad in a COLLECTED slot is therefore a
 * mis-built model, not a frozen layer, and fails fast here rather than
 * crashing mid-training.
 *
 * Extracted from SgdApi.c (#328 groundwork) so non-SGD factories reuse the
 * same guard; `factoryName` names the caller in the error message. */
void validateOptimizerGradStorage(optimizer_t *optim, const char *factoryName);

/* Frees one parameter's momentum/state buffers: every tensor_t in
 * `state->stateBuffers[0..statesPerParameter-1]` (via freeTensor), then the
 * buffers array and the states_t shell itself. Generic over
 * statesPerParameter -- not SGD-specific, so any impl that allocates 0..N
 * state tensors per parameter can reuse it as-is.
 *
 * Moved from SgdApi.c (#328 groundwork); only called internally by freeOptim. */
void freeState(states_t *state);

/* Frees an entire optimizer_t and everything it owns: every registered
 * parameter_t* (via freeParameter -- this also frees each parameter's grad
 * tensor, per the ownership contract established at parameter registration),
 * every per-parameter states_t* (via freeState, skipped when
 * momentumFactor == 0 left optim->states NULL, #308), then
 * optim->parameter[], optim->states[], optim->impl's payload, optim->impl,
 * and finally optim itself.
 *
 * Moved from SgdApi.c and renamed to drop the SGD-specific suffix (#328
 * groundwork): the body never touched anything SGD-specific -- it walks
 * optim->impl through the optimImpl_t union, so it generalizes to any impl
 * (e.g. PR C's AdamW) without modification. */
void freeOptim(optimizer_t *optim);

#endif // OPTIMIZER_API_H
