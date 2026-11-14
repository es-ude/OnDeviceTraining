#ifndef SGD_API_H
#define SGD_API_H

#include "Sgd.h"

/*! Builds an SGD(+momentum) optimizer over a model's trainable parameters.
 *
 * Each parameter's momentum state buffer is allocated at the PARAMETER'S
 * SHAPE but with its OWN quantization config (`momentumQuant`, deep-cloned
 * per state via getQLike) -- decoupled from the parameter's own dtype.
 * Pass quantizationInitFloat() for the conventional FLOAT32 accumulator
 * (the sensible default for every existing caller); a caller wanting e.g. a
 * SYM momentum accumulator may pass a SYM quantization instead, subject to
 * getQLike's supported-type set (BOOL is not supported for state clones).
 *
 * \param momentumQuant: Quantization template for momentum state buffers
 *        (caller-owned; not consumed -- cloned once per state via getQLike).
 *        Must be per-tensor; ignored when momentumFactor == 0.
 * \param updateMath: arithmetic_t the update ops run in (by-value,
 *        mirroring the layer-side per-op knobs). Pass
 *        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY}
 *        for the established behavior. Only ARITH_FLOAT32 is implemented;
 *        any other type fails fast at creation until the integer-update
 *        numerics design lands. Rounding ownership: updateMath.roundingMode
 *        is superseded at step time -- every write-back op runs with the
 *        optimizer's writeBackRounding as its operation-owned rounding (seeded
 *        SR factory default, optimizerSetWriteBackRounding to opt out).
 *
 * \note momentumFactor is creation-fixed: it determines whether momentum-state
 *       buffers are allocated (momentumFactor == 0 → none), so callers
 *       must not mutate sgd_t.momentumFactor from 0 to nonzero after creation.
 */
optimizer_t *sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             const model_t *m, quantization_t *momentumQuant,
                             arithmetic_t updateMath);

#endif // SGD_API_H
