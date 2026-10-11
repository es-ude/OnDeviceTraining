#ifndef BERNOULLI_H
#define BERNOULLI_H

#include "Tensor.h"

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngStream rngStream_t;

/*! Fills `mask` (a BOOL tensor) with independent Bernoulli draws: each bit is
 *  set (1) with probability `probTrue`, clear (0) with probability 1-probTrue.
 *  Uses the canonical RNG (`rngNextFloat`); seed via `rngSetSeed` for
 *  reproducibility. `mask` must be BOOL-quantized with its data buffer
 *  allocated. */
void bernoulliFillMask(tensor_t *mask, float probTrue, rngStream_t *rs);

/*! Signature of a Bernoulli mask-fill implementation, for swapping in
 *  optimized variants (SIMD, bit-parallel, hardware RNG). */
typedef void (*bernoulliFillMaskFn_t)(tensor_t *mask, float probTrue, rngStream_t *rs);

/*! The reference implementation (XorShift32 inverse-transform). Exported so it
 *  can be restored after swapping. */
void bernoulliFillMaskReference(tensor_t *mask, float probTrue, rngStream_t *rs);

/*! Install a sampler implementation. Passing NULL restores the reference.
 *  The slot is process-wide: install the sampler before training starts. An
 *  installed sampler must be a pure function of its `rs`, `probTrue` and the
 *  mask geometry, or a recomputed forward draws a different mask. */
void bernoulliSetFillMaskFn(bernoulliFillMaskFn_t fn);

/*! Returns the currently active sampler (for save/restore in tests). */
bernoulliFillMaskFn_t bernoulliGetFillMaskFn(void);

#endif // BERNOULLI_H
