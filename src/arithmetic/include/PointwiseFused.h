#ifndef POINTWISE_FUSED_H
#define POINTWISE_FUSED_H

#include <stddef.h>

#include "Tensor.h"

/*! Fused elementwise AdamW primitives whose contract is PyTorch's ROUNDING ORDER.
 * The TU is built with -ffp-contract=off so the separate roundings stay separate;
 * lerp's fmaf() is the one deliberate fusion.
 *
 * Operands are FLOAT32 with the same dims and the same storage layout; the loops
 * run over flat storage. Every read of index i precedes the write of index i, so
 * any aliasing among operands is safe. */

/*! a[i] = fmaf(weight, b[i]-a[i], a[i]), torch lerp_. Bit-identical to ATen only
 * for |weight| < 0.5 (AdamW: weight = 1-beta1); ATen switches formula above. */
void lerpFloat32TensorsInplace(tensor_t *a, tensor_t *b, float weight);

/*! a[i] += (s*b[i])*c[i], torch addcmul_: three left-associated roundings. */
void addcmulFloat32TensorsInplace(tensor_t *a, tensor_t *b, tensor_t *c, float s);

/*! a[i] += (s*b[i]) / (sqrtf(v[i])/dScale + eps), torch
 * addcdiv_(b, (v.sqrt()/dScale).add_(eps), value=s) with the same six roundings,
 * without materializing the denominator. torch's Linux AVX2 path fuses the final
 * multiply-add and may differ by <=1 ulp near a zero accumulator; the gold tests
 * pin this rounding order, not a torch capture.
 * Caller guarantees v[i] >= 0, dScale > 0, eps > 0. */
void addcdivDenomFloat32TensorsInplace(tensor_t *a, tensor_t *b, tensor_t *v, float dScale,
                                       float eps, float s);

size_t getLerpInstructionCounter(void);
size_t getAddcmulInstructionCounter(void);
size_t getAddcdivDenomInstructionCounter(void);

#endif // POINTWISE_FUSED_H
