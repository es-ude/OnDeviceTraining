#ifndef DESERIALIZEINTERNAL_H
#define DESERIALIZEINTERNAL_H

#include <stdio.h>

#include "ArithmeticType.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

/*! Deserializes shape of tensor from given file (u32 LE fields, #370).
 *  Fails fast if the file rank does not match the skeleton's rank — the
 *  skeleton's dimension arrays were sized by the build-time rank.
 *
 * \param shape: Pointer to shape to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeShape(shape_t *shape, FILE *f);

/*! Deserializes quantization of tensor from given file.
 *
 * \param q: Pointer to quantization to deserialize into
 * \param f: Pointer of file to deserialize from
 * \param numberOfElements: element count the config attaches to, forwarded to
 *  deserializeQConfig's SYM divisibility check; 0 when no live tensor backs
 *  this quantization_t at this call site (layer outputQ/propLossQ wire
 *  configs — group-quant PR2 carrier gate keeps these per-tensor anyway, so
 *  skipping the check here costs nothing)
 */
static void deserializeQuantization(quantization_t *q, FILE *f, size_t numberOfElements);

/*! Deserializes a declared compute representation from given file: u8 type +
 *  u8 roundingMode.
 *
 * \param arithmetic: Pointer to arithmetic_t to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeArithmetic(arithmetic_t *arithmetic, FILE *f);

/*! Deserializes kernel geometry (all kernel_t fields) from given file.
 *  Mirrors serializeKernel.
 *
 * \param kernel: Pointer to kernel_t to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeKernel(kernel_t *kernel, FILE *f);

/*! Deserializes quantization config of tensor from given file.
 *
 *  Group-quant PR2 (Task 5): the SYM arm tolerates a file numGroups that
 *  differs from q's current numGroups by REALLOCATING q's scales[] to the
 *  file's shape (freeReservedMemory the old array, reserveMemory the new
 *  one) rather than failing fast — the #316 no-silent-misparse discipline
 *  moves from "reject any mismatch" to "validate the resulting shape",
 *  enforced by validateSymQConfigShape (Quantization.h) against
 *  numberOfElements once the full record is parsed. The sentinel invariant
 *  (numGroups==1 <=> groupSize==0) is checked on the file's raw values
 *  first and is untouched by this relax.
 *
 * \param q: Pointer to quantization to deserialize into
 * \param f: Pointer of file to deserialize from
 * \param numberOfElements: element count q attaches to; 0 = skip-path/unknown
 *  (no divisibility validation) — see skipSerializedTensor, whose scratch
 *  qConfig is discarded rather than attached to a live tensor
 */
static void deserializeQConfig(quantization_t *q, FILE *f, size_t numberOfElements);

/*! Not implemented yet!
 */
static void deserializeSparsity();

/*! Parses past one full tensor record (shape + quantization header + packed
 *  payload + sparsity stub) without writing any output. Used by
 *  deserializeParameter to discard a grad record whose file hasGrad=1
 *  disagrees with a frozen skeleton (parameter->grad == NULL) — #380 PR3.
 *  Leaves the stream positioned right after the record, so a following
 *  sibling record parses in sync. Dims live in an 8-deep stack array and the
 *  payload is skipped via fseek, not read into a scratch buffer. The SYM
 *  scratch qConfig's scales[1] IS heap-allocated (unlike the other scratch
 *  qConfigs here), since a GROUPED SYM record (group-quant PR2, Task 5)
 *  makes deserializeQConfig reallocate it via the same free-then-reserveMemory
 *  path a live tensor's qConfig would take — a stack-backed array there would
 *  make that free() undefined behavior. Freed unconditionally after the call
 *  (see the .c file). Requires a seekable stream (fseek/ftell), matching the
 *  ODTR/PPCA deserialize precedent (#316 wave).
 *
 * \param f: Pointer of file to skip a tensor record from
 */
static void skipSerializedTensor(FILE *f);

/*! Deserializes layer from given file.
 *
 * \param layer: Pointer to layer to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeLayer(layer_t *layer, FILE *f);

#endif // DESERIALIZEINTERNAL_H
