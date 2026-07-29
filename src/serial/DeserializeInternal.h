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
 *  Task-5 review fix (Critical): the file's numGroups is untrusted wire
 *  input read directly into an allocation size (fileNumGroups *
 *  sizeof(float)) BEFORE any of the above -- SERIAL_MAX_QCONFIG_GROUPS (see
 *  the .c file) rejects it outright before the realloc runs, and whenever
 *  numberOfElements != 0 it is additionally rejected if it already exceeds
 *  the element count (a config cannot have more groups than elements).
 *  SERIAL_MAX_QCONFIG_GROUPS alone protects the numberOfElements == 0 call
 *  sites, where the elements-bound guard cannot apply. The ASYM arm
 *  (group-quant PR4, Task 4) applies the identical discipline to its own
 *  numGroups, reallocating BOTH its scales[] and zeroPoints[] arrays.
 *
 * \param q: Pointer to quantization to deserialize into
 * \param f: Pointer of file to deserialize from
 * \param numberOfElements: element count q attaches to; 0 only at the layer
 *  outputQ/propLossQ wire-config call sites in deserializeLayer, where no
 *  live tensor backs q (group-quant PR2's carrier gate keeps those
 *  per-tensor anyway, so skipping the divisibility validate there costs
 *  nothing). Every other caller — including skipSerializedTensor's grad-skip
 *  path (Task-5 review fix: it now threads the real element count it just
 *  parsed off the wire, not a hardcoded 0) — passes its true count and gets
 *  the full validate.
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
 *  (see the .c file). Also threads the record's OWN element count (computed
 *  from the dims it just read, Task-5 review fix) into deserializeQConfig
 *  instead of a hardcoded 0, so a grouped record whose numGroups*groupSize
 *  does not divide its own element count fails fast on this path too, not
 *  just when a live tensor backs the config. Requires a seekable stream
 *  (fseek/ftell), matching the ODTR/PPCA deserialize precedent (#316 wave).
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
