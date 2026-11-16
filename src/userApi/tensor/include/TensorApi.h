#ifndef TENSOR_H
#define TENSOR_H

#include "Distributions.h"
#include "Tensor.h"

/*! Initializes a tensor that owns its data buffer.
 *
 * The tensor takes ownership of `shape` and `quantization`; it only borrows
 * `sparsity`, which freeTensor never frees. The data buffer is allocated by
 * initTensor itself (size derived from shape and quantization) and is
 * zero-initialized.
 *
 * After this call, freeTensor(t) is unconditionally safe and frees data, shape
 * and quantization.
 *
 * \param shape: Caller-allocated shape; ownership transferred to the tensor.
 * \param quantization: Caller-allocated quantization; ownership transferred.
 * \param sparsity: Optional borrowed marker (NULL means no sparsity); the caller
 *                  keeps it alive as long as the tensor.
 *
 * \returns Pointer to an initialized tensor with own-allocated zero data.
 */
tensor_t *initTensor(shape_t *shape, quantization_t *quantization, sparsity_t *sparsity);

/*! Fills a FLOAT32 tensor with values drawn from `distribution`; the
 * union member matching `distribution->type` supplies the parameters.
 *
 * \param rs: The stream every random kind (UNIFORM, NORMAL and the XAVIER_* and
 *            KAIMING_* kinds) draws from, element by element; NULL stops the program
 *            for those kinds on a non-empty tensor. ZEROS and ONES draw nothing and
 *            take NULL.
 */
void initDistribution(tensor_t *tensor, const distribution_t *distribution, rngStream_t *rs);

/*! Copies `count` floats from a caller-owned source into a tensor's data
 * buffer. Caller retains ownership of `source`. If the tensor is non-FLOAT32
 * quantized, values are converted via the tensor's quantization.
 *
 * \param tensor: Pre-initialized tensor.
 * \param source: Caller-owned float array (any storage class).
 * \param count: Number of floats; must equal the tensor's element count.
 */
void tensorFillFromFloatBuffer(tensor_t *tensor, const float *source, size_t count);

/*! Copies `count` unpacked bools from a caller-owned source into a
 *  bit-packed BOOL tensor's data buffer. Caller retains ownership
 *  of `source`. Tensor must be BOOL-quantized and have its data buffer
 *  already allocated (via initTensor).
 *
 *  \param tensor: Pre-initialized BOOL tensor.
 *  \param source: Caller-owned bool array.
 *  \param count: Number of bools; must equal the tensor's element count.
 */
void tensorFillFromBoolBuffer(tensor_t *tensor, const bool *source, size_t count);

/*! Clones param's shape and gradQ's storage config; gradQ must be per-tensor.
 *  Caller keeps ownership of gradQ. */
tensor_t *gradInit(tensor_t *param, quantization_t *gradQ, sparsity_t *sparsity);
/*! INT32 grad shaped like param. */
tensor_t *gradInitInt32(tensor_t *param, sparsity_t *sparsity);
/*! FLOAT32 grad shaped like param. */
tensor_t *gradInitFloat(tensor_t *param, sparsity_t *sparsity);
/*! SYM_INT32 grad shaped like param, at the fixed grad-accumulation width
 *  ODT_SYM_GRAD_QMAXBITS (16 — an accumulate-soundness ceiling, not a memory
 *  saving: SYM_INT32 stays 4 B/element). Test/research helper — production
 *  grad storage defaults to FLOAT32 via the weightGradStorage/biasGradStorage
 *  knob path (gradInit). */
tensor_t *gradInitSymInt32(tensor_t *param, roundingMode_t roundingMode, sparsity_t *sparsity);
/*! Per-tensor ASYM grad shaped like param. */
tensor_t *gradInitAsym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode,
                       sparsity_t *sparsity);
/*! Per-tensor packed SYM grad shaped like param, for a layer whose grad-storage
 *  knob names SYM. */
tensor_t *gradInitSym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode,
                      sparsity_t *sparsity);

/*! Takes ownership of param and grad (grad may be NULL); freeParameter frees
 *  both. */
parameter_t *parameterInit(tensor_t *param, tensor_t *grad);

/*! Fresh clone of a storage config, for a new grad/state tensor. Per-tensor
 *  SYM/ASYM/BFP clones keep widths and rounding but reset the grid (scale 1,
 *  zero point 0, exponent bias); grouped clones keep the geometry and copy the
 *  grid values. Use deepCopyQuantization for a full value copy. BOOL exits. */
quantization_t *getQLike(quantization_t *quantization);
/*! Fresh tensor with a copy of tensor's shape, a getQLike clone of its
 *  quantization and zeroed data; the values are not copied. The clone carries
 *  no sparsity marker (a marker is borrowed, see initTensor). */
tensor_t *getTensorLike(tensor_t *tensor);
/*! Heap-allocates a shape with a copy of `dims` and identity order. Ownership passes to the
 *  tensor built on it: freeTensor frees the shape. */
shape_t *buildOwnedShape(const size_t *dims, size_t numberOfDims);
/*! buildOwnedShape of `shape`'s physical dims: a transposed view comes back dense, in
 *  storage order. */
shape_t *getShapeLike(shape_t *shape);
/*! Zeroed data buffer for numberOfValues elements of quantization's dtype
 *  (freed with freeReservedMemory). BOOL exits. */
uint8_t *getDataLike(quantization_t *quantization, size_t numberOfValues);

/*! Frees only the data buffer; the tensor keeps its shape and quantization. */
void freeData(tensor_t *tensor);
void freeShape(shape_t *shape);
/*! Frees the qConfig, its owned per-group arrays and the wrapper. */
void freeQuantization(quantization_t *quantization);

/*! Requantizes `t`'s data + quantization to `targetQ`'s dtype/shape-of-
 * quantization IN PLACE: builds a fresh buffer sized for t's own element
 * count, dynamically quantizes via convertTensor, then swaps t's data +
 * quantization pointers and frees the old ones. t's shape and sparsity are
 * untouched. `targetQ` is read-only (cloned via getQLike); caller retains
 * ownership of it. A grouped target must describe exactly t's element count.
 *
 * \param t: Tensor to requantize in place; must already own heap-allocated
 *           data + quantization (e.g. via initTensor).
 * \param targetQ: Quantization template to requantize into (cloned, not
 *                 taken over).
 */
void requantizeTensorInPlace(tensor_t *t, quantization_t *targetQ);

/*! Frees data, shape, quantization and the tensor itself. */
void freeTensor(tensor_t *tensor);
/*! Frees param, the grad when non-NULL, and the parameter itself. */
void freeParameter(parameter_t *parameter);

#endif // TENSOR_H
