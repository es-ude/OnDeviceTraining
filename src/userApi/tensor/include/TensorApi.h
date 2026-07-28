#ifndef TENSOR_H
#define TENSOR_H

#include "Distributions.h"
#include "Tensor.h"

/*! Initializes a tensor that owns its data buffer.
 *
 * The tensor takes ownership of `shape`, `quantization`, and `sparsity` (if non-NULL).
 * The data buffer is allocated by initTensor itself (size derived from shape and
 * quantization) and is zero-initialized.
 *
 * After this call, freeTensor(t) is unconditionally safe and frees all four
 * ownership domains (data, shape, quantization, sparsity).
 *
 * \param shape: Caller-allocated shape; ownership transferred to the tensor.
 * \param quantization: Caller-allocated quantization; ownership transferred.
 * \param sparsity: Optional; ownership transferred (NULL means no sparsity).
 *
 * \returns Pointer to an initialized tensor with own-allocated zero data.
 */
tensor_t *initTensor(shape_t *shape, quantization_t *quantization, sparsity_t *sparsity);

/*! Populates an already-initialized tensor's data buffer with values drawn
 * from the given distribution. Tensor must be FLOAT32-quantized in this
 * iteration; non-FLOAT32 support is a follow-up.
 *
 * The tensor must have its data buffer already allocated (via initTensor).
 * The distribution value is read by `distribution->type`; the matching
 * union member supplies the kind-specific parameters.
 *
 * \param tensor: Pre-initialized FLOAT32 tensor.
 * \param distribution: Recipe for value generation. Read-only.
 */
void initDistribution(tensor_t *tensor, const distribution_t *distribution);

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

/*! Initializes a gradient tensor whose quantization is cloned from `gradQ`
 *  (deep copy via getQLike, so the grad tensor owns its own quantization_t)
 *  and whose shape is cloned from `param->shape`. This is the config-respecting
 *  grad-init: a layer's grad takes the dtype its backwardMath config declares.
 *
 * \param param: Pointer to param tensor (supplies the shape to clone)
 * \param gradQ: Quantization template to clone for the grad tensor
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInit(tensor_t *param, quantization_t *gradQ, sparsity_t *sparsity);
/*! Initializes int32 gradient tensor to match given param tensor.
 *
 * \param param: Pointer to param tensor
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitInt32(tensor_t *param, sparsity_t *sparsity);
/*! Initializes float gradient tensor to match given param tensor.
 *
 * \param param: Pointer to param tensor
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitFloat(tensor_t *param, sparsity_t *sparsity);
/*! Initializes a SYM_INT32 gradient tensor to match the given param tensor,
 *  at the fixed grad-accumulation width ODT_SYM_GRAD_QMAXBITS (16 — an
 *  accumulate-soundness ceiling, not a memory saving: SYM_INT32 stays 4
 *  B/element, #261). Test/research helper — production grad storage defaults
 *  to FLOAT32 via the weightGradStorage/biasGradStorage knob path (gradInit).
 *
 * \param param: Pointer to param tensor
 * \param roundingMode: Rounding mode for the grad's qConfig
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitSymInt32(tensor_t *param, roundingMode_t roundingMode, sparsity_t *sparsity);
/*! Initializes asym gradient tensor to match given param tensor.
 *
 * \param param: Pointer to param tensor
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitAsym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode,
                       sparsity_t *sparsity);
/*! Initializes SYM (packed sub-byte) gradient tensor to match given param
 *  tensor's shape. Use when a layer's grad-storage knob names SYM explicitly
 *  for memory-constrained packed grad storage (#269, PR3); direct
 *  quantizationInitSym, no getQLike detour.
 *
 * \param param: Pointer to param tensor (supplies the shape to clone)
 * \param qBits: Sub-byte width of the packed SYM grad
 * \param roundingMode: Rounding mode for the SYM quantization config
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitSym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode,
                      sparsity_t *sparsity);

/*! Initializes parameter with given param and graident tensor.
 *
 * \param param: Pointer to param tensor
 * \param grad: Pointer to gradient tensor
 *
 * \returns Pointer to initialized parameter
 */
parameter_t *parameterInit(tensor_t *param, tensor_t *grad);

/*! Gets quantization that matches a given quantization.
 *
 * \param quantization: Pointer to quantization
 *
 * \returns Pointer to quantization with matching type and config
 */
quantization_t *getQLike(quantization_t *quantization);
/*! Gets tensor that matches a given tensor.
 *
 * \param tensor: Pointer to tensor
 *
 * \returns Pointer to tensor with matching shape, sparsity and quantization
 */
tensor_t *getTensorLike(tensor_t *tensor);
/*! Gets shape that matches a given shape.
 *
 * \param shape: Pointer to shape
 *
 * \returns Pointer to shape with matching numberOfDims, dims, and orderOfDims
 */
shape_t *getShapeLike(shape_t *shape);
/*! Gets data array by quantization and number of values.
 *
 * \param quantization: Pointer to quantization
 * \param numberOfValues: Number of values
 *
 * \returns Pointer to data array with matching size
 */
uint8_t *getDataLike(quantization_t *quantization, size_t numberOfValues);
/*! Gets sparsity that matches a given sparsity.
 *
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer to sparsity
 */
sparsity_t *getSparsityLike(sparsity_t *sparsity);

// IMPORTANT: these are needed for trainingAPI.c to free gradTensors without freeing the actual
// Pointer
/*! Frees data of tensor.
 *
 * \param tensor: Pointer to tensor
 */
void freeData(tensor_t *tensor);
/*! Frees shape of tensor.
 *
 * \param shape: Pointer to shape
 */
void freeShape(shape_t *shape);
/*! Frees quantization of tensor.
 *
 * \param quantization: Pointer to quantization
 */
void freeQuantization(quantization_t *quantization);

/*! Requantizes `t`'s data + quantization to `targetQ`'s dtype/shape-of-
 * quantization IN PLACE: builds a fresh buffer sized for t's own element
 * count, dynamically quantizes via convertTensor (the same path
 * tensorFillFromFloatBuffer / layerLoadWeights use for non-FLOAT32 targets),
 * then swaps t's data + quantization pointers and frees the old ones. t's
 * shape and sparsity are untouched. `targetQ` is read-only (its dtype/config
 * is cloned via getQLike); caller retains ownership of `targetQ` and must
 * free it separately if heap-allocated.
 *
 * \param t: Tensor to requantize in place; must already own heap-allocated
 *           data + quantization (e.g. via initTensor).
 * \param targetQ: Quantization template to requantize into (cloned, not
 *                 taken over).
 */
void requantizeTensorInPlace(tensor_t *t, quantization_t *targetQ);

/*! Frees tensor.
 *
 * \param tensor: Pointer to tensor
 */
void freeTensor(tensor_t *tensor);
/*! Frees parameter.
 *
 * \param parameter: Pointer to parameter
 */
void freeParameter(parameter_t *parameter);

#endif // TENSOR_H
