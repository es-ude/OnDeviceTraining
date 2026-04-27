#ifndef TENSORAPIINTERNAL_H
#define TENSORAPIINTERNAL_H

/*! Initializes tensor with given int32 quantization.
 *
 * Used for tensorInitInt32, tensorInit and tensorInitWithDistribution.
 *
 * \param data: Pointer to int32 data array
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Pointer to quantization
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer initialized tensor
 */
static tensor_t *initTensorWithQInt32(int32_t *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, sparsity_t *sparsity);
/*! Initializes tensor with given float quantization.
 *
 * Used for tensorInitFloat, tensorInit and tensorInitWithDistribution.
 *
 * \param data: Pointer to float data array
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Pointer to quantization
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer initialized tensor
 */
static tensor_t *initTensorWithQFloat(float *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, sparsity_t *sparsity);
/*! Initializes tensor with given symInt32 quantization.
 *
 * Used for tensorInitSymInt32, tensorInit and tensorInitWithDistribution.
 *
 * \param data: Pointer to float data array
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Pointer to quantization
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer initialized tensor
 */
static tensor_t *initTensorWithQSymInt32(float *data, size_t *dims, size_t numberOfDims,
                                         quantization_t *quantization, sparsity_t *sparsity);

/*! Initializes tensor with given asym quantization.
 *
 * Used for tensorInitAsym, tensorInit and tensorInitWithDistribution.
 *
 * \param data: Pointer to float data array
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Pointer to quantization
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer initialized tensor
 */
static tensor_t *initTensorWithQAsym(float *data, size_t *dims, size_t numberOfDims,
                                     quantization_t *quantization, sparsity_t *sparsity);

/*! Frees only the pointer of a tensor.
 *
 * Only used in freeTensor.
 *
 * \param tensor: Pointer to tensor
 */
static void freeTensorPointer(tensor_t *tensor);

#endif // TENSORAPIINTERNAL_H
