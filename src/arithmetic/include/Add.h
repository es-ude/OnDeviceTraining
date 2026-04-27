#ifndef ENV5_RUNTIME_ADD_H
#define ENV5_RUNTIME_ADD_H

#include "Tensor.h"
#include <stddef.h>
#include <stdint.h>

/*!
 * Calculates the sum of two floats.
 *
 * \param a: First float
 * \param b: Second float
 * \return Sum of a and b
 */
float addFloat32s(float a, float b);
/*!
 * Calculates sum of two int32s.
 *
 * \param a: First float
 * \param b: Second float
 * \return Sum of a and b
 */
int32_t addInt32s(int32_t a, int32_t b);

/*!
 * Calculates the sum of two int32 tensors.
 *
 * \param a: First tensor
 * \param b: Second tensor
 * \param outputTensor: Tensor for results
 */
void addInt32Tensors(tensor_t *a, tensor_t *b, tensor_t *outputTensor);
/*!
 * Calculates the sum of two int32 tensors inplace.
 *
 * The results will be stored in the first tensor (a).
 *
 * \param a: First tensor
 * \param b: Second tensor
 */
void addInt32TensorsInplace(tensor_t *a, tensor_t *b);

/*!
 * Calculates the sum of int32 tensor and int32 element.
 *
 * \param a: tensor
 * \param b: element
 * \param outputTensor: Tensor for results
 */
void addInt32ElementWithInt32Tensor(tensor_t *a, int32_t b, tensor_t *outputTensor);
/*!
 * Calculates the sum of int32 tensor and int32 element inplace.
 *
 * The result is stored in the tensor (a).
 *
 * \param a: tensor
 * \param b: element
 */
void addInt32ElementWithInt32TensorInplace(tensor_t *a, int32_t b);

/*!
 * Calculates the sum of two float tensors.
 *
 * \param a: First tensor
 * \param b: Second tensor
 * \param outputTensor: Tensor for results
 */
void addFloat32Tensors(tensor_t *a, tensor_t *b, tensor_t *outputTensor);
/*!
 * Calculates the sum of two float tensors inplace.
 *
 * The results will be stored in the first tensor (a).
 *
 * \param a: First tensor
 * \param b: Second tensor
 */
void addFloat32TensorsInplace(tensor_t *a, tensor_t *b);

/*!
 * Calculates the sum of float tensor and float element.
 *
 * \param a: tensor
 * \param b: element
 * \param outputTensor: Tensor for results
 */
void addFloat32ElementWithFloat32Tensor(tensor_t *a, float b, tensor_t *outputTensor);
/*!
 * Calculates the sum of float tensor and float element inplace.
 *
 * The result is stored in the tensor (a).
 *
 * \param a: tensor
 * \param b: element
 */
void addFloat32ElementWithFloat32TensorInplace(tensor_t *a, float b);

/*!
 * Calculates the sum of symInt32 tensor and int32 tensor inplace.
 *
 * The result is stored in the symInt32 tensor.
 *
 * \param symInt32Tensor: symInt32Tensor
 * \param int32Tensor: int32Tensor
 */
void addInt32TensorToSymInt32TensorInplace(tensor_t *symInt32Tensor, tensor_t *int32Tensor);
/*!
 * Calculates the sum of symInt32 tensor and float tensor inplace.
 *
 * The result is stored in the symInt32 tensor.
 *
 * \param symInt32Tensor: symInt32Tensor
 * \param float32Tensor: float32Tensor
 */
void addFloat32TensorToSymInt32TensorInplace(tensor_t *symInt32Tensor, tensor_t *float32Tensor);
/*!
 * Calculates the sum of two symInt32 tensors.
 *
 * \param aTensor: First tensor
 * \param bTensor: Second tensor
 * \param outputTensor: Tensor for results
 */
void addSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);
/*!
 * Calculates the sum of two symInt32 tensors inplace.
 *
 * The results will be stored in the first tensor (aTensor).
 *
 * \param aTensor: First tensor
 * \param bTensor: Second tensor
 */
void addSymInt32TensorsInplace(tensor_t *aTensor, tensor_t *bTensor);

/*!
 * Gets the current value of the add instruction counter.
 *
 * \returns Current value of add instruction counter
 */
size_t getAddInstructionCounter();

#endif // ENV5_RUNTIME_ADD_H
