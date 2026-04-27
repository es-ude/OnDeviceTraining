#ifndef QUANTIZATIONAPI_H
#define QUANTIZATIONAPI_H

#include "Quantization.h"
#include "Rounding.h"

/*! Initializes float quantization.
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitFloat();

/*! Initializes int32 quantization.
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitInt32();

/*! Initializes symInt32 quantization.
 *
 * \param roundingMode: Rounding mode to be used
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitSymInt32(roundingMode_t roundingMode);

/*! Initializes asym quantization.
 *
 * \param qBits: Number of bits for qMax
 * \param roundingMode: Rounding mode to be used
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitAsym(uint8_t qBits, roundingMode_t roundingMode);

#endif // QUANTIZATIONAPI_H
