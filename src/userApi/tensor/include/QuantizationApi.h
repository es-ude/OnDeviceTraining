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

/*! SymInt32 with explicit qMaxBits in [1, 31]. Plain quantizationInitSymInt32(rm)
 *  uses the int12 operand default (ODT_SYM_OPERAND_QMAXBITS). Widths >16 need
 *  scale=1 (raw-int, unvalidated); 32 marks a raw kernel output, which is
 *  requantized into a <= 31-bit tensor before it is cloned or used as a target. */
quantization_t *quantizationInitSymInt32WithBits(roundingMode_t roundingMode, uint8_t qMaxBits);

/*! Sub-byte symmetric quantization with explicit bit width and rounding. */
quantization_t *quantizationInitSym(uint8_t qBits, roundingMode_t roundingMode);

/*! Sub-byte symmetric quantization with an explicit group shape (numGroups,
 *  groupSize); per-tensor is {1,0}. Pairs with freeQuantization. See
 *  initSymQConfigGrouped for the shape invariant enforced at construction. */
quantization_t *quantizationInitSymGrouped(uint8_t qBits, roundingMode_t roundingMode,
                                           size_t numGroups, size_t groupSize);

/*! Per-tensor block-floating-point quantization with explicit mantissa/exponent
 *  widths and rounding. Pairs with freeQuantization. */
quantization_t *quantizationInitBfp(uint8_t mantissaBits, uint8_t exponentBits,
                                    roundingMode_t roundingMode);

/*! Block-floating-point quantization with an explicit group shape (numGroups,
 *  groupSize); per-tensor is {1,0}. Pairs with freeQuantization. See
 *  initBfpQConfigGrouped for the shape invariant enforced at construction. */
quantization_t *quantizationInitBfpGrouped(uint8_t mantissaBits, uint8_t exponentBits,
                                           roundingMode_t roundingMode, size_t numGroups,
                                           size_t groupSize);

/*! Initializes asym quantization.
 *
 * \param qBits: Number of bits for qMax
 * \param roundingMode: Rounding mode to be used
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitAsym(uint8_t qBits, roundingMode_t roundingMode);

/*! Packed asymmetric quantization with an explicit group shape (numGroups,
 *  groupSize); per-tensor is {1,0}. Pairs with freeQuantization. See
 *  initAsymQConfigGrouped for the shape invariant and qBits range enforced at
 *  construction. */
quantization_t *quantizationInitAsymGrouped(uint8_t qBits, roundingMode_t rm, size_t numGroups,
                                            size_t groupSize);

/*! Initializes bool quantization.
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitBool(void);

#endif // QUANTIZATIONAPI_H
