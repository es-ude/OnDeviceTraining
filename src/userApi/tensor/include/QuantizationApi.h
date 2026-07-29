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

/*! SymInt32 with explicit qMaxBits. Plain quantizationInitSymInt32(rm) uses the
 *  int12 operand default (ODT_SYM_OPERAND_QMAXBITS). Widths >16 need scale=1
 *  (raw-int, unvalidated); 32 is not cast-safe in the converters (#202). */
quantization_t *quantizationInitSymInt32WithBits(roundingMode_t roundingMode, uint8_t qMaxBits);

/*! Sub-byte symmetric quantization with explicit bit width and rounding. */
quantization_t *quantizationInitSym(uint8_t qBits, roundingMode_t roundingMode);

/*! Group-quant PR2: sub-byte symmetric quantization with an explicit group
 *  shape (numGroups, groupSize), mirroring quantizationInitSym's per-tensor
 *  {1,0}. Pairs with freeQuantization. See initSymQConfigGrouped for the
 *  shape invariant enforced at construction. */
quantization_t *quantizationInitSymGrouped(uint8_t qBits, roundingMode_t roundingMode,
                                           size_t numGroups, size_t groupSize);

/*! Initializes asym quantization.
 *
 * \param qBits: Number of bits for qMax
 * \param roundingMode: Rounding mode to be used
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitAsym(uint8_t qBits, roundingMode_t roundingMode);

/*! Group-quant PR4: packed asymmetric quantization with an explicit group
 *  shape (numGroups, groupSize), mirroring quantizationInitSymGrouped. Pairs
 *  with freeQuantization. See initAsymQConfigGrouped for the shape invariant
 *  and the D6 qBits ceiling enforced at construction. */
quantization_t *quantizationInitAsymGrouped(uint8_t qBits, roundingMode_t rm, size_t numGroups,
                                            size_t groupSize);

/* Note: uses (void) explicitly; siblings use () for historical K&R-style; the
 * _Static_assert in UnitTestTensorApi pattern-matches (*)(void). */
/*! Initializes bool quantization.
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitBool(void);

#endif // QUANTIZATIONAPI_H
