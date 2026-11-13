#ifndef ENV5_RUNTIME_ARITHMETIC_TYPE_H
#define ENV5_RUNTIME_ARITHMETIC_TYPE_H

#include "Quantization.h"
#include "Rounding.h"

/* Declared compute representation of an op. By value in layer configs: no ownership,
 * no teardown. Only compute-capable representations exist here — storage
 * dtypes (SYM/ASYM/BOOL/INT32) are expressed on tensors/storage configs. */
typedef enum arithmeticType { ARITH_FLOAT32, ARITH_SYM_INT32, ARITH_BFP } arithmeticType_t;

typedef struct arithmetic {
    arithmeticType_t type;
    /* The operation's rounding: governs the funnel's SYM compute intermediates and
     * the OUT_WRITE requant into a quantized target. Derive it with
     * arithmeticFromQuantization unless the op needs its own (the optimizer's
     * write-backs use optim->writeBackRounding). Bare conversions (executeConvert)
     * and the ACC epilogues keep the storage rounding. */
    roundingMode_t roundingMode;
} arithmetic_t;

/* FLOAT32 -> ARITH_FLOAT32, SYM_INT32 -> ARITH_SYM_INT32, BFP -> ARITH_BFP; the
 * storage-only dtypes (SYM/ASYM/BOOL/INT32) -> ARITH_FLOAT32, float being the
 * universal compute bridge. roundingMode comes from the qConfig if the dtype has
 * one, else HALF_AWAY. For fake-quant over a native dtype (BFP/SYM_INT32 storage,
 * float compute) set the slot to ARITH_FLOAT32 instead of deriving it. */
arithmetic_t arithmeticFromQuantization(const quantization_t *q);

/* Like arithmeticFromQuantization, but a NULL quantization gives
 * {ARITH_FLOAT32, HALF_AWAY}. Use it where a config slot may be unset. */
arithmetic_t arithmeticFromQuantizationOrDefault(const quantization_t *q);

#endif // ENV5_RUNTIME_ARITHMETIC_TYPE_H
