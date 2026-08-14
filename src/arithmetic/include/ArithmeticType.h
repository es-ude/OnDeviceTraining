#ifndef ENV5_RUNTIME_ARITHMETIC_TYPE_H
#define ENV5_RUNTIME_ARITHMETIC_TYPE_H

#include "Quantization.h"
#include "Rounding.h"

/* Declared compute representation of an op (design spec 2026-07-02
 * arithmetic-type-split, D1). BY VALUE in layer configs: no ownership,
 * no teardown. Only compute-capable representations exist here — storage
 * dtypes (SYM/ASYM/BOOL/INT32) are expressed on tensors/storage configs. */
typedef enum arithmeticType { ARITH_FLOAT32, ARITH_SYM_INT32, ARITH_BFP } arithmeticType_t;

typedef struct arithmetic {
    arithmeticType_t type;
    /* The OPERATION's rounding (#282): governs the funnel's SYM compute
     * intermediates AND the OUT_WRITE epilogue requant into a quantized
     * target. Derive via arithmeticFromQuantization (rounding == the storage
     * default; every layer does this) unless the op needs a rounding of its
     * own — the optimizer's training write-backs set it to
     * optim->writeBackRounding (#279). Bare conversions (executeConvert) and
     * the ACC epilogues stay storage-owned. */
    roundingMode_t roundingMode;
} arithmetic_t;

/* Derivation rule (spec D5, as amended by BFP epic PR2): FLOAT32 ->
 * ARITH_FLOAT32; SYM_INT32 -> ARITH_SYM_INT32; BFP -> ARITH_BFP; the
 * remaining storage-only dtypes (SYM/ASYM/BOOL/INT32) -> ARITH_FLOAT32
 * (float is the universal compute bridge). roundingMode is taken from the
 * qConfig when the dtype carries one, else HALF_AWAY — the storage mode is
 * the DEFAULT the op's rounding seeds from (#282); ops needing a different
 * rounding overwrite the field after deriving. Fake-quant over a NATIVE
 * dtype (BFP/SYM_INT32 storage, float compute) is expressed by pinning the
 * slot to ARITH_FLOAT32 rather than deriving it. */
arithmetic_t arithmeticFromQuantization(const quantization_t *q);

/* NULL -> {ARITH_FLOAT32, HALF_AWAY}; else identical to
 * arithmeticFromQuantization(q). Legacy callers may pass NULL quantizations
 * (arithmeticFromQuantization itself does not NULL-check) — this guard lets
 * them derive an arithmetic_t eagerly without crashing on that path. */
arithmetic_t arithmeticFromQuantizationOrDefault(const quantization_t *q);

#endif // ENV5_RUNTIME_ARITHMETIC_TYPE_H
