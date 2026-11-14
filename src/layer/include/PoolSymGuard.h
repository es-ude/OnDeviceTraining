#ifndef ODT_POOL_SYM_GUARD_H
#define ODT_POOL_SYM_GUARD_H

#include <stdlib.h>

#include "Common.h"
#include "Quantization.h"
#include "Tensor.h"

/* The pool SYM_INT32 kernels sum operand-width mantissas in an int32 (window
 * sum or backward scatter, no products). Call it once per kernel, before the
 * loop, with that kernel's worst-case term count per accumulator. Requires
 * qMaxBits in [1,16] and maxTerms < 2^(32-qMaxBits); qMaxBits == 0 would make
 * the shift UB. The operand must be SYM_INT32, which the executeOp prologue
 * guarantees inside an ARITH_SYM_INT32 kernel. */
#define POOL_SYM_VALUESUM_QMAXBITS 16u
static inline void poolValidateSymValueSum(const tensor_t *operand, size_t maxTerms,
                                           const char *op) {
    const symInt32QConfig_t *qc = operand->quantization->qConfig;
    if (qc->qMaxBits == 0 || qc->qMaxBits > POOL_SYM_VALUESUM_QMAXBITS) {
        PRINT_ERROR("%s: operand qMaxBits (%u) outside the value-sum bound [1,%u]", op,
                    (unsigned)qc->qMaxBits, (unsigned)POOL_SYM_VALUESUM_QMAXBITS);
        exit(1);
    }
    size_t bound = (size_t)1 << (32u - qc->qMaxBits);
    if (maxTerms >= bound) {
        PRINT_ERROR("%s: worst-case summed terms (%zu) reach the value-sum bound for "
                    "qMaxBits (%u) -- must be < 2^(32-qMaxBits) (%zu)",
                    op, maxTerms, (unsigned)qc->qMaxBits, bound);
        exit(1);
    }
}

#endif // ODT_POOL_SYM_GUARD_H
