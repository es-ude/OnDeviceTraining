#define SOURCE_FILE "ARITHMETIC-TYPE"

#include <stddef.h>

#include "ArithmeticType.h"

arithmetic_t arithmeticFromQuantization(const quantization_t *q) {
    arithmetic_t a = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    switch (q->type) {
    case SYM_INT32:
        a.type = ARITH_SYM_INT32;
        a.roundingMode = ((symInt32QConfig_t *)q->qConfig)->roundingMode;
        break;
    case SYM:
        a.roundingMode = ((symQConfig_t *)q->qConfig)->roundingMode;
        break;
    case ASYM:
        a.roundingMode = ((asymQConfig_t *)q->qConfig)->roundingMode;
        break;
    case BFP:
        /* Fake-quant over BFP storage: pin ARITH_FLOAT32 instead of deriving. The
         * norms and Softmax refuse that pin on their backward. */
        a.type = ARITH_BFP;
        a.roundingMode = ((bfpQConfig_t *)q->qConfig)->roundingMode;
        break;
    case FLOAT32:
    case INT32:
    case BOOL:
    default:
        break;
    }
    return a;
}

arithmetic_t arithmeticFromQuantizationOrDefault(const quantization_t *q) {
    return (q == NULL) ? (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY}
                       : arithmeticFromQuantization(q);
}
