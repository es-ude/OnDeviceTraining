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
        /* Epic PR2: BFP is a COMPUTE representation, not just storage -- the
         * D5 float-bridge staging rule of PR1 is retired and BFP derives
         * native ARITH_BFP (the documented breaking change of this PR).
         * Consequences worth knowing at this seam:
         *  - Fake-quant over BFP storage is still available, but no longer
         *    free: pin the math slots to ARITH_FLOAT32 explicitly instead of
         *    deriving them (the funnel then dequantizes BFP operands as it
         *    does for any other storage-only dtype).
         *  - PR2 ships the FORWARD only (Linear/Conv1d/ConvT1d). A model that
         *    derives all four layer slots from one BFP config -- what
         *    layerQuantInitUniform does -- trains its forward natively and
         *    then dies at the layer's backward kernel dispatch (every
         *    GEMM-family layer guards its three backward slots; the funnel's
         *    missing-bfpStage gate backstops FLOAT32-stored operands) until
         *    epic PR3 lands the BFP backward arms. See
         *    docs/conventions/arithmetic-bfp.md. */
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
