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
         *    does for any other storage-only dtype). Two layer families do
         *    NOT offer the pin on the BACKWARD, because their FLOAT32
         *    backwards run outside the funnel and raw-cast their wires: the
         *    norms reject it at the factory (PR5, R-N6 rules 4/7) and softmax
         *    fails fast at the arm's guard (PR6, R-S6 -- its forward pin
         *    still works).
         *  - Epic PR3: the GEMM family (Linear/Conv1d/ConvT1d) now runs
         *    natively end-to-end -- a model that derives all four layer slots
         *    from one BFP config -- what layerQuantInitUniform does -- trains
         *    its forward AND backward natively. Pools shipped with epic PR4,
         *    the norms (LayerNorm/GroupNorm) with epic PR5 and Softmax with
         *    epic PR6 (integer i-exp forward + funnel backward), so EVERY
         *    layer now derives a usable ARITH_BFP arm; only the loss
         *    functions stay fake-quant.
         *    See docs/conventions/arithmetic-bfp.md. */
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
