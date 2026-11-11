#define SOURCE_FILE "DYADIC_SCALE"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "DyadicScale.h"

/* IEEE-754 binary32: 23 stored fraction bits, a 24-bit significand. m keeps
 * the top 15 significand bits, so 9 bits are dropped. A canonical m has its
 * leading one at bit 14, a float significand at bit 23, so
 * e = (biased exponent - 127) - 14. */
#define FLOAT_FRAC_BITS 23u
#define FLOAT_FRAC_MASK 0x007FFFFFu
#define FLOAT_HIDDEN_BIT 0x00800000u
#define FLOAT_EXP_ALL_ONES 255u
#define FLOAT_DROPPED_BITS 9u
#define FLOAT_BIAS_TO_E 141
#define M_FRAC_MASK 0x3FFFu

#define RATIO_OPERAND_MAX 0x80000000u
#define QUOTIENT_BITS_DONE 0x8000u /* 16 significant quotient bits collected */
#define PRODUCT_TOP_BIT 45         /* m * num < 2^15 * 2^31 = 2^46 */

static bool isCanonical(odtScale_t s) {
    return s.m >= ODT_SCALE_M_MIN && s.m <= ODT_SCALE_M_MAX && s.e >= ODT_SCALE_E_MIN &&
           s.e <= ODT_SCALE_E_MAX;
}

static void requireCanonical(odtScale_t s, const char *caller) {
    if (!isCanonical(s)) {
        PRINT_ERROR("%s: {m = %u, e = %d} is not a canonical scale (m in [%d, %d], e in [%d, %d])",
                    caller, (unsigned)s.m, (int)s.e, ODT_SCALE_M_MIN, ODT_SCALE_M_MAX,
                    ODT_SCALE_E_MIN, ODT_SCALE_E_MAX);
        exit(1);
    }
}

/* Round the truncated mantissa m0 in [2^14, 2^15) once. roundBit is the first
 * dropped bit, sticky whether any lower dropped bit (or division remainder) is
 * non-zero. A carry out of 32767 renormalises to 16384 one binade up. The
 * exponent stays int32 until it is validated, then narrows to int16. */
static odtScale_t roundAndPack(uint32_t m0, bool roundBit, bool sticky, int32_t e,
                               odtScaleRound_t r, const char *caller) {
    switch (r) {
    case ODT_SCALE_NEAREST:
        m0 += roundBit ? 1u : 0u;
        break;
    case ODT_SCALE_CEIL:
        m0 += (roundBit || sticky) ? 1u : 0u;
        break;
    case ODT_SCALE_FLOOR:
        break;
    default:
        PRINT_ERROR("%s: unknown rounding mode %d", caller, (int)r);
        exit(1);
    }
    if (m0 > ODT_SCALE_M_MAX) {
        m0 = ODT_SCALE_M_MIN;
        e++;
    }
    if (e < ODT_SCALE_E_MIN || e > ODT_SCALE_E_MAX) {
        PRINT_ERROR("%s: result m = %" PRIu32 " * 2^%" PRId32
                    " is outside the admissible range [2^-126, FLT_MAX]",
                    caller, m0, e);
        exit(1);
    }
    odtScale_t out = {.m = (uint16_t)m0, .e = (int16_t)e};
    return out;
}

odtScale_t odtScaleFromFloat(float x, odtScaleRound_t r) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    /* The sign bit lands at bit 8 of biased: negatives (including -0) read
     * >= 256, Inf/NaN 255, zero and subnormals 0. */
    uint32_t biased = bits >> FLOAT_FRAC_BITS;
    if (biased == 0u || biased >= FLOAT_EXP_ALL_ONES) {
        PRINT_ERROR("input bits 0x%08" PRIx32 " are not a positive normal float", bits);
        exit(1);
    }
    uint32_t sig = (bits & FLOAT_FRAC_MASK) | FLOAT_HIDDEN_BIT;
    uint32_t m0 = sig >> FLOAT_DROPPED_BITS;
    bool roundBit = ((sig >> (FLOAT_DROPPED_BITS - 1u)) & 1u) != 0u;
    bool sticky = (sig & ((1u << (FLOAT_DROPPED_BITS - 1u)) - 1u)) != 0u;
    return roundAndPack(m0, roundBit, sticky, (int32_t)biased - FLOAT_BIAS_TO_E, r,
                        "odtScaleFromFloat");
}

float odtScaleToFloat(odtScale_t s) {
    requireCanonical(s, "odtScaleToFloat");
    uint32_t biased = (uint32_t)((int32_t)s.e + FLOAT_BIAS_TO_E);
    uint32_t bits =
        (biased << FLOAT_FRAC_BITS) | (((uint32_t)s.m & M_FRAC_MASK) << FLOAT_DROPPED_BITS);
    float x;
    memcpy(&x, &bits, sizeof x);
    return x;
}

odtScale_t odtScaleMul(odtScale_t a, odtScale_t b, odtScaleRound_t r) {
    requireCanonical(a, "odtScaleMul");
    requireCanonical(b, "odtScaleMul");
    /* [2^28, 2^30): exact in uint32. Keep its top 15 bits. */
    uint32_t p = (uint32_t)a.m * (uint32_t)b.m;
    uint32_t shift = (p >= (1u << 29)) ? 15u : 14u;
    uint32_t m0 = p >> shift;
    bool roundBit = ((p >> (shift - 1u)) & 1u) != 0u;
    bool sticky = (p & ((1u << (shift - 1u)) - 1u)) != 0u;
    int32_t e = (int32_t)a.e + (int32_t)b.e + (int32_t)shift;
    return roundAndPack(m0, roundBit, sticky, e, r, "odtScaleMul");
}

/* s * num / den without a division operator or a 64-bit type:
 * 1. P = m * num (< 2^46) as hi:lo words from num = nh * 2^16 + nl, so each
 *    partial product fits uint32.
 * 2. Restoring long division of P by den, one bit per step from the top of P
 *    (then zero bits once P is exhausted), until the quotient holds 16
 *    significant bits. The remainder stays below den <= 2^31, so the shifted
 *    remainder fits uint32.
 * 3. The 16th quotient bit is the round bit; the sticky bit is a non-zero
 *    remainder or any P bit not yet streamed. */
odtScale_t odtScaleMulRatio(odtScale_t s, uint32_t num, uint32_t den, odtScaleRound_t r) {
    requireCanonical(s, "odtScaleMulRatio");
    if (num == 0u || num > RATIO_OPERAND_MAX) {
        PRINT_ERROR("num = %" PRIu32 " is outside [1, 2^31]", num);
        exit(1);
    }
    if (den == 0u || den > RATIO_OPERAND_MAX) {
        PRINT_ERROR("den = %" PRIu32 " is outside [1, 2^31]", den);
        exit(1);
    }

    uint32_t m = s.m;
    uint32_t highPart = m * (num >> 16);      /* < 2^15 * 2^15 */
    uint32_t lowPart = m * (num & 0xFFFFu);   /* < 2^15 * 2^16 */
    uint32_t lo = (highPart << 16) + lowPart; /* modulo 2^32; the carry is recovered below */
    uint32_t hi = (highPart >> 16) + (lo < lowPart ? 1u : 0u);

    /* pos is the P bit streamed next; below 0 the stream is zeros. When
     * hi < den every quotient bit at or above 2^32 is zero and the remainder
     * after streaming hi is hi itself, so start at the low word. */
    int32_t pos;
    uint32_t rem;
    if (hi < den) {
        rem = hi;
        pos = 31;
    } else {
        rem = 0u;
        pos = PRODUCT_TOP_BIT;
    }
    uint32_t q = 0u;
    while (q < QUOTIENT_BITS_DONE) {
        uint32_t bit = 0u;
        if (pos >= 32) {
            bit = (hi >> (uint32_t)(pos - 32)) & 1u;
        } else if (pos >= 0) {
            bit = (lo >> (uint32_t)pos) & 1u;
        }
        rem = (rem << 1) | bit;
        q <<= 1;
        if (rem >= den) {
            rem -= den;
            q |= 1u;
        }
        pos--;
    }

    /* P bits [0, pos] were not streamed. They all lie in lo: P / den < 2^46
     * puts the leading quotient bit at or below 2^45, so the 16th lands at or
     * below 2^30 and pos <= 29 here. */
    bool unstreamed = pos >= 0 && (lo & (0xFFFFFFFFu >> (uint32_t)(31 - pos))) != 0u;

    /* The last quotient bit has weight 2^(pos + 1), so q >> 1 has 2^(pos + 2). */
    int32_t e = (int32_t)s.e + pos + 2;
    return roundAndPack(q >> 1, (q & 1u) != 0u, rem != 0u || unstreamed, e, r, "odtScaleMulRatio");
}

int odtScaleCmp(odtScale_t a, odtScale_t b) {
    requireCanonical(a, "odtScaleCmp");
    requireCanonical(b, "odtScaleCmp");
    /* Normalised m: a larger exponent is a larger value. */
    if (a.e != b.e) {
        return a.e < b.e ? -1 : 1;
    }
    if (a.m != b.m) {
        return a.m < b.m ? -1 : 1;
    }
    return 0;
}

bool odtScaleEq(odtScale_t a, odtScale_t b) {
    requireCanonical(a, "odtScaleEq");
    requireCanonical(b, "odtScaleEq");
    return a.m == b.m && a.e == b.e;
}

float odtScaleCanonFloat(float x, odtScaleRound_t r) {
    return odtScaleToFloat(odtScaleFromFloat(x, r));
}

#ifdef ODT_SCALE_STRICT
void odtScaleAssertCanonical(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    uint32_t biased = bits >> FLOAT_FRAC_BITS;
    if (biased == 0u || biased >= FLOAT_EXP_ALL_ONES ||
        (bits & ((1u << FLOAT_DROPPED_BITS) - 1u)) != 0u) {
        PRINT_ERROR("odtScaleAssertCanonical: input bits 0x%08" PRIx32
                    " are not a canonical scale (ODT_SCALE_STRICT)",
                    bits);
        exit(1);
    }
}
#endif
