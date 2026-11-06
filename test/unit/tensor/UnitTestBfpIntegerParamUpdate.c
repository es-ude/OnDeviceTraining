#include "BfpAccumulateRef.h"
#include "BfpKernelSupport.h"
#include "DTypes.h"
#include "DeathTest.h"
#include "Quantization.h"
#include "RNG.h"
#include "RngPinnedDraw.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "unity.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The integer SGD parameter update w' = w - lr * u on a BFP parameter w from
 * a BFP update u (the stored new momentum, or the gradient without momentum)
 * (paramUpdateBfpInPlace, arithmetic-bfp.md §15.7). Every expected value comes
 * from the exact 128-bit reference below, written from the documented
 * result, not from the engine's quotient/remainder code. RNG discipline: the
 * engine draws exactly one value per element of w, in element order, from
 * the stream it is given; the reference consumes its own copy of the same
 * stream in the same order, and every SR comparison also checks the stream
 * position afterwards. */

void setUp(void) {}
void tearDown(void) {}

static uint32_t testRngState = 0x3A71C0DEu;

static uint32_t testRand(void) {
    uint32_t x = testRngState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    testRngState = x;
    return x;
}

static int32_t testRandInt(int32_t lo, int32_t hi) { /* inclusive */
    return lo + (int32_t)(testRand() % (uint32_t)(hi - lo + 1));
}

/* ---- fixtures --------------------------------------------------------- */

#define PU_MAX_N 6144
#define PU_MAX_GROUPS 4096

typedef struct {
    size_t n;
    size_t dims[1];
    size_t order[1];
    shape_t shape;
    uint8_t exps[PU_MAX_GROUPS];
    bfpQConfig_t qc;
    quantization_t q;
    uint8_t data[PU_MAX_N * 2 + 8];
    tensor_t t;
} puFixture_t;

static puFixture_t fixW; /* the parameter, updated in place */
static puFixture_t fixU; /* the update (stored momentum or gradient), read only */

static void puInit(puFixture_t *f, size_t n, size_t numGroups, size_t groupSize, uint8_t m,
                   uint8_t e, const int32_t *codes, const uint8_t *stored) {
    TEST_ASSERT_TRUE(n <= PU_MAX_N && numGroups <= PU_MAX_GROUPS);
    f->n = n;
    f->dims[0] = n;
    f->order[0] = 0;
    setShape(&f->shape, f->dims, 1, f->order);
    memcpy(f->exps, stored, numGroups);
    /* the storage mode is deliberately the OTHER one: the engine must round by
     * the mode it is given (the optimizer's write-back mode), never by this */
    f->qc = (bfpQConfig_t){.exponents = f->exps,
                           .numGroups = numGroups,
                           .groupSize = groupSize,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = m,
                           .exponentBits = e};
    initBfpQuantization(&f->qc, &f->q);
    memset(f->data, 0xA5, sizeof f->data);
    if (n > 0) {
        static int32_t tmp[PU_MAX_N];
        memcpy(tmp, codes, n * sizeof(int32_t));
        byteConversion((uint8_t *)tmp, 32, f->data, m, n);
    }
    setTensorValues(&f->t, f->data, &f->shape, &f->q, NULL);
}

static void puCodes(const puFixture_t *f, int32_t *out) {
    unpackSignExtend(f->data, f->qc.mantissaBits, 0, out, f->n);
}

static void puRun(bfpScalar_t lr, roundingMode_t rm, rngStream_t *rs) {
    paramUpdateBfpInPlace(&fixW.t, &fixU.t, lr, rm, rs);
}

static bfpScalar_t lrOf(float x) {
    return bfpScalarFromFloat(x, "test lr");
}

static const bfpScalar_t LR_ONE = {.q = 16384, .e = -14};
static const bfpScalar_t LR_ZERO = {.q = 0, .e = 0};

/* ---- the exact reference ----------------------------------------------
 * Per destination (parameter) group, with E_w the unbiased stored exponent of
 * the parameter's group, E_u that of the update's group (the update's only
 * group when it is per-tensor) and lr = q * 2^e:
 *   terms   w_i on 2^E_w and -(u_i * q) on 2^(E_u + e) (exact);
 *   E_c   = max(E_w + bitlen(A_w), E_u + e + bitlen(q * A_u)) - 30, capped
 *           at the parameter's capE (a zero side has no top);
 *   E_d   = the smallest E >= E_c with max|sum| <= qMax * 2^E, clamped to
 *           [-bias, capE]; an all-zero sum is the zero state;
 *   code  = |sum| / 2^E_d rounded once (F = the top 32 remainder bits; SR
 *           lifts iff u < F, HALF_AWAY iff F >= 2^31), sign after, clamped;
 *   draws = one per element in element order under SR, on every path. */
static void refParamUpdate(const int32_t *w, const uint8_t *wStored, uint8_t mw, uint8_t ew,
                           const int32_t *u, const uint8_t *uStored, uint8_t eu, bool uPerTensor,
                           size_t n, size_t numGroups, size_t groupSize, bfpScalar_t lr,
                           roundingMode_t rm, rngStream_t *rs, int32_t *outCodes,
                           uint8_t *outStored) {
    const int biasW = (1 << (ew - 1)) - 1;
    const int biasU = (1 << (eu - 1)) - 1;
    int cap = (1 << ew) - 1;
    if (cap > biasW + 127) {
        cap = biasW + 127;
    }
    const int capE = cap - biasW;
    const uint32_t posCap = (1u << (mw - 1)) - 1u;
    const uint32_t negCap = 1u << (mw - 1);
    const size_t gsz = groupSize == 0 ? n : groupSize;
    const bool sr = rm == SR_HALF_AWAY;
    const int64_t q = lr.q;
    for (size_t gi = 0; gi < numGroups; gi++) {
        const size_t start = gi * gsz;
        const size_t end = start + gsz;
        const int eW = (int)wStored[gi] - biasW;
        const int eU = (int)uStored[uPerTensor ? 0 : gi] - biasU + lr.e;
        uint64_t aw = 0;
        uint64_t au = 0;
        for (size_t i = start; i < end; i++) {
            const int64_t pu = (int64_t)u[i] * q;
            const uint64_t mu = pu < 0 ? (uint64_t)(-pu) : (uint64_t)pu;
            const uint64_t mw64 = w[i] < 0 ? (uint64_t)(-(int64_t)w[i]) : (uint64_t)w[i];
            aw = mw64 > aw ? mw64 : aw;
            au = mu > au ? mu : au;
        }
        static bfpAccRefI128_t vals[PU_MAX_N];
        bfpAccRefU128_t vmax = 0;
        int ec = 0;
        int emin = 0;
        if (aw != 0 || au != 0) {
            const int topW = aw ? eW + bfpAccRefBitLen(aw) : INT32_MIN;
            const int topU = au ? eU + bfpAccRefBitLen(au) : INT32_MIN;
            ec = (topW > topU ? topW : topU) - 30;
            ec = ec > capE ? capE : ec;
            emin = aw == 0 ? eU : (au == 0 ? eW : (eW < eU ? eW : eU));
            TEST_ASSERT_TRUE_MESSAGE((aw == 0 || eW - emin <= 90) && (au == 0 || eU - emin <= 90),
                                     "refParamUpdate: exponent gap outside the reference");
            for (size_t i = start; i < end; i++) {
                const bfpAccRefI128_t tw =
                    aw == 0 ? 0 : (bfpAccRefI128_t)w[i] * ((bfpAccRefI128_t)1 << (eW - emin));
                const bfpAccRefI128_t tu = au == 0 ? 0
                                                   : (bfpAccRefI128_t)((int64_t)u[i] * q) *
                                                         ((bfpAccRefI128_t)1 << (eU - emin));
                vals[i - start] = tw - tu;
                const bfpAccRefU128_t a = bfpAccRefAbs(tw - tu);
                vmax = a > vmax ? a : vmax;
            }
        }
        int ed = 0;
        if (vmax == 0) {
            outStored[gi] = (uint8_t)biasW;
        } else {
            int E = ec;
            for (;; E++) {
                const int s = E - emin;
                bool ok;
                if (s >= 0) {
                    ok = s >= 100 || vmax <= ((bfpAccRefU128_t)posCap << s);
                } else {
                    ok = (vmax << (unsigned)(-s)) <= posCap;
                }
                if (ok || E >= capE) {
                    break;
                }
            }
            E = E < -biasW ? -biasW : E;
            E = E > capE ? capE : E;
            outStored[gi] = (uint8_t)(E + biasW);
            ed = E;
        }
        for (size_t i = start; i < end; i++) {
            const uint32_t d = sr ? rngNextU32(rs) : 0u;
            outCodes[i] =
                vmax == 0 ? 0 : bfpAccRefRound(vals[i - start], ed - emin, sr, d, posCap, negCap);
        }
    }
}

typedef struct {
    size_t n, numGroups, groupSize;
    uint8_t mw, ew, mu, eu;
    bool uPerTensor;
} puGeo_t;

/* Runs engine and reference from the same seed; asserts codes, exponents,
 * the update untouched, and (SR) the stream position: exactly n draws. */
static void assertParamUpdateMatchesReference(const puGeo_t *geo, const int32_t *w,
                                              const uint8_t *ws, const int32_t *u,
                                              const uint8_t *us, bfpScalar_t lr, roundingMode_t rm,
                                              uint32_t seed) {
    static int32_t expCodes[PU_MAX_N];
    static int32_t gotCodes[PU_MAX_N];
    static int32_t uAfter[PU_MAX_N];
    static uint8_t expStored[PU_MAX_GROUPS];
    static uint8_t uDataBefore[sizeof fixU.data];
    const size_t n = geo->n;
    const size_t uGroups = geo->uPerTensor ? 1 : geo->numGroups;
    rngStream_t refStream = rngOpen((rngKey_t){{seed, 33u}});
    rngStream_t engStream = rngOpen((rngKey_t){{seed, 33u}});
    refParamUpdate(w, ws, geo->mw, geo->ew, u, us, geo->eu, geo->uPerTensor, n, geo->numGroups,
                   geo->groupSize, lr, rm, &refStream, expCodes, expStored);
    puInit(&fixW, n, geo->numGroups, geo->groupSize, geo->mw, geo->ew, w, ws);
    puInit(&fixU, n, uGroups, geo->uPerTensor ? 0 : geo->groupSize, geo->mu, geo->eu, u, us);
    memcpy(uDataBefore, fixU.data, sizeof fixU.data);
    puRun(lr, rm, rm == SR_HALF_AWAY ? &engStream : NULL);
    puCodes(&fixW, gotCodes);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expStored, fixW.exps, geo->numGroups);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expCodes, gotCodes, n);
    TEST_ASSERT_EQUAL_MEMORY(uDataBefore, fixU.data, sizeof fixU.data);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(us, fixU.exps, uGroups);
    puCodes(&fixU, uAfter);
    TEST_ASSERT_EQUAL_INT32_ARRAY(u, uAfter, n);
    if (rm == SR_HALF_AWAY) {
        TEST_ASSERT_EQUAL_UINT32(rngNextU32(&refStream), rngNextU32(&engStream));
    }
}

static int clampInt(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

/* Random codes for one group: flavour 0 zero, 1 tight, 2 not tight (fewer
 * bits), 3 the negative code bound present. */
static void randomGroupCodes(int32_t *codes, size_t len, uint8_t m, int flavour) {
    const int32_t qMax = (1 << (m - 1)) - 1;
    const int32_t lim = flavour == 2 ? (int32_t)(1u << testRandInt(0, m - 2)) : qMax;
    for (size_t i = 0; i < len; i++) {
        codes[i] = flavour == 0 ? 0 : testRandInt(-lim, lim);
    }
    if (flavour == 3 && len > 0) {
        codes[0] = -(1 << (m - 1));
    }
}

/* ---- dead zone (written first: the review's #1 risk) -------------------
 * w = 64 code steps on stored 130; the update lr * u = 1/256 of a step, as
 * lr = 1 with u = one code eight binades lower, and as lr = 2^-8 with u = one
 * code on the parameter's own exponent. w' = 64 - 1/256 = 63 + 255/256 stays
 * on the grid; SR lifts back to 64 iff the draw < 255 * 2^24, so 63 appears
 * iff the draw >= 255 * 2^24 -- the small update survives with probability
 * 1/256 instead of being erased. The opposite sign moves to 65 iff the draw <
 * 2^24. */
void testParamUpdateDeadZonePinned(void) {
    const uint8_t stored = 130;
    const int32_t w64[1] = {64};
    const int32_t uPlus[1] = {1};
    const int32_t uMinus[1] = {-1};
    const struct {
        bfpScalar_t lr;
        uint8_t uStored;
    } scales[] = {{LR_ONE, stored - 8}, {{.q = 16384, .e = -22}, stored}};
    const struct {
        const int32_t *u;
        uint32_t draw;
        int32_t expect;
    } cases[] = {
        {uPlus, 255u << 24, 63},       {uPlus, (255u << 24) - 1u, 64},
        {uPlus, 0xFFFFFFFFu, 63},      {uPlus, 1u, 64},
        {uMinus, (1u << 24) - 1u, 65}, {uMinus, 1u << 24, 64},
        {uMinus, 0xFFFFFFFFu, 64},     {uMinus, 2u, 65},
    };
    for (size_t k = 0; k < sizeof scales / sizeof scales[0]; k++) {
        for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            puInit(&fixW, 1, 1, 0, 8, 8, w64, &stored);
            puInit(&fixU, 1, 1, 0, 8, 8, cases[c].u, &scales[k].uStored);
            rngStream_t s = rngStreamWithNextDraw(cases[c].draw);
            puRun(scales[k].lr, SR_HALF_AWAY, &s);
            int32_t got;
            puCodes(&fixW, &got);
            TEST_ASSERT_EQUAL_UINT8(stored, fixW.exps[0]);
            TEST_ASSERT_EQUAL_INT32(cases[c].expect, got);
        }
    }
}

/* The same update over 102400 independent draws: the code moves to 63 about
 * 400 times (binomial, p = 1/256; 5 sigma ~ 100), never zero times. Then on
 * a per-channel parameter under a per-tensor update, as in the HAR weights:
 * 16 groups of 64 whose exponents alternate (stored 110 / 111), one lr = 4
 * with u = one code on stored 100, so the even groups see 1/256 of a step
 * and the odd ones 1/512 -- each group moves with its OWN probability. */
void testParamUpdateDeadZoneStatistical(void) {
    static int32_t w[1024];
    static int32_t u[1024];
    static int32_t got[1024];
    for (size_t i = 0; i < 1024; i++) {
        w[i] = 64;
        u[i] = 1;
    }
    rngStream_t s = rngOpen((rngKey_t){{0xC0FFEEu, 7u}});
    const uint8_t wStored = 130;
    const uint8_t uS = wStored - 8;
    size_t moves = 0;
    for (int rep = 0; rep < 100; rep++) {
        puInit(&fixW, 1024, 1, 0, 8, 8, w, &wStored);
        puInit(&fixU, 1024, 1, 0, 8, 8, u, &uS);
        puRun(LR_ONE, SR_HALF_AWAY, &s);
        puCodes(&fixW, got);
        TEST_ASSERT_EQUAL_UINT8(wStored, fixW.exps[0]);
        for (size_t i = 0; i < 1024; i++) {
            TEST_ASSERT_TRUE(got[i] == 63 || got[i] == 64);
            moves += got[i] == 63;
        }
    }
    TEST_ASSERT_TRUE(moves >= 300 && moves <= 500);

    uint8_t ws[16];
    for (size_t gi = 0; gi < 16; gi++) {
        ws[gi] = (uint8_t)(110 + gi % 2);
    }
    const uint8_t uStored = 100;
    size_t movesEven = 0;
    size_t movesOdd = 0;
    for (int rep = 0; rep < 100; rep++) {
        puInit(&fixW, 1024, 16, 64, 8, 8, w, ws);
        puInit(&fixU, 1024, 1, 0, 8, 8, u, &uStored);
        puRun((bfpScalar_t){.q = 16384, .e = -12}, SR_HALF_AWAY, &s); /* lr = 4 */
        puCodes(&fixW, got);
        for (size_t gi = 0; gi < 16; gi++) {
            TEST_ASSERT_EQUAL_UINT8(ws[gi], fixW.exps[gi]);
        }
        for (size_t i = 0; i < 1024; i++) {
            TEST_ASSERT_TRUE(got[i] == 63 || got[i] == 64);
            if ((i / 64) % 2 == 0) {
                movesEven += got[i] == 63;
            } else {
                movesOdd += got[i] == 63;
            }
        }
    }
    /* 51200 trials each: 200 expected (5 sigma ~ 70) and 100 (5 sigma ~ 50) */
    TEST_ASSERT_TRUE(movesEven >= 130 && movesEven <= 270);
    TEST_ASSERT_TRUE(movesOdd >= 50 && movesOdd <= 150);
}

/* ---- the exact reference over random parameters ------------------------ */

/* Random parameters and updates against the exact reference: per-tensor and
 * grouped parameters (the HAR per-channel weights), the update per-tensor
 * (HAR: momentum and gradient storage) or in the parameter's own geometry;
 * group size 1 with more than 257 groups wraps the window rings, groups
 * straddle the 256-element chunk boundaries; byte, sub-byte and 16-bit widths,
 * parameter and update in DIFFERENT widths and exponent ranges (narrow
 * parameter exponents reach the cap and the floor); zero sides, the negative
 * code bound, near-cancelling groups (lr * u ~ w); lr in {0, 1, powers of two,
 * 0.01, 0.05, 1e-6, > 1, ...}; HALF_AWAY and SR. */
void testParamUpdateMatchesExactReferenceOnRandomParameters(void) {
    static int32_t w[PU_MAX_N];
    static int32_t u[PU_MAX_N];
    static uint8_t ws[PU_MAX_GROUPS];
    static uint8_t us[PU_MAX_GROUPS];
    const puGeo_t geos[] = {
        {1, 1, 0, 8, 8, 8, 8, false},       {6, 1, 0, 8, 8, 8, 8, false},
        {600, 1, 0, 8, 8, 8, 8, false},     {4096, 1, 0, 8, 8, 8, 8, false},
        {700, 700, 1, 8, 8, 8, 8, false},   {700, 700, 1, 8, 8, 8, 8, true},
        {699, 233, 3, 5, 8, 8, 8, true},    {1000, 10, 100, 8, 8, 6, 7, true},
        {640, 10, 64, 16, 8, 16, 8, false}, {6144, 64, 96, 8, 8, 8, 8, true},
        {513, 1, 0, 2, 8, 8, 8, false},     {770, 110, 7, 12, 8, 4, 8, false},
        {300, 3, 100, 6, 4, 8, 8, true},    {32, 2, 16, 8, 3, 8, 6, false},
        {2048, 2048, 1, 8, 4, 8, 8, true},  {1024, 4, 256, 16, 8, 8, 8, true},
        {999, 333, 3, 7, 6, 3, 5, false},   {257, 257, 1, 8, 8, 16, 8, true},
        {511, 1, 0, 16, 5, 16, 8, false},
    };
    const float lrs[] = {0.0f,    1.0f,  0.5f,       0.0625f, 0.01f, 0.05f,
                         0.0011f, 1e-6f, 0.6180339f, 3.7f,    0.1f,  0.999f};
    for (int round = 0; round < 2; round++) {
        for (size_t c = 0; c < sizeof geos / sizeof geos[0]; c++) {
            const puGeo_t *geo = &geos[c];
            const size_t n = geo->n;
            const size_t gsz = geo->groupSize == 0 ? n : geo->groupSize;
            const int biasW = (1 << (geo->ew - 1)) - 1;
            const int biasU = (1 << (geo->eu - 1)) - 1;
            const int maxW = (1 << geo->ew) - 1 > biasW + 127 ? biasW + 127 : (1 << geo->ew) - 1;
            const int maxU = (1 << geo->eu) - 1 > biasU + 127 ? biasU + 127 : (1 << geo->eu) - 1;
            for (size_t k = 0; k < sizeof lrs / sizeof lrs[0]; k++) {
                const bfpScalar_t lr = lrOf(lrs[k]);
                const int lrE = lr.q == 0 ? 0 : lr.e + 14; /* lr ~ 2^lrE */
                /* a per-tensor update sits near the first group's exponent */
                const int base = testRandInt(0, maxW);
                bool zeroU = false;
                for (size_t gi = 0; gi < geo->numGroups; gi++) {
                    const int flavourW = testRandInt(0, 9);
                    const int flavourU = testRandInt(0, 9);
                    if (geo->uPerTensor) {
                        ws[gi] = (uint8_t)clampInt(base + testRandInt(-12, 12), 0, maxW);
                    } else {
                        ws[gi] = (uint8_t)(flavourW < 3 ? testRandInt(0, maxW < 12 ? maxW : 12)
                                                        : testRandInt(0, maxW));
                    }
                    const int eW = (int)ws[gi] - biasW;
                    if (!geo->uPerTensor || gi == 0) {
                        us[gi] =
                            (uint8_t)clampInt(eW - lrE + testRandInt(-24, 24) + biasU, 0, maxU);
                    }
                    int32_t *wg = &w[gi * gsz];
                    int32_t *ug = &u[gi * gsz];
                    randomGroupCodes(wg, gsz, geo->mw, flavourW == 0 ? 0 : 1 + flavourW % 3);
                    randomGroupCodes(ug, gsz, geo->mu, flavourU == 0 ? 0 : 1 + flavourU % 3);
                    if (flavourU == 9 && lr.q != 0 && !geo->uPerTensor) {
                        /* near-cancelling: u = round(w * 2^14 / q) on the
                         * exponent E_w - e - 14, so lr * u ~ w */
                        us[gi] = (uint8_t)clampInt(eW - lr.e - 14 + biasU, 0, maxU);
                        const int32_t uMax = (1 << (geo->mu - 1)) - 1;
                        const int sh = ((int)us[gi] - biasU) - (eW - lr.e - 14);
                        for (size_t i = 0; i < gsz; i++) {
                            int64_t p = (int64_t)wg[i] * 16384 / lr.q;
                            p = sh >= 0 ? (sh >= 62 ? 0 : p / ((int64_t)1 << sh))
                                        : p * ((int64_t)1 << (-sh > 30 ? 30 : -sh));
                            p += testRandInt(-1, 1);
                            p = p > uMax ? uMax : (p < -uMax ? -uMax : p);
                            ug[i] = (int32_t)p;
                        }
                    }
                    /* the reference's exact domain */
                    const int gap = ((int)us[geo->uPerTensor ? 0 : gi] - biasU + lr.e) - eW;
                    if (gap > 80 || gap < -80) {
                        if (geo->uPerTensor) {
                            zeroU = true;
                        } else {
                            memset(ug, 0, gsz * sizeof(int32_t));
                        }
                    }
                }
                if (zeroU) {
                    memset(u, 0, n * sizeof(int32_t));
                }
                const uint32_t seed = 0x9000u + (uint32_t)(round * 7919 + (int)c * 131 + (int)k);
                assertParamUpdateMatchesReference(geo, w, ws, u, us, lr, HALF_AWAY, seed);
                assertParamUpdateMatchesReference(geo, w, ws, u, us, lr, SR_HALF_AWAY, seed);
            }
        }
    }
}

/* lr != 2^k: the lr * u term has a tail on every element, carried to the one
 * final SR -- E[w'] equals the exact w - lr_enc * u (5 sigma). */
void testParamUpdateIsUnbiasedUnderSr(void) {
    static int32_t w[1024];
    static int32_t u[1024];
    static int32_t got[1024];
    const uint8_t stored = 128;
    const bfpScalar_t lr = lrOf(0.05f);
    const int32_t wCodes[] = {100, -77, 3, -1};
    const int32_t uCodes[] = {-5, 9, 0, 1};
    for (size_t c = 0; c < 4; c++) {
        for (size_t i = 0; i < 1024; i++) {
            w[i] = wCodes[c];
            u[i] = uCodes[c];
        }
        w[0] = 127; /* pins the group's grid: the exponent stays 128 */
        u[0] = 0;
        const uint8_t uStored = stored + 3; /* lr * u ~ 0.4 * u steps */
        const double exact = (double)wCodes[c] - (double)uCodes[c] * ldexp((double)lr.q, lr.e + 3);
        rngStream_t s = rngOpen((rngKey_t){{0xBEEF10u + (uint32_t)c, 5u}});
        double sum = 0.0;
        size_t cnt = 0;
        for (int rep = 0; rep < 60; rep++) {
            puInit(&fixW, 1024, 1, 0, 8, 8, w, &stored);
            puInit(&fixU, 1024, 1, 0, 8, 8, u, &uStored);
            puRun(lr, SR_HALF_AWAY, &s);
            TEST_ASSERT_EQUAL_UINT8(stored, fixW.exps[0]);
            puCodes(&fixW, got);
            for (size_t i = 1; i < 1024; i++) {
                sum += got[i];
                cnt++;
            }
        }
        const double mean = sum / (double)cnt;
        TEST_ASSERT_TRUE(mean > exact - 5 * 0.5 / 248.0 && mean < exact + 5 * 0.5 / 248.0);
    }
}

/* ---- exact cases -------------------------------------------------------- */

/* lr = 2^-k with the update k binades above the parameter: lr * u lands on
 * the parameter's grid, so w' = w - u exactly (every sum in [-127, 127],
 * absmax in [64, 127]) -- no element rounds under any draw. */
void testParamUpdatePowerOfTwoLrIsExact(void) {
    static int32_t w[500];
    static int32_t u[500];
    static int32_t got[500];
    for (size_t i = 0; i < 500; i++) {
        w[i] = testRandInt(-63, 63);
        u[i] = testRandInt(-64, 64);
    }
    w[250] = 63;
    u[250] = -64;
    const uint8_t stored = 120;
    for (int k = 0; k <= 12; k += 3) {
        const uint8_t uStored = (uint8_t)(stored + k);
        for (int sr = 0; sr < 2; sr++) {
            puInit(&fixW, 500, 1, 0, 8, 8, w, &stored);
            puInit(&fixU, 500, 1, 0, 8, 8, u, &uStored);
            rngStream_t s = rngOpen((rngKey_t){{99u + (uint32_t)k, 2u}});
            puRun((bfpScalar_t){.q = 16384, .e = (int16_t)(-14 - k)}, sr ? SR_HALF_AWAY : HALF_AWAY,
                  sr ? &s : NULL);
            puCodes(&fixW, got);
            TEST_ASSERT_EQUAL_UINT8(stored, fixW.exps[0]);
            for (size_t i = 0; i < 500; i++) {
                TEST_ASSERT_EQUAL_INT32(w[i] - u[i], got[i]);
            }
        }
    }
}

/* lr = 0: w' = w, re-gridded onto its own format -- a tight parameter keeps
 * codes and exponent, a loose one is renormalized by an exact left shift; no
 * rounding ever happens, whatever the draws, yet n draws are consumed. */
void testParamUpdateZeroLrIsTheParameterExactly(void) {
    static int32_t w[300];
    static int32_t u[300];
    static int32_t got[300];
    for (size_t i = 0; i < 300; i++) {
        w[i] = testRandInt(-127, 127);
        u[i] = testRandInt(-127, 127);
    }
    w[17] = 127;
    const uint8_t wStored = 101;
    const uint8_t uStored = 140;
    for (int sr = 0; sr < 2; sr++) {
        puInit(&fixW, 300, 1, 0, 8, 8, w, &wStored);
        puInit(&fixU, 300, 1, 0, 8, 8, u, &uStored);
        rngStream_t s = rngOpen((rngKey_t){{77u, 1u}});
        rngStream_t probe = rngOpen((rngKey_t){{77u, 1u}});
        puRun(LR_ZERO, sr ? SR_HALF_AWAY : HALF_AWAY, sr ? &s : NULL);
        puCodes(&fixW, got);
        TEST_ASSERT_EQUAL_UINT8(wStored, fixW.exps[0]);
        TEST_ASSERT_EQUAL_INT32_ARRAY(w, got, 300);
        if (sr) {
            for (int i = 0; i < 300; i++) {
                (void)rngNextU32(&probe);
            }
            TEST_ASSERT_EQUAL_UINT32(rngNextU32(&probe), rngNextU32(&s));
        }
    }
    /* loose parameter (absmax 20): exponent drops by 2, codes x4, exact */
    for (size_t i = 0; i < 300; i++) {
        w[i] = testRandInt(-20, 20);
    }
    w[3] = 20;
    puInit(&fixW, 300, 1, 0, 8, 8, w, &wStored);
    puInit(&fixU, 300, 1, 0, 8, 8, u, &uStored);
    rngStream_t s2 = rngOpen((rngKey_t){{78u, 1u}});
    puRun(LR_ZERO, SR_HALF_AWAY, &s2);
    puCodes(&fixW, got);
    TEST_ASSERT_EQUAL_UINT8(wStored - 2, fixW.exps[0]);
    for (size_t i = 0; i < 300; i++) {
        TEST_ASSERT_EQUAL_INT32(4 * w[i], got[i]);
    }
}

/* ---- cancellation ------------------------------------------------------ */

/* lr * u == w exactly (lr = 1, u = w on the same exponent): every sum is zero
 * -> the zero state (stored = bias, codes 0); under SR the n draws are still
 * consumed. */
void testParamUpdateExactCancellationIsTheZeroState(void) {
    static int32_t w[400];
    static int32_t got[400];
    for (size_t i = 0; i < 400; i++) {
        w[i] = testRandInt(-127, 127);
    }
    const uint8_t stored = 133;
    puInit(&fixW, 400, 1, 0, 8, 8, w, &stored);
    puInit(&fixU, 400, 1, 0, 8, 8, w, &stored);
    rngStream_t s = rngOpen((rngKey_t){{5u, 6u}});
    rngStream_t probe = rngOpen((rngKey_t){{5u, 6u}});
    puRun(LR_ONE, SR_HALF_AWAY, &s);
    puCodes(&fixW, got);
    TEST_ASSERT_EQUAL_UINT8(127, fixW.exps[0]);
    for (size_t i = 0; i < 400; i++) {
        TEST_ASSERT_EQUAL_INT32(0, got[i]);
        (void)rngNextU32(&probe);
    }
    TEST_ASSERT_EQUAL_UINT32(rngNextU32(&probe), rngNextU32(&s));
}

/* lr * u ~ w: the residues (|r| <= 3 code steps) are exact on a grid 5
 * binades finer -- the exponent comes from the RESULT, not from w or u. */
void testParamUpdateNearCancellationRegridsToTheResidue(void) {
    static int32_t w[400];
    static int32_t u[400];
    static int32_t got[400];
    for (size_t i = 0; i < 400; i++) {
        w[i] = testRandInt(-120, 120);
        u[i] = w[i] + testRandInt(-3, 3);
    }
    w[9] = 100;
    u[9] = 97; /* residue 3: absmax 3 -> 3 * 2^5 = 96 <= 127 */
    const uint8_t stored = 133;
    puInit(&fixW, 400, 1, 0, 8, 8, w, &stored);
    puInit(&fixU, 400, 1, 0, 8, 8, u, &stored);
    rngStream_t s = rngOpen((rngKey_t){{6u, 6u}});
    puRun(LR_ONE, SR_HALF_AWAY, &s);
    puCodes(&fixW, got);
    TEST_ASSERT_EQUAL_UINT8(stored - 5, fixW.exps[0]);
    for (size_t i = 0; i < 400; i++) {
        TEST_ASSERT_EQUAL_INT32((w[i] - u[i]) * 32, got[i]);
    }
}

/* ---- exponent extremes -------------------------------------------------- */

/* Parameter with 4 exponent bits (bias 7, stored 0..15, E in [-7, 8]), update
 * with 8: an update far above the parameter's cap saturates every element
 * with the sign of -lr * u on stored 15; one below the floor flushes toward
 * zero on stored 0 with the exact SR probability (here 1/2). */
void testParamUpdateExponentCapAndFloor(void) {
    const int32_t w[3] = {5, -5, 0};
    const int32_t uBig[3] = {100, -100, 1};
    const uint8_t wStored = 10;
    const uint8_t uStoredBig = 127 + 60;
    puInit(&fixW, 3, 1, 0, 8, 4, w, &wStored);
    puInit(&fixU, 3, 1, 0, 8, 8, uBig, &uStoredBig);
    rngStream_t s = rngOpen((rngKey_t){{9u, 10u}});
    puRun(lrOf(0.05f), SR_HALF_AWAY, &s);
    int32_t got[3];
    puCodes(&fixW, got);
    TEST_ASSERT_EQUAL_UINT8(15, fixW.exps[0]);
    TEST_ASSERT_EQUAL_INT32(-128, got[0]);
    TEST_ASSERT_EQUAL_INT32(127, got[1]);
    TEST_ASSERT_EQUAL_INT32(-128, got[2]); /* 0.05 * 2^60 still saturates */

    /* the same cap with a SMALL left shift of the update term: lr = 1 (q =
     * 2^14, e = -14), u on 2^33, so k_u = 8 - (33 - 14) = -11. u = +-100
     * (term 100 * 2^14 * 2^11 >= 2^30) saturates on the saturation path;
     * u = 1 (2^25 on the grid) does not, and clamps through the rounding.
     * A saturation test gated at the raw-code threshold (k <= -15) would
     * shift 100 * 2^14 left by 11 bits past int32 here (sign flip). */
    const uint8_t uStoredMid = 127 + 33;
    puInit(&fixW, 3, 1, 0, 8, 4, w, &wStored);
    puInit(&fixU, 3, 1, 0, 8, 8, uBig, &uStoredMid);
    rngStream_t sMid = rngOpen((rngKey_t){{9u, 11u}});
    puRun(LR_ONE, SR_HALF_AWAY, &sMid);
    puCodes(&fixW, got);
    TEST_ASSERT_EQUAL_UINT8(15, fixW.exps[0]);
    TEST_ASSERT_EQUAL_INT32(-128, got[0]);
    TEST_ASSERT_EQUAL_INT32(127, got[1]);
    TEST_ASSERT_EQUAL_INT32(-128, got[2]);

    /* floor: w = 1 at stored 0 (2^-7), lr = 1, u = 2^-8 -> 1/2 of a step */
    const int32_t one[1] = {1};
    const uint8_t s0 = 0;
    const uint8_t uS = 127 - 8;
    const struct {
        uint32_t draw;
        int32_t expect;
    } cases[] = {{0x7FFFFFFFu, 1}, {0x80000000u, 0}, {1u, 1}, {0xFFFFFFFFu, 0}};
    for (size_t c = 0; c < 4; c++) {
        puInit(&fixW, 1, 1, 0, 8, 4, one, &s0);
        puInit(&fixU, 1, 1, 0, 8, 8, one, &uS);
        rngStream_t ps = rngStreamWithNextDraw(cases[c].draw);
        puRun(LR_ONE, SR_HALF_AWAY, &ps);
        int32_t c0;
        puCodes(&fixW, &c0);
        TEST_ASSERT_EQUAL_UINT8(0, fixW.exps[0]);
        TEST_ASSERT_EQUAL_INT32(cases[c].expect, c0);
    }
}

/* ---- in place: every group's exponent changes -------------------------- */

/* Groups of 100 straddling the 256-element chunks (update per-tensor and in
 * the parameter's geometry) and groups of 1 beyond the 257-slot rings, with
 * every fresh exponent different from the old one: lr * u = w / 2 exactly,
 * so every tight group drops one binade. A group whose new exponent were
 * published before its last element was read would mis-read that element. */
void testParamUpdateInPlaceKeepsOldExponentsUntilTheGroupIsRead(void) {
    static int32_t w[1000];
    static int32_t u[1000];
    static uint8_t ws[1000];
    static uint8_t us[1000];
    const puGeo_t geos[] = {{1000, 10, 100, 8, 8, 16, 8, true},
                            {1000, 10, 100, 8, 8, 8, 8, false},
                            {700, 700, 1, 8, 8, 8, 8, false}};
    for (size_t c = 0; c < sizeof geos / sizeof geos[0]; c++) {
        const size_t gsz = geos[c].groupSize;
        us[0] = 99; /* per-tensor: at or below every group's exponent - 1 */
        for (size_t gi = 0; gi < geos[c].numGroups; gi++) {
            ws[gi] = (uint8_t)(100 + (gi * 3) % 7);
            if (!geos[c].uPerTensor) {
                us[gi] = (uint8_t)(ws[gi] - 1);
            }
            const int sh = (int)ws[gi] - 1 - (int)us[geos[c].uPerTensor ? 0 : gi];
            for (size_t i = gi * gsz; i < (gi + 1) * gsz; i++) {
                w[i] = testRandInt(-120, 120);
                u[i] = w[i] * (1 << sh); /* lr = 1: lr * u = w / 2 */
            }
            w[gi * gsz] = 127;
            u[gi * gsz] = 127 * (1 << sh);
        }
        assertParamUpdateMatchesReference(&geos[c], w, ws, u, us, LR_ONE, SR_HALF_AWAY,
                                          0x4343u + (uint32_t)c);
        for (size_t gi = 0; gi < geos[c].numGroups; gi++) {
            TEST_ASSERT_EQUAL_UINT8(ws[gi] - 1, fixW.exps[gi]); /* every exponent moved */
        }
    }
}

void testParamUpdateEmptyTensorIsTheZeroState(void) {
    const uint8_t stored = 140;
    puInit(&fixW, 0, 1, 0, 8, 8, NULL, &stored);
    puInit(&fixU, 0, 1, 0, 8, 8, NULL, &stored);
    puRun(lrOf(0.05f), HALF_AWAY, NULL);
    TEST_ASSERT_EQUAL_UINT8(127, fixW.exps[0]);
}

/* ---- fail fast ---------------------------------------------------------- */

static const int32_t badCodes[4] = {1, 2, 3, 4};
static const uint8_t badStored[4] = {127, 127, 127, 127};

static void setupBad(void) {
    puInit(&fixW, 4, 1, 0, 8, 8, badCodes, badStored);
    puInit(&fixU, 4, 1, 0, 8, 8, badCodes, badStored);
}
static void badSrNoStream(void) {
    setupBad();
    puRun(lrOf(0.05f), SR_HALF_AWAY, NULL);
}
static void badLrNotNormalized(void) {
    setupBad();
    puRun((bfpScalar_t){.q = 100, .e = -10}, HALF_AWAY, NULL);
}
static void badLrTooWide(void) {
    setupBad();
    puRun((bfpScalar_t){.q = 32768, .e = -15}, HALF_AWAY, NULL);
}
static void badRoundingMode(void) {
    setupBad();
    puRun(lrOf(0.05f), (roundingMode_t)99, NULL);
}
static void badAlias(void) {
    setupBad();
    paramUpdateBfpInPlace(&fixW.t, &fixW.t, lrOf(0.05f), HALF_AWAY, NULL);
}
static void badUpdateGroupedFinerThanParam(void) {
    puInit(&fixW, 4, 1, 0, 8, 8, badCodes, badStored);
    puInit(&fixU, 4, 2, 2, 8, 8, badCodes, badStored);
    puRun(lrOf(0.05f), HALF_AWAY, NULL);
}
static void badUpdateOtherGrouping(void) {
    puInit(&fixW, 4, 2, 2, 8, 8, badCodes, badStored);
    puInit(&fixU, 4, 4, 1, 8, 8, badCodes, badStored);
    puRun(lrOf(0.05f), HALF_AWAY, NULL);
}
static void badCount(void) {
    puInit(&fixW, 4, 1, 0, 8, 8, badCodes, badStored);
    puInit(&fixU, 3, 1, 0, 8, 8, badCodes, badStored);
    puRun(lrOf(0.05f), HALF_AWAY, NULL);
}
static void badWide(void) {
    puInit(&fixW, 4, 1, 0, 8, 8, badCodes, badStored);
    puInit(&fixU, 4, 1, 0, 8, 8, badCodes, badStored);
    fixU.qc.mantissaBits = 17;
    puRun(lrOf(0.05f), HALF_AWAY, NULL);
}
static void badUpdateDtype(void) {
    setupBad();
    static float uf[4] = {1, 2, 3, 4};
    static quantization_t fq;
    initFloat32Quantization(&fq);
    static tensor_t ut;
    setTensorValues(&ut, (uint8_t *)uf, &fixU.shape, &fq, NULL);
    paramUpdateBfpInPlace(&fixW.t, &ut, lrOf(0.05f), HALF_AWAY, NULL);
}

/* Exponent storage overlap: the engine publishes each parameter group's new
 * exponent once that group is done, so an update whose (per-tensor) exponent
 * IS the parameter's first exponent byte would see its grid move under it
 * (params [100, 100] on groups E = [0, 0], update [50, 50], lr 1, HALF_AWAY:
 * [50, 63.5] instead of [50, 50]). Payload overlap: a one-byte shift. */
static void badExponentOverlap(void) {
    const int32_t w[2] = {100, 100};
    const int32_t u[2] = {50, 50};
    const uint8_t sw[2] = {127, 127};
    const uint8_t su[1] = {127};
    puInit(&fixW, 2, 2, 1, 8, 8, w, sw);
    puInit(&fixU, 2, 1, 0, 8, 8, u, su);
    fixU.qc.exponents = fixW.exps;
    puRun(LR_ONE, HALF_AWAY, NULL);
}
static void badPayloadOverlap(void) {
    setupBad();
    fixU.t.data = fixW.data + 1;
    puRun(lrOf(0.05f), HALF_AWAY, NULL);
}
/* lr.q = INT32_MIN: the update negates lr before the engine sees it, which
 * would be signed overflow; the significand is validated first. */
static void badLrIntMin(void) {
    setupBad();
    puRun((bfpScalar_t){.q = INT32_MIN, .e = -14}, HALF_AWAY, NULL);
}

void testParamUpdateRejectsOverlappingStorageAndAnUnnegatableLr(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "exponents overlap", badExponentOverlap());
    ASSERT_EXITS_WITH_OUTPUT(1, "payload overlaps", badPayloadOverlap());
    ASSERT_EXITS_WITH_OUTPUT(1, "paramUpdateBfpInPlace: lr {q=", badLrIntMin());
}

void testParamUpdateRejectsBadInputs(void) {
    ASSERT_EXITS_WITH_FAILURE(badSrNoStream());
    ASSERT_EXITS_WITH_FAILURE(badLrNotNormalized());
    ASSERT_EXITS_WITH_FAILURE(badLrTooWide());
    ASSERT_EXITS_WITH_FAILURE(badRoundingMode());
    ASSERT_EXITS_WITH_FAILURE(badAlias());
    ASSERT_EXITS_WITH_FAILURE(badUpdateGroupedFinerThanParam());
    ASSERT_EXITS_WITH_FAILURE(badUpdateOtherGrouping());
    ASSERT_EXITS_WITH_FAILURE(badCount());
    ASSERT_EXITS_WITH_FAILURE(badWide());
    ASSERT_EXITS_WITH_FAILURE(badUpdateDtype());
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testParamUpdateDeadZonePinned);
    RUN_TEST(testParamUpdateDeadZoneStatistical);
    RUN_TEST(testParamUpdateMatchesExactReferenceOnRandomParameters);
    RUN_TEST(testParamUpdateIsUnbiasedUnderSr);
    RUN_TEST(testParamUpdatePowerOfTwoLrIsExact);
    RUN_TEST(testParamUpdateZeroLrIsTheParameterExactly);
    RUN_TEST(testParamUpdateExactCancellationIsTheZeroState);
    RUN_TEST(testParamUpdateNearCancellationRegridsToTheResidue);
    RUN_TEST(testParamUpdateExponentCapAndFloor);
    RUN_TEST(testParamUpdateInPlaceKeepsOldExponentsUntilTheGroupIsRead);
    RUN_TEST(testParamUpdateEmptyTensorIsTheZeroState);
    RUN_TEST(testParamUpdateRejectsBadInputs);
    RUN_TEST(testParamUpdateRejectsOverlappingStorageAndAnUnnegatableLr);
    return UNITY_END();
}
