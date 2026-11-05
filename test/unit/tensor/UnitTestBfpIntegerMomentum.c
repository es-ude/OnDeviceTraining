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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The integer SGD momentum store v' = mu * v + g on BFP state and BFP
 * gradient (momentumStoreBfpInPlace, arithmetic-bfp.md §15.7). Every expected
 * value comes from the exact 128-bit reference below, written from the
 * documented result, not from the engine's quotient/remainder code. */

void setUp(void) {}
void tearDown(void) {}

static uint32_t testRngState = 0x5D1E2B97u;

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

#define MO_MAX_N 4096
#define MO_MAX_GROUPS 4096

typedef struct {
    size_t n;
    size_t dims[1];
    size_t order[1];
    shape_t shape;
    uint8_t exps[MO_MAX_GROUPS];
    bfpQConfig_t qc;
    quantization_t q;
    uint8_t data[MO_MAX_N * 2 + 8];
    tensor_t t;
} moFixture_t;

static moFixture_t fixV; /* the momentum state, updated in place */
static moFixture_t fixG; /* the gradient, read only */

static void moInit(moFixture_t *f, size_t n, size_t numGroups, size_t groupSize, uint8_t m,
                   uint8_t e, const int32_t *codes, const uint8_t *stored) {
    TEST_ASSERT_TRUE(n <= MO_MAX_N && numGroups <= MO_MAX_GROUPS);
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
        static int32_t tmp[MO_MAX_N];
        memcpy(tmp, codes, n * sizeof(int32_t));
        byteConversion((uint8_t *)tmp, 32, f->data, m, n);
    }
    setTensorValues(&f->t, f->data, &f->shape, &f->q, NULL);
}

static void moCodes(const moFixture_t *f, int32_t *out) {
    unpackSignExtend(f->data, f->qc.mantissaBits, 0, out, f->n);
}

static void moRun(bfpScalar_t mu, roundingMode_t rm, rngStream_t *rs) {
    momentumStoreBfpInPlace(&fixV.t, &fixG.t, mu, rm, rs);
}

static bfpScalar_t muOf(float x) {
    return bfpScalarFromFloat(x, "test mu");
}

static const bfpScalar_t MU_ONE = {.q = 16384, .e = -14};
static const bfpScalar_t MU_ZERO = {.q = 0, .e = 0};

/* ---- the exact reference ----------------------------------------------
 * Per destination group, with E_v, E_g the unbiased stored exponents of the
 * state's and the gradient's group and the scalar mu = q * 2^e:
 *   terms   v_i * q on 2^(E_v + e) and g_i on 2^E_g (exact);
 *   E_c   = max(E_v + e + bitlen(q * A_v), E_g + bitlen(A_g)) - 30, capped
 *           at the state's capE (a zero side has no top);
 *   E_d   = the smallest E >= E_c with max|sum| <= qMax * 2^E, clamped to
 *           [-bias, capE]; an all-zero sum is the zero state;
 *   code  = |sum| / 2^E_d rounded once (F = the top 32 remainder bits; SR
 *           lifts iff u < F, HALF_AWAY iff F >= 2^31), sign after, clamped;
 *   draws = one per element in element order under SR, on every path. */
static void refMomentum(const int32_t *v, const uint8_t *vStored, uint8_t mv, uint8_t ev,
                        const int32_t *g, const uint8_t *gStored, uint8_t eg, size_t n,
                        size_t numGroups, size_t groupSize, bfpScalar_t mu, roundingMode_t rm,
                        rngStream_t *rs, int32_t *outCodes, uint8_t *outStored) {
    const int biasV = (1 << (ev - 1)) - 1;
    const int biasG = (1 << (eg - 1)) - 1;
    int cap = (1 << ev) - 1;
    if (cap > biasV + 127) {
        cap = biasV + 127;
    }
    const int capE = cap - biasV;
    const uint32_t posCap = (1u << (mv - 1)) - 1u;
    const uint32_t negCap = 1u << (mv - 1);
    const size_t gsz = groupSize == 0 ? n : groupSize;
    const bool sr = rm == SR_HALF_AWAY;
    const int64_t q = mu.q;
    for (size_t gi = 0; gi < numGroups; gi++) {
        const size_t start = gi * gsz;
        const size_t end = start + gsz;
        const int eV = (int)vStored[gi] - biasV + mu.e;
        const int eG = (int)gStored[gi] - biasG;
        uint64_t av = 0;
        uint64_t ag = 0;
        for (size_t i = start; i < end; i++) {
            int64_t pv = (int64_t)v[i] * q;
            uint64_t mvq = pv < 0 ? (uint64_t)(-pv) : (uint64_t)pv;
            uint64_t mg = g[i] < 0 ? (uint64_t)(-(int64_t)g[i]) : (uint64_t)g[i];
            av = mvq > av ? mvq : av;
            ag = mg > ag ? mg : ag;
        }
        static bfpAccRefI128_t vals[MO_MAX_N];
        bfpAccRefU128_t vmax = 0;
        int ec = 0;
        int emin = 0;
        if (av != 0 || ag != 0) {
            const int topV = av ? eV + bfpAccRefBitLen(av) : INT32_MIN;
            const int topG = ag ? eG + bfpAccRefBitLen(ag) : INT32_MIN;
            ec = (topV > topG ? topV : topG) - 30;
            ec = ec > capE ? capE : ec;
            emin = av == 0 ? eG : (ag == 0 ? eV : (eV < eG ? eV : eG));
            TEST_ASSERT_TRUE_MESSAGE((av == 0 || eV - emin <= 90) && (ag == 0 || eG - emin <= 90),
                                     "refMomentum: exponent gap outside the reference");
            for (size_t i = start; i < end; i++) {
                bfpAccRefI128_t tv = av == 0 ? 0
                                             : (bfpAccRefI128_t)((int64_t)v[i] * q) *
                                                   ((bfpAccRefI128_t)1 << (eV - emin));
                bfpAccRefI128_t tg =
                    ag == 0 ? 0 : (bfpAccRefI128_t)g[i] * ((bfpAccRefI128_t)1 << (eG - emin));
                vals[i - start] = tv + tg;
                bfpAccRefU128_t a = bfpAccRefAbs(tv + tg);
                vmax = a > vmax ? a : vmax;
            }
        }
        int ed = 0;
        if (vmax == 0) {
            outStored[gi] = (uint8_t)biasV;
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
            E = E < -biasV ? -biasV : E;
            E = E > capE ? capE : E;
            outStored[gi] = (uint8_t)(E + biasV);
            ed = E;
        }
        for (size_t i = start; i < end; i++) {
            const uint32_t u = sr ? rngNextU32(rs) : 0u;
            outCodes[i] =
                vmax == 0 ? 0 : bfpAccRefRound(vals[i - start], ed - emin, sr, u, posCap, negCap);
        }
    }
}

typedef struct {
    size_t n, numGroups, groupSize;
    uint8_t mv, ev, mg, eg;
} moGeo_t;

/* Runs engine and reference from the same seed; asserts codes, exponents,
 * the gradient untouched, and (SR) the stream position: exactly n draws. */
static void assertMomentumMatchesReference(const moGeo_t *geo, const int32_t *v, const uint8_t *vs,
                                           const int32_t *g, const uint8_t *gs, bfpScalar_t mu,
                                           roundingMode_t rm, uint32_t seed) {
    static int32_t expCodes[MO_MAX_N];
    static int32_t gotCodes[MO_MAX_N];
    static int32_t gAfter[MO_MAX_N];
    static uint8_t expStored[MO_MAX_GROUPS];
    static uint8_t gDataBefore[sizeof fixG.data];
    const size_t n = geo->n;
    rngStream_t refStream = rngOpen((rngKey_t){{seed, 21u}});
    rngStream_t engStream = rngOpen((rngKey_t){{seed, 21u}});
    refMomentum(v, vs, geo->mv, geo->ev, g, gs, geo->eg, n, geo->numGroups, geo->groupSize, mu, rm,
                &refStream, expCodes, expStored);
    moInit(&fixV, n, geo->numGroups, geo->groupSize, geo->mv, geo->ev, v, vs);
    moInit(&fixG, n, geo->numGroups, geo->groupSize, geo->mg, geo->eg, g, gs);
    memcpy(gDataBefore, fixG.data, sizeof fixG.data);
    moRun(mu, rm, rm == SR_HALF_AWAY ? &engStream : NULL);
    moCodes(&fixV, gotCodes);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expStored, fixV.exps, geo->numGroups);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expCodes, gotCodes, n);
    TEST_ASSERT_EQUAL_MEMORY(gDataBefore, fixG.data, sizeof fixG.data);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(gs, fixG.exps, geo->numGroups);
    moCodes(&fixG, gAfter);
    TEST_ASSERT_EQUAL_INT32_ARRAY(g, gAfter, n);
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
    const int32_t lim = flavour == 2 ? (int32_t)((1u << testRandInt(0, m - 2)) - 0u) : qMax;
    for (size_t i = 0; i < len; i++) {
        codes[i] = flavour == 0 ? 0 : testRandInt(-lim, lim);
    }
    if (flavour == 3 && len > 0) {
        codes[0] = -(1 << (m - 1));
    }
}

/* Random states and gradients against the exact reference, over per-tensor
 * and grouped geometries (group size 1 with more than 257 groups wraps the
 * window rings; groups straddle the 256-element chunk boundaries, so a fresh
 * exponent published before its group's last read shows), byte, sub-byte and
 * 16-bit widths, state and gradient in DIFFERENT widths and exponent ranges
 * (narrow state exponents reach the cap and the floor), zero sides, the
 * negative code bound, cancelling groups (g = -mu * v on the gradient's grid),
 * every relative exponent offset in [-24, 24], and mu in {0, 1, powers of
 * two, 0.9, 0.99, ...}; HALF_AWAY and SR. */
void testMomentumMatchesExactReferenceOnRandomStates(void) {
    static int32_t v[MO_MAX_N];
    static int32_t g[MO_MAX_N];
    static uint8_t vs[MO_MAX_GROUPS];
    static uint8_t gs[MO_MAX_GROUPS];
    const moGeo_t geos[] = {
        {1, 1, 0, 8, 8, 8, 8},       {6, 1, 0, 8, 8, 8, 8},       {600, 1, 0, 8, 8, 8, 8},
        {4096, 1, 0, 8, 8, 8, 8},    {700, 700, 1, 8, 8, 8, 8},   {699, 233, 3, 5, 8, 8, 8},
        {1000, 10, 100, 8, 8, 6, 7}, {640, 10, 64, 16, 8, 16, 8}, {513, 1, 0, 2, 8, 8, 8},
        {770, 110, 7, 12, 8, 4, 8},  {300, 3, 100, 6, 4, 8, 8},   {32, 2, 16, 8, 3, 8, 6},
        {2048, 2048, 1, 8, 4, 8, 8}, {1024, 4, 256, 16, 8, 8, 8}, {999, 333, 3, 7, 6, 3, 5},
        {257, 257, 1, 8, 8, 16, 8},  {511, 1, 0, 16, 5, 16, 8},
    };
    const float mus[] = {0.0f,   1.0f,   0.5f,      0.25f,      0.9f,  0.99f,
                         0.875f, 0.999f, 0.123456f, 0.0000011f, 0.75f, 0.6180339f};
    for (int round = 0; round < 3; round++) {
        for (size_t c = 0; c < sizeof geos / sizeof geos[0]; c++) {
            const moGeo_t *geo = &geos[c];
            const size_t n = geo->n;
            const size_t gsz = geo->groupSize == 0 ? n : geo->groupSize;
            const int biasV = (1 << (geo->ev - 1)) - 1;
            const int biasG = (1 << (geo->eg - 1)) - 1;
            const int maxV = (1 << geo->ev) - 1 > biasV + 127 ? biasV + 127 : (1 << geo->ev) - 1;
            const int maxG = (1 << geo->eg) - 1 > biasG + 127 ? biasG + 127 : (1 << geo->eg) - 1;
            for (size_t k = 0; k < sizeof mus / sizeof mus[0]; k++) {
                const bfpScalar_t mu = muOf(mus[k]);
                for (size_t gi = 0; gi < geo->numGroups; gi++) {
                    const int flavourV = testRandInt(0, 9);
                    const int flavourG = testRandInt(0, 9);
                    /* a third of the state groups near the exponent floor */
                    vs[gi] = (uint8_t)(flavourV < 3 ? testRandInt(0, maxV < 12 ? maxV : 12)
                                                    : testRandInt(0, maxV));
                    const int eV = (int)vs[gi] - biasV;
                    gs[gi] = (uint8_t)clampInt(eV + testRandInt(-24, 24) + biasG, 0, maxG);
                    int32_t *vg = &v[gi * gsz];
                    int32_t *gg = &g[gi * gsz];
                    randomGroupCodes(vg, gsz, geo->mv, flavourV == 0 ? 0 : 1 + flavourV % 3);
                    randomGroupCodes(gg, gsz, geo->mg, flavourG == 0 ? 0 : 1 + flavourG % 3);
                    if (flavourG == 9 && mu.q != 0) {
                        /* cancelling: g = -round(mu * v) on the gradient's grid
                         * at the exponent mu * v naturally lands on */
                        const int eTarget = eV + mu.e + 30 - (geo->mg - 1);
                        gs[gi] = (uint8_t)clampInt(eTarget + biasG, 0, maxG);
                        const int sh = ((int)gs[gi] - biasG) - (eV + mu.e);
                        const int32_t gMax = (1 << (geo->mg - 1)) - 1;
                        for (size_t i = 0; i < gsz; i++) {
                            int64_t p = (int64_t)vg[i] * mu.q;
                            int64_t r = sh >= 0 ? (sh >= 62 ? 0 : p / ((int64_t)1 << sh))
                                                : p * ((int64_t)1 << (-sh > 30 ? 30 : -sh));
                            r = r > gMax ? gMax : (r < -gMax ? -gMax : r);
                            gg[i] = (int32_t)-r;
                        }
                    }
                    /* the reference's exact domain: a gradient clamped far
                     * from mu * v (narrow gradient exponents) is a zero side */
                    const int gap = ((int)gs[gi] - biasG) - (eV + mu.e);
                    if (gap > 80 || gap < -80) {
                        memset(gg, 0, gsz * sizeof(int32_t));
                    }
                }
                const uint32_t seed = 0x8000u + (uint32_t)(round * 7919 + (int)c * 131 + (int)k);
                assertMomentumMatchesReference(geo, v, vs, g, gs, mu, HALF_AWAY, seed);
                assertMomentumMatchesReference(geo, v, vs, g, gs, mu, SR_HALF_AWAY, seed);
            }
        }
    }
}

/* A per-tensor gradient on a GROUPED state (HAR's per-tensor BFP grads under
 * per-channel state would look like this): every state group reads the
 * gradient under its one exponent. The reference sees that exponent
 * replicated per state group, which is the same value; the engine sees the
 * per-tensor {1, 0} gradient. Groups straddle the 256-element chunks; the
 * state exponents spread so some groups sit above and some below the
 * gradient's, including the cap and floor of a narrow state exponent. */
static void assertMomentumPerTensorGradMatchesReference(const moGeo_t *geo, const int32_t *v,
                                                        const uint8_t *vs, const int32_t *g,
                                                        uint8_t gStored, bfpScalar_t mu,
                                                        roundingMode_t rm, uint32_t seed) {
    static int32_t expCodes[MO_MAX_N];
    static int32_t gotCodes[MO_MAX_N];
    static int32_t gAfter[MO_MAX_N];
    static uint8_t expStored[MO_MAX_GROUPS];
    static uint8_t gsRep[MO_MAX_GROUPS];
    static uint8_t gDataBefore[sizeof fixG.data];
    const size_t n = geo->n;
    memset(gsRep, gStored, geo->numGroups);
    rngStream_t refStream = rngOpen((rngKey_t){{seed, 23u}});
    rngStream_t engStream = rngOpen((rngKey_t){{seed, 23u}});
    refMomentum(v, vs, geo->mv, geo->ev, g, gsRep, geo->eg, n, geo->numGroups, geo->groupSize, mu,
                rm, &refStream, expCodes, expStored);
    moInit(&fixV, n, geo->numGroups, geo->groupSize, geo->mv, geo->ev, v, vs);
    moInit(&fixG, n, 1, 0, geo->mg, geo->eg, g, &gStored);
    memcpy(gDataBefore, fixG.data, sizeof fixG.data);
    moRun(mu, rm, rm == SR_HALF_AWAY ? &engStream : NULL);
    moCodes(&fixV, gotCodes);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expStored, fixV.exps, geo->numGroups);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expCodes, gotCodes, n);
    TEST_ASSERT_EQUAL_MEMORY(gDataBefore, fixG.data, sizeof fixG.data);
    TEST_ASSERT_EQUAL_UINT8(gStored, fixG.exps[0]);
    moCodes(&fixG, gAfter);
    TEST_ASSERT_EQUAL_INT32_ARRAY(g, gAfter, n);
    if (rm == SR_HALF_AWAY) {
        TEST_ASSERT_EQUAL_UINT32(rngNextU32(&refStream), rngNextU32(&engStream));
    }
}

void testMomentumPerTensorGradientOnGroupedStateMatchesReference(void) {
    static int32_t v[MO_MAX_N];
    static int32_t g[MO_MAX_N];
    static uint8_t vs[MO_MAX_GROUPS];
    const moGeo_t geos[] = {
        {4032, 42, 96, 8, 8, 8, 8}, /* HAR conv3 channel length 96 */
        {1000, 10, 100, 8, 8, 6, 7},
        {700, 700, 1, 8, 8, 8, 8},
        {300, 3, 100, 6, 4, 8, 8},
    };
    const float mus[] = {0.9f, 1.0f, 0.5f, 0.0f, 0.6180339f};
    int distinctGroupEdges = 0;
    for (size_t c = 0; c < sizeof geos / sizeof geos[0]; c++) {
        const moGeo_t *geo = &geos[c];
        const size_t n = geo->n;
        const size_t gsz = geo->groupSize;
        const int biasV = (1 << (geo->ev - 1)) - 1;
        const int biasG = (1 << (geo->eg - 1)) - 1;
        const int maxV = (1 << geo->ev) - 1 > biasV + 127 ? biasV + 127 : (1 << geo->ev) - 1;
        const uint8_t gStored = (uint8_t)(biasG - 6);
        randomGroupCodes(g, n, geo->mg, 1);
        for (size_t k = 0; k < sizeof mus / sizeof mus[0]; k++) {
            for (size_t gi = 0; gi < geo->numGroups; gi++) {
                /* state groups 0..20 binades around the gradient's exponent */
                const int eV = clampInt(-6 + testRandInt(-10, 10) + biasV, 0, maxV);
                vs[gi] = (uint8_t)eV;
                distinctGroupEdges += gi > 0 && vs[gi] != vs[gi - 1];
                randomGroupCodes(&v[gi * gsz], gsz, geo->mv, testRandInt(0, 9) == 0 ? 0 : 1);
            }
            const uint32_t seed = 0x9100u + (uint32_t)((int)c * 37 + (int)k);
            assertMomentumPerTensorGradMatchesReference(geo, v, vs, g, gStored, muOf(mus[k]),
                                                        HALF_AWAY, seed);
            assertMomentumPerTensorGradMatchesReference(geo, v, vs, g, gStored, muOf(mus[k]),
                                                        SR_HALF_AWAY, seed);
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(distinctGroupEdges > 100, "state exponents must differ by group");
}

/* ---- dead zone ---------------------------------------------------------
 * mu = 1 makes mu * v exact, so the only rounding is the final SR: v = 64
 * code steps, g = -1/256 of a step (a grad code of -1, eight binades lower).
 * v' = 63 + 255/256 stays on the grid (> qMax / 2); SR lifts back to 64 iff
 * u < 255 * 2^24, so 63 appears iff u >= 255 * 2^24 -- the small update
 * survives with probability 1/256 instead of being erased. */
void testMomentumDeadZonePinned(void) {
    const uint8_t stored = 130;
    const uint8_t gStored = stored - 8;
    const int32_t v64[1] = {64};
    const int32_t gMinus[1] = {-1};
    const int32_t gPlus[1] = {1};
    const struct {
        const int32_t *g;
        uint32_t u;
        int32_t expect;
    } cases[] = {
        {gMinus, 255u << 24, 63}, {gMinus, (255u << 24) - 1u, 64}, {gMinus, 0xFFFFFFFFu, 63},
        {gMinus, 1u, 64},         {gPlus, (1u << 24) - 1u, 65},    {gPlus, 1u << 24, 64},
        {gPlus, 0xFFFFFFFFu, 64},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        moInit(&fixV, 1, 1, 0, 8, 8, v64, &stored);
        moInit(&fixG, 1, 1, 0, 8, 8, cases[c].g, &gStored);
        rngStream_t s = rngStreamWithNextDraw(cases[c].u);
        moRun(MU_ONE, SR_HALF_AWAY, &s);
        int32_t got;
        moCodes(&fixV, &got);
        TEST_ASSERT_EQUAL_UINT8(stored, fixV.exps[0]);
        TEST_ASSERT_EQUAL_INT32(cases[c].expect, got);
    }
}

/* The same update over 102400 independent draws: the code moves to 63 about
 * 400 times (binomial, p = 1/256; 5 sigma ~ 100), never zero times. */
void testMomentumDeadZoneStatistical(void) {
    static int32_t v[1024];
    static int32_t g[1024];
    static int32_t got[1024];
    const uint8_t stored = 130;
    const uint8_t gStored = stored - 8;
    for (size_t i = 0; i < 1024; i++) {
        v[i] = 64;
        g[i] = -1;
    }
    size_t moves = 0;
    rngStream_t s = rngOpen((rngKey_t){{0xC0FFEEu, 3u}});
    for (int rep = 0; rep < 100; rep++) {
        moInit(&fixV, 1024, 1, 0, 8, 8, v, &stored);
        moInit(&fixG, 1024, 1, 0, 8, 8, g, &gStored);
        moRun(MU_ONE, SR_HALF_AWAY, &s);
        moCodes(&fixV, got);
        TEST_ASSERT_EQUAL_UINT8(stored, fixV.exps[0]);
        for (size_t i = 0; i < 1024; i++) {
            TEST_ASSERT_TRUE(got[i] == 63 || got[i] == 64);
            moves += got[i] == 63;
        }
    }
    TEST_ASSERT_TRUE(moves >= 300 && moves <= 500);
}

/* mu != 1: the mu * v term itself has a tail on every element, carried to
 * the one final SR -- E[v'] equals the exact mu_enc * v + g (5 sigma). */
void testMomentumIsUnbiasedUnderSr(void) {
    static int32_t v[1024];
    static int32_t g[1024];
    static int32_t got[1024];
    const uint8_t stored = 128;
    const bfpScalar_t mu = muOf(0.9f);
    const int32_t vCodes[] = {100, -77, 3, -1};
    const int32_t gCodes[] = {-5, 9, 0, 1};
    for (size_t c = 0; c < 4; c++) {
        for (size_t i = 0; i < 1024; i++) {
            v[i] = vCodes[c];
            g[i] = gCodes[c];
        }
        v[0] = 127; /* pins the group's grid: v' exponent stays 128 */
        g[0] = 0;
        const double exact = (double)vCodes[c] * mu.q * 0x1p-15 + (double)gCodes[c] * 0.0625;
        const uint8_t gStored = stored - 4;
        rngStream_t s = rngOpen((rngKey_t){{0xBEEF00u + (uint32_t)c, 5u}});
        double sum = 0.0;
        size_t cnt = 0;
        for (int rep = 0; rep < 60; rep++) {
            moInit(&fixV, 1024, 1, 0, 8, 8, v, &stored);
            moInit(&fixG, 1024, 1, 0, 8, 8, g, &gStored);
            moRun(mu, SR_HALF_AWAY, &s);
            TEST_ASSERT_EQUAL_UINT8(stored, fixV.exps[0]);
            moCodes(&fixV, got);
            for (size_t i = 1; i < 1024; i++) {
                sum += got[i];
                cnt++;
            }
        }
        const double mean = sum / (double)cnt;
        /* per-draw sd <= 0.5 */
        TEST_ASSERT_TRUE(mean > exact - 5 * 0.5 / 248.0 && mean < exact + 5 * 0.5 / 248.0);
    }
}

/* ---- mu = 0 and mu = 1 are exact -------------------------------------- */

/* mu = 0: v' is g re-gridded onto the state's format -- the old state is
 * ignored (nonzero garbage here), a tight gradient keeps codes and exponent,
 * a loose one is renormalized by an exact left shift; no rounding ever
 * happens, whatever the draws, yet n draws are consumed. */
void testMomentumMuZeroIsTheGradientExactly(void) {
    static int32_t v[300];
    static int32_t g[300];
    static int32_t got[300];
    for (size_t i = 0; i < 300; i++) {
        v[i] = testRandInt(-127, 127);
        g[i] = testRandInt(-127, 127);
    }
    g[17] = 127;
    const uint8_t vStored = 140;
    const uint8_t gStored = 101;
    for (int sr = 0; sr < 2; sr++) {
        moInit(&fixV, 300, 1, 0, 8, 8, v, &vStored);
        moInit(&fixG, 300, 1, 0, 8, 8, g, &gStored);
        rngStream_t s = rngOpen((rngKey_t){{77u, 1u}});
        rngStream_t probe = rngOpen((rngKey_t){{77u, 1u}});
        moRun(MU_ZERO, sr ? SR_HALF_AWAY : HALF_AWAY, sr ? &s : NULL);
        moCodes(&fixV, got);
        TEST_ASSERT_EQUAL_UINT8(gStored, fixV.exps[0]);
        TEST_ASSERT_EQUAL_INT32_ARRAY(g, got, 300);
        if (sr) {
            for (int i = 0; i < 300; i++) {
                (void)rngNextU32(&probe);
            }
            TEST_ASSERT_EQUAL_UINT32(rngNextU32(&probe), rngNextU32(&s));
        }
    }
    /* loose gradient (absmax 20): exponent drops by 2, codes x4, exact */
    for (size_t i = 0; i < 300; i++) {
        g[i] = testRandInt(-20, 20);
    }
    g[3] = 20;
    moInit(&fixV, 300, 1, 0, 8, 8, v, &vStored);
    moInit(&fixG, 300, 1, 0, 8, 8, g, &gStored);
    rngStream_t s2 = rngOpen((rngKey_t){{78u, 1u}});
    moRun(MU_ZERO, SR_HALF_AWAY, &s2);
    moCodes(&fixV, got);
    TEST_ASSERT_EQUAL_UINT8(gStored - 2, fixV.exps[0]);
    for (size_t i = 0; i < 300; i++) {
        TEST_ASSERT_EQUAL_INT32(4 * g[i], got[i]);
    }
}

/* mu = 1: v' = v + g exactly when the sum is representable on the state's
 * grid -- here every sum lies in [-127, 127] with absmax in [64, 127], so the
 * exponent stays and no element rounds under any draw. */
void testMomentumMuOneIsExactAddition(void) {
    static int32_t v[500];
    static int32_t g[500];
    static int32_t got[500];
    for (size_t i = 0; i < 500; i++) {
        v[i] = testRandInt(-63, 63);
        g[i] = testRandInt(-64, 64);
    }
    v[250] = 63;
    g[250] = 64;
    const uint8_t stored = 120;
    for (int sr = 0; sr < 2; sr++) {
        moInit(&fixV, 500, 1, 0, 8, 8, v, &stored);
        moInit(&fixG, 500, 1, 0, 8, 8, g, &stored);
        rngStream_t s = rngOpen((rngKey_t){{99u, 2u}});
        moRun(MU_ONE, sr ? SR_HALF_AWAY : HALF_AWAY, sr ? &s : NULL);
        moCodes(&fixV, got);
        TEST_ASSERT_EQUAL_UINT8(stored, fixV.exps[0]);
        for (size_t i = 0; i < 500; i++) {
            TEST_ASSERT_EQUAL_INT32(v[i] + g[i], got[i]);
        }
    }
}

/* ---- cancellation ------------------------------------------------------ */

/* v = -g exactly (mu = 1): every sum is zero -> the zero state (stored =
 * bias, codes 0), and under SR the n draws are still consumed. */
void testMomentumExactCancellationIsTheZeroState(void) {
    static int32_t v[400];
    static int32_t g[400];
    static int32_t got[400];
    for (size_t i = 0; i < 400; i++) {
        v[i] = testRandInt(-127, 127);
        g[i] = -v[i];
    }
    const uint8_t stored = 133;
    moInit(&fixV, 400, 1, 0, 8, 8, v, &stored);
    moInit(&fixG, 400, 1, 0, 8, 8, g, &stored);
    rngStream_t s = rngOpen((rngKey_t){{5u, 5u}});
    rngStream_t probe = rngOpen((rngKey_t){{5u, 5u}});
    moRun(MU_ONE, SR_HALF_AWAY, &s);
    moCodes(&fixV, got);
    TEST_ASSERT_EQUAL_UINT8(127, fixV.exps[0]);
    for (size_t i = 0; i < 400; i++) {
        TEST_ASSERT_EQUAL_INT32(0, got[i]);
        (void)rngNextU32(&probe);
    }
    TEST_ASSERT_EQUAL_UINT32(rngNextU32(&probe), rngNextU32(&s));
}

/* v ~ -g (mu = 1): the residues (|r| <= 3 code steps) are exact on a grid 5
 * binades finer -- the exponent comes from the SUM, not from v or g. */
void testMomentumNearCancellationRegridsToTheResidue(void) {
    static int32_t v[400];
    static int32_t g[400];
    static int32_t got[400];
    for (size_t i = 0; i < 400; i++) {
        v[i] = testRandInt(-120, 120);
        g[i] = -v[i] + testRandInt(-3, 3);
    }
    v[9] = 100;
    g[9] = -97; /* residue 3: absmax 3 -> 3 * 2^5 = 96 <= 127 */
    const uint8_t stored = 133;
    moInit(&fixV, 400, 1, 0, 8, 8, v, &stored);
    moInit(&fixG, 400, 1, 0, 8, 8, g, &stored);
    rngStream_t s = rngOpen((rngKey_t){{6u, 5u}});
    moRun(MU_ONE, SR_HALF_AWAY, &s);
    moCodes(&fixV, got);
    TEST_ASSERT_EQUAL_UINT8(stored - 5, fixV.exps[0]);
    for (size_t i = 0; i < 400; i++) {
        TEST_ASSERT_EQUAL_INT32((v[i] + g[i]) * 32, got[i]);
    }
}

/* Cancellation below the working grid: 16-bit state v and mu = q * 2^-15 with
 * v * q = 1 (mod 2^15), gradient g = -(v * q >> 15) on the state's own
 * exponent. The exact sum is ONE unit of 2^(E_v - 15) = 2^E_c, so the
 * exponent rule (which would pick E_c - 14) is floored at E_c: code 1 on the
 * stored exponent E_c + bias, exact. */
void testMomentumCancellationBelowTheWorkingGridKeepsEc(void) {
    int32_t q = 0;
    int32_t v0 = 0;
    for (int32_t cand = 32767; cand >= 16385 && v0 == 0; cand -= 2) {
        uint32_t inv = 1; /* Newton: inverse of an odd cand mod 2^32 */
        for (int it = 0; it < 5; it++) {
            inv *= 2u - (uint32_t)cand * inv;
        }
        const int32_t candV = (int32_t)(inv & 0x7FFFu);
        if (candV >= 16384 && (int64_t)candV * cand >= ((int64_t)1 << 29)) {
            q = cand;
            v0 = candV;
        }
    }
    TEST_ASSERT_TRUE(v0 >= 16384);
    const int64_t p = (int64_t)v0 * q;
    TEST_ASSERT_EQUAL_INT64(1, p & 0x7FFF);
    const int32_t vCodes[2] = {v0, 0};
    const int32_t gCodes[2] = {(int32_t)-(p >> 15), 0};
    const uint8_t stored = 150; /* E_v = 23, E_c = 8 */
    const bfpScalar_t mu = {.q = q, .e = -15};
    for (int sr = 0; sr < 2; sr++) {
        moInit(&fixV, 2, 1, 0, 16, 8, vCodes, &stored);
        moInit(&fixG, 2, 1, 0, 16, 8, gCodes, &stored);
        rngStream_t s = rngOpen((rngKey_t){{8u, 8u}});
        moRun(mu, sr ? SR_HALF_AWAY : HALF_AWAY, sr ? &s : NULL);
        int32_t got[2];
        moCodes(&fixV, got);
        TEST_ASSERT_EQUAL_UINT8(stored - 15, fixV.exps[0]);
        TEST_ASSERT_EQUAL_INT32(1, got[0]);
        TEST_ASSERT_EQUAL_INT32(0, got[1]);
    }
}

/* ---- exponent extremes -------------------------------------------------- */

/* State with 4 exponent bits (bias 7, stored 0..15, E in [-7, 8]), gradient
 * with 8: a gradient far above the state's cap saturates every element with
 * the gradient's sign on stored 15; one far below the floor flushes toward
 * zero on stored 0 with the exact SR probability (here 1/2). */
void testMomentumExponentCapAndFloor(void) {
    const int32_t v[3] = {5, -5, 0};
    const int32_t gBig[3] = {100, -100, 1};
    const uint8_t vStored = 10;
    const uint8_t gStoredBig = 127 + 60;
    moInit(&fixV, 3, 1, 0, 8, 4, v, &vStored);
    moInit(&fixG, 3, 1, 0, 8, 8, gBig, &gStoredBig);
    rngStream_t s = rngOpen((rngKey_t){{9u, 9u}});
    moRun(muOf(0.9f), SR_HALF_AWAY, &s);
    int32_t got[3];
    moCodes(&fixV, got);
    TEST_ASSERT_EQUAL_UINT8(15, fixV.exps[0]);
    TEST_ASSERT_EQUAL_INT32(127, got[0]);
    TEST_ASSERT_EQUAL_INT32(-128, got[1]);
    TEST_ASSERT_EQUAL_INT32(127, got[2]); /* 1 * 2^60 still saturates */

    /* floor: v = 1 at stored 0 (2^-7), mu = 1/2, g = 0 -> 1/2 of a step */
    const int32_t one[1] = {1};
    const int32_t zero[1] = {0};
    const uint8_t s0 = 0;
    const uint8_t gS = 100;
    const struct {
        uint32_t u;
        int32_t expect;
    } cases[] = {{0x7FFFFFFFu, 1}, {0x80000000u, 0}, {1u, 1}, {0xFFFFFFFFu, 0}};
    for (size_t c = 0; c < 4; c++) {
        moInit(&fixV, 1, 1, 0, 8, 4, one, &s0);
        moInit(&fixG, 1, 1, 0, 8, 8, zero, &gS);
        rngStream_t ps = rngStreamWithNextDraw(cases[c].u);
        moRun(muOf(0.5f), SR_HALF_AWAY, &ps);
        int32_t c0;
        moCodes(&fixV, &c0);
        TEST_ASSERT_EQUAL_UINT8(0, fixV.exps[0]);
        TEST_ASSERT_EQUAL_INT32(cases[c].expect, c0);
    }
}

/* ---- in place: every group's exponent changes -------------------------- */

/* Groups of 100 straddling the 256-element chunks and groups of 1 beyond the
 * 257-slot rings, with every fresh exponent different from the old one (mu =
 * 1/2 drops every tight group one binade): a group whose new exponent were
 * published before its last element was read would mis-read that element. */
void testMomentumInPlaceKeepsOldExponentsUntilTheGroupIsRead(void) {
    static int32_t v[1000];
    static int32_t g[1000];
    static uint8_t vs[1000];
    static uint8_t gs[1000];
    const moGeo_t geos[] = {{1000, 10, 100, 8, 8, 8, 8}, {700, 700, 1, 8, 8, 8, 8}};
    for (size_t c = 0; c < 2; c++) {
        const size_t gsz = geos[c].groupSize;
        for (size_t gi = 0; gi < geos[c].numGroups; gi++) {
            vs[gi] = (uint8_t)(100 + (gi * 7) % 50);
            gs[gi] = (uint8_t)(vs[gi] - 10);
            for (size_t i = gi * gsz; i < (gi + 1) * gsz; i++) {
                v[i] = testRandInt(-120, 120);
                g[i] = testRandInt(-127, 127);
            }
            v[gi * gsz] = 127; /* v' = 63.5 - 127/1024 < 63.5: one binade down */
            g[gi * gsz] = -127;
        }
        assertMomentumMatchesReference(&geos[c], v, vs, g, gs, muOf(0.5f), SR_HALF_AWAY,
                                       0x4242u + (uint32_t)c);
        for (size_t gi = 0; gi < geos[c].numGroups; gi++) {
            TEST_ASSERT_NOT_EQUAL_UINT8(vs[gi], fixV.exps[gi]); /* every exponent moved */
        }
    }
}

void testMomentumEmptyTensorIsTheZeroState(void) {
    const uint8_t stored = 140;
    moInit(&fixV, 0, 1, 0, 8, 8, NULL, &stored);
    moInit(&fixG, 0, 1, 0, 8, 8, NULL, &stored);
    moRun(muOf(0.9f), HALF_AWAY, NULL);
    TEST_ASSERT_EQUAL_UINT8(127, fixV.exps[0]);
}

/* ---- the checked sticky rule of bfpQrAdd -------------------------------- */

static void twoStickyBody(void) {
    (void)bfpQrAdd(bfpQrOfTerm(1, 40), bfpQrOfTerm(3, 41));
}

void testQrAddRejectsTwoStickyOperands(void) {
    /* one sticky operand is exact and fine */
    bfpQr_t r = bfpQrAdd(bfpQrOfTerm(1, 40), bfpQrOfTerm(5, 0));
    TEST_ASSERT_EQUAL_INT32(5, r.q);
    TEST_ASSERT_TRUE(r.sticky);
    ASSERT_EXITS_WITH_FAILURE(twoStickyBody());
}

/* ---- fail fast ---------------------------------------------------------- */

static const int32_t badCodes[4] = {1, 2, 3, 4};
static const uint8_t badStored[4] = {127, 127, 127, 127};

static void setupBad(void) {
    moInit(&fixV, 4, 1, 0, 8, 8, badCodes, badStored);
    moInit(&fixG, 4, 1, 0, 8, 8, badCodes, badStored);
}
static void badSrNoStream(void) {
    setupBad();
    moRun(muOf(0.9f), SR_HALF_AWAY, NULL);
}
static void badMuAboveOne(void) {
    setupBad();
    moRun((bfpScalar_t){.q = 16385, .e = -14}, HALF_AWAY, NULL);
}
static void badMuNegative(void) {
    setupBad();
    moRun((bfpScalar_t){.q = -16384, .e = -15}, HALF_AWAY, NULL);
}
static void badMuNotNormalized(void) {
    setupBad();
    moRun((bfpScalar_t){.q = 100, .e = -10}, HALF_AWAY, NULL);
}
static void badAlias(void) {
    setupBad();
    momentumStoreBfpInPlace(&fixV.t, &fixV.t, muOf(0.9f), HALF_AWAY, NULL);
}
/* a per-tensor gradient on a grouped state is legal (one exponent for all
 * groups); a gradient grouped differently from the state is not */
static void badGeometry(void) {
    moInit(&fixV, 4, 1, 0, 8, 8, badCodes, badStored);
    moInit(&fixG, 4, 2, 2, 8, 8, badCodes, badStored);
    moRun(muOf(0.9f), HALF_AWAY, NULL);
}
static void badCount(void) {
    moInit(&fixV, 4, 1, 0, 8, 8, badCodes, badStored);
    moInit(&fixG, 3, 1, 0, 8, 8, badCodes, badStored);
    moRun(muOf(0.9f), HALF_AWAY, NULL);
}
static void badGradDtype(void) {
    setupBad();
    static float gf[4] = {1, 2, 3, 4};
    static quantization_t fq;
    initFloat32Quantization(&fq);
    static tensor_t gt;
    setTensorValues(&gt, (uint8_t *)gf, &fixG.shape, &fq, NULL);
    momentumStoreBfpInPlace(&fixV.t, &gt, muOf(0.9f), HALF_AWAY, NULL);
}

void testMomentumRejectsBadInputs(void) {
    ASSERT_EXITS_WITH_FAILURE(badSrNoStream());
    ASSERT_EXITS_WITH_FAILURE(badMuAboveOne());
    ASSERT_EXITS_WITH_FAILURE(badMuNegative());
    ASSERT_EXITS_WITH_FAILURE(badMuNotNormalized());
    ASSERT_EXITS_WITH_FAILURE(badAlias());
    ASSERT_EXITS_WITH_FAILURE(badGeometry());
    ASSERT_EXITS_WITH_FAILURE(badCount());
    ASSERT_EXITS_WITH_FAILURE(badGradDtype());
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMomentumMatchesExactReferenceOnRandomStates);
    RUN_TEST(testMomentumPerTensorGradientOnGroupedStateMatchesReference);
    RUN_TEST(testMomentumDeadZonePinned);
    RUN_TEST(testMomentumDeadZoneStatistical);
    RUN_TEST(testMomentumIsUnbiasedUnderSr);
    RUN_TEST(testMomentumMuZeroIsTheGradientExactly);
    RUN_TEST(testMomentumMuOneIsExactAddition);
    RUN_TEST(testMomentumExactCancellationIsTheZeroState);
    RUN_TEST(testMomentumNearCancellationRegridsToTheResidue);
    RUN_TEST(testMomentumCancellationBelowTheWorkingGridKeepsEc);
    RUN_TEST(testMomentumExponentCapAndFloor);
    RUN_TEST(testMomentumInPlaceKeepsOldExponentsUntilTheGroupIsRead);
    RUN_TEST(testMomentumEmptyTensorIsTheZeroState);
    RUN_TEST(testQrAddRejectsTwoStickyOperands);
    RUN_TEST(testMomentumRejectsBadInputs);
    return UNITY_END();
}
