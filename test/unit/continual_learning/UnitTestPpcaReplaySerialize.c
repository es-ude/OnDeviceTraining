#define SOURCE_FILE "UNIT_TEST_PPCA_REPLAY_SERIALIZE"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeathTest.h"
#include "ExecuteOp.h" /* executeConvert for the packed-state snapshot */
#include "PpcaReplay.h"
#include "PpcaReplayApi.h"
#include "PpcaReplaySerialize.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#define FILE_PATH PPCA_SERIALIZE_TEST_FILE_PATH

void setUp(void) {}
void tearDown(void) {}

/* floatConfig / packedConfig: copied VERBATIM from UnitTestPpcaReplay.c
 * (same file-local static pattern; Tasks 7/11). */
static ppcaReplayConfig_t floatConfig(size_t dim, size_t rank, size_t maxM) {
    static quantization_t floatQ; /* static: outlives the call; cloned by create */
    initFloat32Quantization(&floatQ);
    ppcaReplayConfig_t cfg = {
        .dim = dim,
        .rank = rank,
        .maxSessionSamples = maxM,
        .mergeMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .streamMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .sampleMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .meanQ = &floatQ,
        .basisQ = &floatQ,
        .eigvalsQ = &floatQ,
        .sigma2Floor = 1e-6f,
        .shrinkageGamma = 0.0f,
    };
    return cfg;
}

static ppcaReplayConfig_t packedConfig(size_t dim, size_t rank, size_t maxM, qtype_t basisType) {
    ppcaReplayConfig_t cfg = floatConfig(dim, rank, maxM);
    /* Stack/static-fixture idiom (group-quant PR1): this helper is called
     * from multiple tests in the same binary, and `static` storage means a
     * heap-allocating initSymQConfig call here would leak its previous
     * scales array on every call after the first. Build the config directly
     * instead (same values initSymQConfig(8, HALF_AWAY, ...) would produce);
     * never freed, per the stack-fixture convention. */
    static float symScale[1] = {1.f};
    static symQConfig_t symQc = {
        .scales = symScale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 8};
    static quantization_t symQ;
    /* PR4: same idiom for ASYM's TWO arrays (initAsymQConfig would leak two
     * heap blocks per call after the first). */
    static float asymScale[1] = {1.f};
    static uint16_t asymZp[1] = {0};
    static asymQConfig_t asymQc = {.scales = asymScale,
                                   .zeroPoints = asymZp,
                                   .numGroups = 1,
                                   .groupSize = 0,
                                   .roundingMode = HALF_AWAY,
                                   .qBits = 8};
    static quantization_t asymQ;
    initSymQuantization(&symQc, &symQ);
    initAsymQuantization(&asymQc, &asymQ);
    if (basisType == SYM) {
        cfg.basisQ = &symQ;
        cfg.meanQ = &asymQ; /* mean has an offset -> ASYM */
    } else {
        cfg.basisQ = &asymQ;
        cfg.meanQ = &asymQ;
    }
    return cfg;
}

/*! Group-quant PR2 (Task 5): GROUPED-SYM twin of packedConfig's SYM branch --
 *  numGroups=3, groupSize=4 (rank*dim = 2*6 = 12 = 3*4, matching
 *  floatConfig(6,2,...)'s basis shape [rank=2,dim=6]). getQLike's grouped
 *  clone (Task 1) deep-copies both numGroups/groupSize AND these scale
 *  VALUES into every generator built from this config -- unlike
 *  packedConfig's per-tensor template, which getQLike instead resets to
 *  scale=1.f (no group grid to preserve). Never freed, per the
 *  stack-fixture convention (see packedConfig above). */
static ppcaReplayConfig_t groupedPackedConfig(size_t dim, size_t rank, size_t maxM) {
    ppcaReplayConfig_t cfg = floatConfig(dim, rank, maxM);
    static float groupedScales[3] = {0.1f, 0.2f, 0.3f};
    static symQConfig_t groupedSymQc = {.scales = groupedScales,
                                        .numGroups = 3,
                                        .groupSize = 4,
                                        .roundingMode = HALF_AWAY,
                                        .qBits = 8};
    static quantization_t groupedSymQ;
    initSymQuantization(&groupedSymQc, &groupedSymQ);
    static float asymScale[1] = {1.f};
    static uint16_t asymZp[1] = {0};
    static asymQConfig_t asymQc = {.scales = asymScale,
                                   .zeroPoints = asymZp,
                                   .numGroups = 1,
                                   .groupSize = 0,
                                   .roundingMode = HALF_AWAY,
                                   .qBits = 8};
    static quantization_t asymQ;
    initAsymQuantization(&asymQc, &asymQ);
    cfg.basisQ = &groupedSymQ;
    cfg.meanQ = &asymQ; /* mean has an offset -> ASYM, mirrors packedConfig */
    return cfg;
}

/*! group-quant PR4 (Task 4): GROUPED-ASYM twin of groupedPackedConfig above
 *  -- numGroups=3, groupSize=4 (same shape as the grouped-SYM basis), scales
 *  AND zeroPoints both pairwise distinct so a v5 record field-order bug
 *  (scales/zeroPoints swap) or byte-order bug is caught on round trip.
 *  meanQ stays per-tensor ASYM, mirroring groupedPackedConfig's meanQ.
 *  Never freed, per the stack-fixture convention (see packedConfig above). */
static ppcaReplayConfig_t groupedAsymPackedConfig(size_t dim, size_t rank, size_t maxM) {
    ppcaReplayConfig_t cfg = floatConfig(dim, rank, maxM);
    static float groupedAsymScales[3] = {0.1f, 0.2f, 0.3f};
    static uint16_t groupedAsymZps[3] = {5, 120, 250};
    static asymQConfig_t groupedAsymQc = {.scales = groupedAsymScales,
                                          .zeroPoints = groupedAsymZps,
                                          .numGroups = 3,
                                          .groupSize = 4,
                                          .roundingMode = HALF_AWAY,
                                          .qBits = 8};
    static quantization_t groupedAsymQ;
    initAsymQuantization(&groupedAsymQc, &groupedAsymQ);
    static float meanAsymScale[1] = {1.f};
    static uint16_t meanAsymZp[1] = {0};
    static asymQConfig_t meanAsymQc = {.scales = meanAsymScale,
                                       .zeroPoints = meanAsymZp,
                                       .numGroups = 1,
                                       .groupSize = 0,
                                       .roundingMode = HALF_AWAY,
                                       .qBits = 8};
    static quantization_t meanAsymQ;
    initAsymQuantization(&meanAsymQc, &meanAsymQ);
    cfg.basisQ = &groupedAsymQ;
    cfg.meanQ = &meanAsymQ;
    return cfg;
}

/* Seed a set with NON-UNIFORM state so field-order swaps corrupt bytes
 * detectably (serial-module fixture discipline). */
static void seedSet(ppcaReplaySet_t *set) {
    for (size_t c = 0; c < set->numClasses; c++) {
        ppcaReplay_t *g = set->generators[c];
        float *mu = (float *)g->mean->data; /* FLOAT32 config in tests */
        float *b = (float *)g->basis->data;
        float *lam = (float *)g->eigvals->data;
        for (size_t j = 0; j < g->dim; j++) {
            mu[j] = 0.25f * (float)(c + 1) * (float)j;
        }
        for (size_t i = 0; i < g->rank * g->dim; i++) {
            b[i] = 0.01f * (float)i - 0.05f * (float)c;
        }
        for (size_t i = 0; i < g->rank; i++) {
            lam[i] = 3.0f - (float)i - 0.1f * (float)c;
        }
        g->sigma2 = 0.5f + (float)c; /* != totalVar != count: order-swap trap */
        g->totalVar = 100.0f + (float)c;
        g->count = 42 + (uint32_t)c;
    }
}

void testRoundTripFloat(void) {
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(2, &cfg);
    ppcaReplaySet_t *deserial = ppcaReplaySetCreate(2, &cfg);
    seedSet(serial);

    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    fclose(f);
    f = fopen(FILE_PATH, "rb");
    ppcaReplaySetDeserialize(deserial, f);
    fclose(f);

    for (size_t c = 0; c < 2; c++) {
        ppcaReplay_t *s = serial->generators[c];
        ppcaReplay_t *d = deserial->generators[c];
        TEST_ASSERT_EQUAL_UINT32(s->count, d->count);
        TEST_ASSERT_EQUAL_FLOAT(s->sigma2, d->sigma2);
        TEST_ASSERT_EQUAL_FLOAT(s->totalVar, d->totalVar);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)s->mean->data, (float *)d->mean->data, 6);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)s->basis->data, (float *)d->basis->data, 12);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)s->eigvals->data, (float *)d->eigvals->data, 2);
    }
    freePpcaReplaySet(deserial);
    freePpcaReplaySet(serial);
}

void testRoundTripPacked(void) {
    /* SYM@8 basis + ASYM@8 mean survive byte-tight (serializeTensor packed
     * round-trip) — also the wire-layout-coupling test: the deserialize
     * peek must consume EXACTLY the record header serializeTensor wrote,
     * or the rewind lands wrong and this roundtrip fails loudly. */
    ppcaReplayConfig_t cfgF = floatConfig(6, 2, 8);
    ppcaReplaySet_t *train = ppcaReplaySetCreate(1, &cfgF);
    seedSet(train);
    ppcaReplayConfig_t cfgP = packedConfig(6, 2, 8, SYM);
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(1, &cfgP);
    ppcaReplaySet_t *deserial = ppcaReplaySetCreate(1, &cfgP);
    executeConvert(train->generators[0]->mean, serial->generators[0]->mean);
    executeConvert(train->generators[0]->basis, serial->generators[0]->basis);
    executeConvert(train->generators[0]->eigvals, serial->generators[0]->eigvals);
    serial->generators[0]->sigma2 = 0.75f;
    serial->generators[0]->totalVar = 12.5f;
    serial->generators[0]->count = 9;

    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    fclose(f);
    f = fopen(FILE_PATH, "rb");
    ppcaReplaySetDeserialize(deserial, f);
    fclose(f);

    size_t basisBytes = calcNumberOfBytesForData(serial->generators[0]->basis->quantization, 12);
    TEST_ASSERT_EQUAL_MEMORY(serial->generators[0]->basis->data,
                             deserial->generators[0]->basis->data, basisBytes);
    size_t meanBytes = calcNumberOfBytesForData(serial->generators[0]->mean->quantization, 6);
    TEST_ASSERT_EQUAL_MEMORY(serial->generators[0]->mean->data, deserial->generators[0]->mean->data,
                             meanBytes);
    TEST_ASSERT_EQUAL_UINT32(9, deserial->generators[0]->count);
    /* qConfig metadata is file-carried (Serialize.c writes scale/zeroPoint per
     * tensor) -- pin it so packed payload bytes can never silently decode
     * against a drifted grid (PR #366 review). */
    TEST_ASSERT_EQUAL_FLOAT(
        ((symQConfig_t *)serial->generators[0]->basis->quantization->qConfig)->scales[0],
        ((symQConfig_t *)deserial->generators[0]->basis->quantization->qConfig)->scales[0]);
    TEST_ASSERT_EQUAL_FLOAT(
        ((asymQConfig_t *)serial->generators[0]->mean->quantization->qConfig)->scales[0],
        ((asymQConfig_t *)deserial->generators[0]->mean->quantization->qConfig)->scales[0]);
    TEST_ASSERT_EQUAL_UINT16(
        ((asymQConfig_t *)serial->generators[0]->mean->quantization->qConfig)->zeroPoints[0],
        ((asymQConfig_t *)deserial->generators[0]->mean->quantization->qConfig)->zeroPoints[0]);
    freePpcaReplaySet(deserial);
    freePpcaReplaySet(serial);
    freePpcaReplaySet(train);
}

/*! Group-quant PR2 (Task 5): a GROUPED SYM basis (numGroups=3, groupSize=4)
 *  deserialized into a PER-TENSOR skeleton (numGroups=1, groupSize=0, from
 *  packedConfig) round-trips via the same reallocation relax
 *  testDeserializeGroupedSymIntoPerTensorSkeleton pins directly at the
 *  Deserialize.c layer -- exercised here through PPCA's OWN peek-parser,
 *  whose numGroups-equality fail-fast this task drops. Pre-Task-5 this dies
 *  in peekValidateThenDeserializeTensor's SYM arm (file numGroups=3 !=
 *  skeleton's 1) before ever reaching the shared deserializeTensor call. */
void testRoundTripPackedGroupedSym(void) {
    ppcaReplayConfig_t cfgF = floatConfig(6, 2, 8);
    ppcaReplaySet_t *train = ppcaReplaySetCreate(1, &cfgF);
    seedSet(train);
    ppcaReplayConfig_t cfgGrouped = groupedPackedConfig(6, 2, 8);
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(1, &cfgGrouped);
    executeConvert(train->generators[0]->mean, serial->generators[0]->mean);
    executeConvert(train->generators[0]->basis, serial->generators[0]->basis);
    executeConvert(train->generators[0]->eigvals, serial->generators[0]->eigvals);
    serial->generators[0]->sigma2 = 0.75f;
    serial->generators[0]->totalVar = 12.5f;
    serial->generators[0]->count = 9;

    /* Skeleton is built PER-TENSOR (not from cfgGrouped) -- the whole point
     * is the numGroups MISMATCH the relax must tolerate. */
    ppcaReplayConfig_t cfgPerTensor = packedConfig(6, 2, 8, SYM);
    ppcaReplaySet_t *deserial = ppcaReplaySetCreate(1, &cfgPerTensor);

    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    fclose(f);
    f = fopen(FILE_PATH, "rb");
    ppcaReplaySetDeserialize(deserial, f);
    fclose(f);

    symQConfig_t *serialBasisQc = serial->generators[0]->basis->quantization->qConfig;
    symQConfig_t *deserialBasisQc = deserial->generators[0]->basis->quantization->qConfig;
    size_t basisBytes = calcNumberOfBytesForData(serial->generators[0]->basis->quantization, 12);

    /* CAPTURE before any free. */
    bool numGroupsMatch = deserialBasisQc->numGroups == serialBasisQc->numGroups;
    bool groupSizeMatch = deserialBasisQc->groupSize == serialBasisQc->groupSize;
    float capturedSerialScales[3];
    float capturedDeserialScales[3];
    for (size_t g = 0; g < 3; g++) {
        capturedSerialScales[g] = serialBasisQc->scales[g];
        capturedDeserialScales[g] = deserialBasisQc->scales[g];
    }
    uint8_t capturedSerialBasisData[64];
    uint8_t capturedDeserialBasisData[64];
    memcpy(capturedSerialBasisData, serial->generators[0]->basis->data, basisBytes);
    memcpy(capturedDeserialBasisData, deserial->generators[0]->basis->data, basisBytes);
    uint32_t capturedCount = deserial->generators[0]->count;

    freePpcaReplaySet(deserial);
    freePpcaReplaySet(serial);
    freePpcaReplaySet(train);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE(numGroupsMatch);
    TEST_ASSERT_TRUE(groupSizeMatch);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialScales, capturedDeserialScales, 3);
    TEST_ASSERT_EQUAL_MEMORY(capturedSerialBasisData, capturedDeserialBasisData, basisBytes);
    TEST_ASSERT_EQUAL_UINT32(9, capturedCount);
}

/*! group-quant PR4 (Task 4): the ASYM twin of testRoundTripPackedGroupedSym
 *  above -- a GROUPED ASYM basis (numGroups=3, groupSize=4, distinct scales
 *  AND zeroPoints) deserialized into a PER-TENSOR ASYM skeleton (from
 *  packedConfig(..., ASYM)) round-trips via the v5 ASYM arm's reallocation
 *  relax (Task 4), exercised through PPCA's own peek-parser (ASYM arm),
 *  mirroring the SYM test's role exactly: serialize -> peek -> deserialize,
 *  scales AND zeroPoints preserved through the reallocation. */
void testRoundTripPackedGroupedAsym(void) {
    ppcaReplayConfig_t cfgF = floatConfig(6, 2, 8);
    ppcaReplaySet_t *train = ppcaReplaySetCreate(1, &cfgF);
    seedSet(train);
    ppcaReplayConfig_t cfgGrouped = groupedAsymPackedConfig(6, 2, 8);
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(1, &cfgGrouped);
    executeConvert(train->generators[0]->mean, serial->generators[0]->mean);
    executeConvert(train->generators[0]->basis, serial->generators[0]->basis);
    executeConvert(train->generators[0]->eigvals, serial->generators[0]->eigvals);
    serial->generators[0]->sigma2 = 0.75f;
    serial->generators[0]->totalVar = 12.5f;
    serial->generators[0]->count = 9;

    /* Skeleton is built PER-TENSOR ASYM (not from cfgGrouped) -- the whole
     * point is the numGroups MISMATCH the relax must tolerate. */
    ppcaReplayConfig_t cfgPerTensor = packedConfig(6, 2, 8, ASYM);
    ppcaReplaySet_t *deserial = ppcaReplaySetCreate(1, &cfgPerTensor);

    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    fclose(f);
    f = fopen(FILE_PATH, "rb");
    ppcaReplaySetDeserialize(deserial, f);
    fclose(f);

    asymQConfig_t *serialBasisQc = serial->generators[0]->basis->quantization->qConfig;
    asymQConfig_t *deserialBasisQc = deserial->generators[0]->basis->quantization->qConfig;
    size_t basisBytes = calcNumberOfBytesForData(serial->generators[0]->basis->quantization, 12);

    /* CAPTURE before any free. */
    bool numGroupsMatch = deserialBasisQc->numGroups == serialBasisQc->numGroups;
    bool groupSizeMatch = deserialBasisQc->groupSize == serialBasisQc->groupSize;
    float capturedSerialScales[3];
    float capturedDeserialScales[3];
    uint16_t capturedSerialZps[3];
    uint16_t capturedDeserialZps[3];
    for (size_t g = 0; g < 3; g++) {
        capturedSerialScales[g] = serialBasisQc->scales[g];
        capturedDeserialScales[g] = deserialBasisQc->scales[g];
        capturedSerialZps[g] = serialBasisQc->zeroPoints[g];
        capturedDeserialZps[g] = deserialBasisQc->zeroPoints[g];
    }
    uint8_t capturedSerialBasisData[64];
    uint8_t capturedDeserialBasisData[64];
    memcpy(capturedSerialBasisData, serial->generators[0]->basis->data, basisBytes);
    memcpy(capturedDeserialBasisData, deserial->generators[0]->basis->data, basisBytes);
    uint32_t capturedCount = deserial->generators[0]->count;

    freePpcaReplaySet(deserial);
    freePpcaReplaySet(serial);
    freePpcaReplaySet(train);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE(numGroupsMatch);
    TEST_ASSERT_TRUE(groupSizeMatch);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialScales, capturedDeserialScales, 3);
    TEST_ASSERT_EQUAL_UINT16_ARRAY(capturedSerialZps, capturedDeserialZps, 3);
    TEST_ASSERT_EQUAL_MEMORY(capturedSerialBasisData, capturedDeserialBasisData, basisBytes);
    TEST_ASSERT_EQUAL_UINT32(9, capturedCount);
}

/*! BFP epic PR1 (Task 7, #400): ppcaReplayCreate's validateStateStorage
 *  (PpcaReplayApi.c) does not yet allow BFP as PPCA state storage -- this
 *  test hand-assembles the ppcaReplaySet_t/ppcaReplay_t structs directly
 *  (both are public structs; same bypass-the-factory idiom
 *  test/unit/serial/UnitTestDeserialize.c uses for parameter_t) so the
 *  peek parser's new BFP case (skip arithmetic: 4 + numGroups + 3 bytes)
 *  gets real mutation-catching coverage even before any producer can build
 *  this shape through the public API. Basis is grouped (numGroups=2,
 *  groupSize=6, matching rank=2/dim=6 row-grouping); mean/eigvals stay
 *  FLOAT32, mirroring every packedConfig-based test above (eigvalsQ is
 *  never varied from float there either). */
void testRoundTripPackedBfp(void) {
    size_t dim = 6, rank = 2;

    size_t *meanDims = reserveMemory(sizeof(size_t));
    meanDims[0] = dim;
    size_t *meanOrder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, meanOrder);
    shape_t *meanShape = reserveMemory(sizeof(shape_t));
    setShape(meanShape, meanDims, 1, meanOrder);
    tensor_t *serialMean = initTensor(meanShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(serialMean, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, dim);

    size_t *basisDims = reserveMemory(2 * sizeof(size_t));
    basisDims[0] = rank;
    basisDims[1] = dim;
    size_t *basisOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, basisOrder);
    shape_t *basisShape = reserveMemory(sizeof(shape_t));
    setShape(basisShape, basisDims, 2, basisOrder);
    tensor_t *serialBasis =
        initTensor(basisShape, quantizationInitBfpGrouped(4, 8, HALF_AWAY, 2, 6), NULL);
    bfpQConfig_t *serialBasisQc = serialBasis->quantization->qConfig;
    serialBasisQc->exponents[0] = 115;
    serialBasisQc->exponents[1] = 140;
    int32_t basisCodes[] = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6};
    byteConversion((uint8_t *)basisCodes, 32, serialBasis->data, 4, rank * dim);

    size_t *eigvalsDims = reserveMemory(sizeof(size_t));
    eigvalsDims[0] = rank;
    size_t *eigvalsOrder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, eigvalsOrder);
    shape_t *eigvalsShape = reserveMemory(sizeof(shape_t));
    setShape(eigvalsShape, eigvalsDims, 1, eigvalsOrder);
    tensor_t *serialEigvals = initTensor(eigvalsShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(serialEigvals, (float[]){3.0f, 2.0f}, rank);

    ppcaReplay_t serialGen = {0};
    serialGen.dim = dim;
    serialGen.rank = rank;
    serialGen.mean = serialMean;
    serialGen.basis = serialBasis;
    serialGen.eigvals = serialEigvals;
    serialGen.sigma2 = 0.75f;
    serialGen.totalVar = 12.5f;
    serialGen.count = 9;
    ppcaReplay_t *serialGenerators[] = {&serialGen};
    ppcaReplaySet_t serialSet = {
        .numClasses = 1, .generators = serialGenerators, .workspace = NULL};

    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(&serialSet, f);
    fclose(f);

    /* Deserial skeleton: same shapes/configs, fresh (zeroed) tensors. */
    size_t *dMeanDims = reserveMemory(sizeof(size_t));
    dMeanDims[0] = dim;
    size_t *dMeanOrder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, dMeanOrder);
    shape_t *dMeanShape = reserveMemory(sizeof(shape_t));
    setShape(dMeanShape, dMeanDims, 1, dMeanOrder);
    tensor_t *deserialMean = initTensor(dMeanShape, quantizationInitFloat(), NULL);

    size_t *dBasisDims = reserveMemory(2 * sizeof(size_t));
    dBasisDims[0] = rank;
    dBasisDims[1] = dim;
    size_t *dBasisOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, dBasisOrder);
    shape_t *dBasisShape = reserveMemory(sizeof(shape_t));
    setShape(dBasisShape, dBasisDims, 2, dBasisOrder);
    tensor_t *deserialBasis =
        initTensor(dBasisShape, quantizationInitBfpGrouped(4, 8, HALF_AWAY, 2, 6), NULL);

    size_t *dEigvalsDims = reserveMemory(sizeof(size_t));
    dEigvalsDims[0] = rank;
    size_t *dEigvalsOrder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, dEigvalsOrder);
    shape_t *dEigvalsShape = reserveMemory(sizeof(shape_t));
    setShape(dEigvalsShape, dEigvalsDims, 1, dEigvalsOrder);
    tensor_t *deserialEigvals = initTensor(dEigvalsShape, quantizationInitFloat(), NULL);

    ppcaReplay_t deserialGen = {0};
    deserialGen.dim = dim;
    deserialGen.rank = rank;
    deserialGen.mean = deserialMean;
    deserialGen.basis = deserialBasis;
    deserialGen.eigvals = deserialEigvals;
    ppcaReplay_t *deserialGenerators[] = {&deserialGen};
    ppcaReplaySet_t deserialSet = {
        .numClasses = 1, .generators = deserialGenerators, .workspace = NULL};

    f = fopen(FILE_PATH, "rb");
    ppcaReplaySetDeserialize(&deserialSet, f);
    fclose(f);

    /* CAPTURE before any free. */
    bfpQConfig_t *deserialBasisQc = deserialBasis->quantization->qConfig;
    size_t basisBytes = calcNumberOfBytesForData(deserialBasis->quantization, rank * dim);
    uint8_t capturedSerialBasisData[8];
    uint8_t capturedDeserialBasisData[8];
    memcpy(capturedSerialBasisData, serialBasis->data, basisBytes);
    memcpy(capturedDeserialBasisData, deserialBasis->data, basisBytes);
    uint8_t capturedExp0 = deserialBasisQc->exponents[0];
    uint8_t capturedExp1 = deserialBasisQc->exponents[1];
    float capturedMean[6];
    float capturedEigvals[2];
    memcpy(capturedMean, deserialMean->data, sizeof(capturedMean));
    memcpy(capturedEigvals, deserialEigvals->data, sizeof(capturedEigvals));
    uint32_t capturedCount = deserialGen.count;
    float capturedSigma2 = deserialGen.sigma2;
    float capturedTotalVar = deserialGen.totalVar;

    /* FREE: hand-assembled structs are stack-local -- free only the
     * heap-backed tensors directly (freePpcaReplaySet/freePpcaReplay expect
     * a heap-allocated generators array/wrapper struct, which this bypass
     * idiom does not have). */
    freeTensor(serialMean);
    freeTensor(serialBasis);
    freeTensor(serialEigvals);
    freeTensor(deserialMean);
    freeTensor(deserialBasis);
    freeTensor(deserialEigvals);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_HEX8_ARRAY(capturedSerialBasisData, capturedDeserialBasisData, basisBytes);
    TEST_ASSERT_EQUAL_UINT8(115, capturedExp0);
    TEST_ASSERT_EQUAL_UINT8(140, capturedExp1);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}), capturedMean, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){3.0f, 2.0f}), capturedEigvals, 2);
    TEST_ASSERT_EQUAL_UINT32(9, capturedCount);
    TEST_ASSERT_EQUAL_FLOAT(0.75f, capturedSigma2);
    TEST_ASSERT_EQUAL_FLOAT(12.5f, capturedTotalVar);
}

void testDeserializeRejectsDtypeMismatch(void) {
    /* FLOAT32 checkpoint into a packed-built skeleton = the #316 4x-overflow
     * scenario. Must exit BEFORE any skeleton write. */
    ppcaReplayConfig_t cfgF = floatConfig(6, 2, 8);
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(1, &cfgF);
    seedSet(serial);
    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    fclose(f);

    ppcaReplayConfig_t cfgP = packedConfig(6, 2, 8, SYM);
    ppcaReplaySet_t *skeleton = ppcaReplaySetCreate(1, &cfgP);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(ppcaReplaySetDeserialize(skeleton, f));
    fclose(f);
    freePpcaReplaySet(skeleton);
    freePpcaReplaySet(serial);
}

void testDeserializeRejectsQBitsMismatch(void) {
    /* SYM@8 record into a SYM@4 skeleton: SAME dtype enum — the type check
     * alone is insufficient (#316), qBits must be compared too. */
    ppcaReplayConfig_t cfg8 = packedConfig(6, 2, 8, SYM);
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(1, &cfg8);
    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    fclose(f);

    ppcaReplayConfig_t cfg4 = floatConfig(6, 2, 8);
    /* Stack-fixture idiom (group-quant PR1): see packedConfig above. */
    static float qc4Scale[1] = {1.f};
    static symQConfig_t qc4 = {
        .scales = qc4Scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 4};
    static quantization_t q4;
    initSymQuantization(&qc4, &q4);
    cfg4.basisQ = &q4;
    static float aqcScale[1] = {1.f};
    static uint16_t aqcZp[1] = {0};
    static asymQConfig_t aqc = {.scales = aqcScale,
                                .zeroPoints = aqcZp,
                                .numGroups = 1,
                                .groupSize = 0,
                                .roundingMode = HALF_AWAY,
                                .qBits = 8};
    static quantization_t aq;
    initAsymQuantization(&aqc, &aq);
    cfg4.meanQ = &aq;
    ppcaReplaySet_t *skeleton = ppcaReplaySetCreate(1, &cfg4);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(ppcaReplaySetDeserialize(skeleton, f));
    fclose(f);
    freePpcaReplaySet(skeleton);
    freePpcaReplaySet(serial);
}

void testDeserializeRejectsDimMismatch(void) {
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(1, &cfg);
    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    fclose(f);
    ppcaReplayConfig_t cfgBig = floatConfig(8, 2, 8);
    ppcaReplaySet_t *skeleton = ppcaReplaySetCreate(1, &cfgBig);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(ppcaReplaySetDeserialize(skeleton, f));
    fclose(f);
    freePpcaReplaySet(skeleton);
    freePpcaReplaySet(serial);
}

void testDeserializeRejectsBadMagicAndTruncation(void) {
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *skeleton = ppcaReplaySetCreate(1, &cfg);

    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("XXXX", 1, 4, f);
    fclose(f);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(ppcaReplaySetDeserialize(skeleton, f));
    fclose(f);

    /* Truncation: serialize a valid set, then cut the file short. */
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(1, &cfg);
    f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    long full = ftell(f);
    fclose(f);
    FILE *in = fopen(FILE_PATH, "rb");
    char *buf = reserveMemory((size_t)full);
    fread(buf, 1, (size_t)full, in);
    fclose(in);
    f = fopen(FILE_PATH, "wb");
    fwrite(buf, 1, (size_t)full / 2, f);
    fclose(f);
    freeReservedMemory(buf);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(ppcaReplaySetDeserialize(skeleton, f));
    fclose(f);
    freePpcaReplaySet(serial);
    freePpcaReplaySet(skeleton);
}

void testDeserializeRejectsPayloadTruncation(void) {
    /* Cut INSIDE the last tensor's payload region (full-2 bytes): every
     * header/scalar read and the peek still succeed; since #370 the payload
     * read itself fails fast (the post-read record-length check remains as
     * the wire-drift alarm). */
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *skeleton = ppcaReplaySetCreate(1, &cfg);
    ppcaReplaySet_t *serial = ppcaReplaySetCreate(1, &cfg);
    seedSet(serial);
    FILE *f = fopen(FILE_PATH, "wb");
    ppcaReplaySetSerialize(serial, f);
    long full = ftell(f);
    fclose(f);
    FILE *in = fopen(FILE_PATH, "rb");
    char *buf = reserveMemory((size_t)full);
    fread(buf, 1, (size_t)full, in);
    fclose(in);
    f = fopen(FILE_PATH, "wb");
    fwrite(buf, 1, (size_t)full - 2, f);
    fclose(f);
    freeReservedMemory(buf);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(ppcaReplaySetDeserialize(skeleton, f));
    fclose(f);
    freePpcaReplaySet(serial);
    freePpcaReplaySet(skeleton);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRoundTripFloat);
    RUN_TEST(testRoundTripPacked);
    RUN_TEST(testRoundTripPackedGroupedSym);
    RUN_TEST(testRoundTripPackedGroupedAsym);
    RUN_TEST(testRoundTripPackedBfp);
    RUN_TEST(testDeserializeRejectsDtypeMismatch);
    RUN_TEST(testDeserializeRejectsQBitsMismatch);
    RUN_TEST(testDeserializeRejectsDimMismatch);
    RUN_TEST(testDeserializeRejectsBadMagicAndTruncation);
    RUN_TEST(testDeserializeRejectsPayloadTruncation);
    return UNITY_END();
}
