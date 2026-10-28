#define SOURCE_FILE "UNIT_TEST_NPY_LOADER_API"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "DeathTest.h"
#include "NPYLoaderApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#define NPY_TEST_ALARM_SECONDS 5u

void setUp(void) {}
void tearDown(void) {}

static void writeRaw(const uint8_t *bytes, size_t n) {
    FILE *f = fopen(NPY_FIXTURE_PATH, "wb");
    TEST_ASSERT_NOT_NULL(f);
    TEST_ASSERT_EQUAL_size_t(n, fwrite(bytes, 1, n, f));
    TEST_ASSERT_EQUAL_INT(0, fclose(f));
}

/* declaredLen is written as-is while strlen(dict) bytes of dict follow, so a
 * test can declare a header length the file does not hold. */
static void writeNpy(uint8_t major, uint8_t minor, uint32_t declaredLen, const char *dict,
                     const void *payload, size_t payloadBytes) {
    FILE *f = fopen(NPY_FIXTURE_PATH, "wb");
    TEST_ASSERT_NOT_NULL(f);
    const uint8_t preamble[12] = {0x93,
                                  'N',
                                  'U',
                                  'M',
                                  'P',
                                  'Y',
                                  major,
                                  minor,
                                  (uint8_t)(declaredLen & 0xFFu),
                                  (uint8_t)((declaredLen >> 8) & 0xFFu),
                                  (uint8_t)((declaredLen >> 16) & 0xFFu),
                                  (uint8_t)((declaredLen >> 24) & 0xFFu)};
    size_t preambleBytes = (major == 1) ? 10u : 12u;
    TEST_ASSERT_EQUAL_size_t(preambleBytes, fwrite(preamble, 1, preambleBytes, f));
    size_t dictBytes = strlen(dict);
    TEST_ASSERT_EQUAL_size_t(dictBytes, fwrite(dict, 1, dictBytes, f));
    if (payloadBytes > 0) {
        TEST_ASSERT_EQUAL_size_t(payloadBytes, fwrite(payload, 1, payloadBytes, f));
    }
    TEST_ASSERT_EQUAL_INT(0, fclose(f));
}

/* Every death test loads through these so a parser that spins dies by SIGALRM
 * instead of hanging ctest. Only ever called inside a forked death-test child. */
static void loadUnderAlarm(void) {
    alarm(NPY_TEST_ALARM_SECONDS);
    (void)npyLoad(NPY_FIXTURE_PATH);
}

static void loadFlatUnderAlarm(void) {
    alarm(NPY_TEST_ALARM_SECONDS);
    (void)npyLoadFlat(NPY_FIXTURE_PATH);
}

static const char *const V2_DICT = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
static const float SIX_FLOATS[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

void testNpyRejectsBadMagic(void) {
    const uint8_t bytes[16] = {'N', 'O', 'T', 'N', 'P', 'Y', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    writeRaw(bytes, sizeof(bytes));
    ASSERT_EXITS_WITH_OUTPUT(1, "is not a .npy file", loadUnderAlarm());
}

void testNpyRejectsUnsupportedVersion(void) {
    writeNpy(4, 0, (uint32_t)strlen(V2_DICT), V2_DICT, SIX_FLOATS, sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "unsupported .npy version 4.0", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "unsupported .npy version 4.0", loadFlatUnderAlarm());
}

void testNpyRejectsTruncatedPreamble(void) {
    const uint8_t bytes[9] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0x40};
    writeRaw(bytes, sizeof(bytes));
    ASSERT_EXITS_WITH_OUTPUT(1, "truncated preamble", loadUnderAlarm());
}

void testNpyRejectsHeaderLengthAboveCap(void) {
    /* The padded dict is a valid header of exactly 10001 bytes and the payload
     * follows, so only the cap stands between this file and a clean load. */
    static char padded[10002];
    size_t dictBytes = strlen(V2_DICT);
    memcpy(padded, V2_DICT, dictBytes);
    memset(padded + dictBytes, ' ', 10000u - dictBytes);
    padded[10000] = '\n';
    padded[10001] = '\0';
    writeNpy(2, 0, 10001u, padded, SIX_FLOATS, sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "exceeds the 10000-byte cap", loadUnderAlarm());
}

void testNpyRejectsHeaderLengthBeyondFile(void) {
    writeNpy(2, 0, 5000u, V2_DICT, NULL, 0);
    ASSERT_EXITS_WITH_OUTPUT(1, "runs past the end of the file", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "runs past the end of the file", loadFlatUnderAlarm());
}

void testNpyLoadsV2Header(void) {
    writeNpy(2, 0, (uint32_t)strlen(V2_DICT), V2_DICT, SIX_FLOATS, sizeof(SIX_FLOATS));

    tensorArray_t *arr = npyLoad(NPY_FIXTURE_PATH);
    size_t capturedSize = arr->size;
    float captured[6] = {0};
    memcpy(&captured[0], arr->array[0]->data, 3 * sizeof(float));
    memcpy(&captured[3], arr->array[1]->data, 3 * sizeof(float));

    freeTensorArray(arr);

    TEST_ASSERT_EQUAL_size_t(2, capturedSize);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(SIX_FLOATS, captured, 6);
}

static void writeDict(const char *dict, const void *payload, size_t payloadBytes) {
    writeNpy(1, 0, (uint32_t)strlen(dict), dict, payload, payloadBytes);
}

void testNpyRejectsMissingShapeKey(void) {
    writeDict("{'descr': '<f4', 'fortran_order': False, }", SIX_FLOATS, sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "no 'shape' key", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "no 'shape' key", loadFlatUnderAlarm());
}

void testNpyRejectsMissingDescrKey(void) {
    writeDict("{'fortran_order': False, 'shape': (2, 3), }", SIX_FLOATS, sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "no 'descr' key", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "no 'descr' key", loadFlatUnderAlarm());
}

void testNpyRejectsMissingFortranOrderKey(void) {
    writeDict("{'descr': '<f4', 'shape': (2, 3), }", SIX_FLOATS, sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "no 'fortran_order' key", loadUnderAlarm());
}

void testNpyRejectsKeyWithoutColon(void) {
    writeDict("{'descr': '<f4', 'fortran_order': False, 'shape' (2, 3), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "no ':' after 'shape'", loadUnderAlarm());
}

void testNpyRejectsUnsupportedDtype(void) {
    writeDict("{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "unsupported dtype", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "unsupported dtype", loadFlatUnderAlarm());
}

void testNpyRejectsFortranOrder(void) {
    writeDict("{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "fortran_order True", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "fortran_order True", loadFlatUnderAlarm());

    writeDict("{'descr': '<f4', 'fortran_order': 0, 'shape': (2, 3), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "fortran_order must be False", loadUnderAlarm());

    writeDict("{'descr': '<f4', 'fortran_order': Falsey, 'shape': (2, 3), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "fortran_order must be False", loadUnderAlarm());
}

void testNpyRejectsBadShapeChar(void) {
    writeDict("{'descr': '<f4', 'fortran_order': False, 'shape': (3, x), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "malformed shape tuple", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "malformed shape tuple", loadFlatUnderAlarm());

    writeDict("{'descr': '<f4', 'fortran_order': False, 'shape': [6), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "malformed shape tuple", loadUnderAlarm());
}

void testNpyRejectsMisplacedShapeCommas(void) {
    writeDict("{'descr': '<f4', 'fortran_order': False, 'shape': (2,, 3), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "malformed shape tuple", loadUnderAlarm());

    writeDict("{'descr': '<f4', 'fortran_order': False, 'shape': (, 6), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "malformed shape tuple", loadUnderAlarm());

    writeDict("{'descr': '<f4', 'fortran_order': False, 'shape': (2 3), }", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "malformed shape tuple", loadUnderAlarm());
}

void testNpyRejectsUnclosedShape(void) {
    writeDict("{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4", SIX_FLOATS,
              sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "malformed shape tuple", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "malformed shape tuple", loadFlatUnderAlarm());
}

void testNpyRejectsDimensionOverflow(void) {
    writeDict("{'descr': '<f4', 'fortran_order': False, 'shape': (99999999999999999999, 1), }",
              SIX_FLOATS, sizeof(SIX_FLOATS));
    ASSERT_EXITS_WITH_OUTPUT(1, "shape dimension overflows", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "shape dimension overflows", loadFlatUnderAlarm());
}

void testNpyRejectsTooManyDims(void) {
    static char dict[256];
    size_t n = 0;
    const char *head = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    memcpy(dict, head, strlen(head));
    n += strlen(head);
    for (size_t i = 0; i < 33; i++) {
        memcpy(dict + n, "1, ", 3);
        n += 3;
    }
    memcpy(dict + n, "), }", 5);
    writeDict(dict, SIX_FLOATS, sizeof(float));
    ASSERT_EXITS_WITH_OUTPUT(1, "more than 32 dimensions", loadUnderAlarm());
    ASSERT_EXITS_WITH_OUTPUT(1, "more than 32 dimensions", loadFlatUnderAlarm());
}

static void assertShapeSpellingLoads(const char *dict, size_t expectedSamples) {
    writeDict(dict, SIX_FLOATS, sizeof(SIX_FLOATS));

    tensor_t *flat = npyLoadFlat(NPY_FIXTURE_PATH);
    size_t flatElements = calcNumberOfElementsByTensor(flat);
    float flatCaptured[6] = {0};
    memcpy(flatCaptured, flat->data, sizeof(flatCaptured));
    freeTensor(flat);
    TEST_ASSERT_EQUAL_size_t(6, flatElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(SIX_FLOATS, flatCaptured, 6);

    tensorArray_t *arr = npyLoad(NPY_FIXTURE_PATH);
    size_t samples = arr->size;
    freeTensorArray(arr);
    TEST_ASSERT_EQUAL_size_t(expectedSamples, samples);
}

void testNpyParsesShapeSpellings(void) {
    assertShapeSpellingLoads("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", 2);
    assertShapeSpellingLoads("{'descr': '<f4', 'fortran_order': False, 'shape': (2,3,), }", 2);
    assertShapeSpellingLoads("{'descr': '<f4', 'fortran_order': False, 'shape': (6,), }", 6);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testNpyRejectsBadMagic);
    RUN_TEST(testNpyRejectsUnsupportedVersion);
    RUN_TEST(testNpyRejectsTruncatedPreamble);
    RUN_TEST(testNpyRejectsHeaderLengthAboveCap);
    RUN_TEST(testNpyRejectsHeaderLengthBeyondFile);
    RUN_TEST(testNpyLoadsV2Header);
    RUN_TEST(testNpyRejectsMissingShapeKey);
    RUN_TEST(testNpyRejectsMissingDescrKey);
    RUN_TEST(testNpyRejectsMissingFortranOrderKey);
    RUN_TEST(testNpyRejectsKeyWithoutColon);
    RUN_TEST(testNpyRejectsUnsupportedDtype);
    RUN_TEST(testNpyRejectsFortranOrder);
    RUN_TEST(testNpyRejectsBadShapeChar);
    RUN_TEST(testNpyRejectsMisplacedShapeCommas);
    RUN_TEST(testNpyRejectsUnclosedShape);
    RUN_TEST(testNpyRejectsDimensionOverflow);
    RUN_TEST(testNpyRejectsTooManyDims);
    RUN_TEST(testNpyParsesShapeSpellings);
    return UNITY_END();
}
