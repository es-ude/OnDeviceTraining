#define SOURCE_FILE "UNIT_TEST_LAYER_EXEC"

#include <stddef.h>
#include <stdint.h>

#include "LayerExec.h"
#include "unity.h"

/* The expected keys come from the Python mirror, not from the C fold, with
 * the tag values spelled as numbers (FORWARD 4, BACKWARD 5). From the
 * repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (fold,
 *   key_from_seed as kfs); k = kfs(0x1a7e); print([[hex(w) for w in x]
 *   for x in (k, fold(k, 4, 0), fold(k, 5, 0))])"
 * Each check also compares with the explicit rngFold chain, so a wrong tag
 * value and a wrong fold fail differently. */
#define LAYER_EXEC_TEST_SEED 0x1a7eu

void setUp(void) {}
void tearDown(void) {}

static void assertKey(uint32_t w0, uint32_t w1, rngKey_t key) {
    TEST_ASSERT_EQUAL_HEX32(w0, key.w[0]);
    TEST_ASSERT_EQUAL_HEX32(w1, key.w[1]);
}

static void assertSameKey(rngKey_t expected, rngKey_t actual) {
    TEST_ASSERT_EQUAL_HEX32(expected.w[0], actual.w[0]);
    TEST_ASSERT_EQUAL_HEX32(expected.w[1], actual.w[1]);
}

/* A NULL context means the caller supplies no randomness: no phase key, and
 * the caller's storage is left alone. */
void testLayerExecPhaseKeyIsNullWithoutAContext(void) {
    rngKey_t storage = {{0xdeadbeefu, 0xdeadbeefu}};

    TEST_ASSERT_NULL(layerExecPhaseKey(NULL, RNG_TAG_FORWARD, &storage));
    TEST_ASSERT_NULL(layerExecPhaseKey(NULL, RNG_TAG_BACKWARD, &storage));
    assertKey(0xdeadbeefu, 0xdeadbeefu, storage);
}

/* The phase key is the layer key folded with (phase, 0), written into the
 * caller's storage; the returned pointer is that storage. */
void testLayerExecPhaseKeyFoldsTheForwardPhase(void) {
    layerExec_t exec = {.key = rngKeyFromSeed(LAYER_EXEC_TEST_SEED)};
    assertKey(0xd3d4190fu, 0x904c56a5u, exec.key);
    rngKey_t storage;

    const rngKey_t *phase = layerExecPhaseKey(&exec, RNG_TAG_FORWARD, &storage);

    TEST_ASSERT_EQUAL_PTR(&storage, phase);
    assertSameKey(rngFold(exec.key, RNG_TAG_FORWARD, 0u), storage);
    assertKey(0x94c78887u, 0x0ebfb008u, storage);
}

/* The backward folds its own phase from the same layer key, so the forward
 * and the backward of one layer never share a key. */
void testLayerExecPhaseKeyFoldsTheBackwardPhase(void) {
    layerExec_t exec = {.key = rngKeyFromSeed(LAYER_EXEC_TEST_SEED)};
    rngKey_t storage;

    const rngKey_t *phase = layerExecPhaseKey(&exec, RNG_TAG_BACKWARD, &storage);

    TEST_ASSERT_EQUAL_PTR(&storage, phase);
    assertSameKey(rngFold(exec.key, RNG_TAG_BACKWARD, 0u), storage);
    assertKey(0xfc1567afu, 0xac9a7ff1u, storage);
}

/* The op key is the phase key folded with (OP, opIndex). Python, tags as
 * numbers (FORWARD 4, BACKWARD 5, OP 7):
 *   uv run python -c "from examples._shared.keyed_rng import (fold,
 *   key_from_seed as kfs); k = kfs(0x1a7e); print([[hex(w) for w in x]
 *   for x in (fold(fold(k, 4, 0), 7, 0), fold(fold(k, 5, 0), 7, 3))])" */
void testLayerExecOpKeyFoldsThePhaseThenTheOpIndex(void) {
    layerExec_t exec = {.key = rngKeyFromSeed(LAYER_EXEC_TEST_SEED)};
    rngKey_t phase;
    rngKey_t storage;

    const rngKey_t *op = layerExecOpKey(&exec, RNG_TAG_FORWARD, 0u, &storage);
    TEST_ASSERT_EQUAL_PTR(&storage, op);
    assertSameKey(rngFold(*layerExecPhaseKey(&exec, RNG_TAG_FORWARD, &phase), RNG_TAG_OP, 0u),
                  storage);
    assertKey(0x260d8a9du, 0x002d0794u, storage);

    op = layerExecOpKey(&exec, RNG_TAG_BACKWARD, 3u, &storage);
    TEST_ASSERT_EQUAL_PTR(&storage, op);
    assertSameKey(rngFold(*layerExecPhaseKey(&exec, RNG_TAG_BACKWARD, &phase), RNG_TAG_OP, 3u),
                  storage);
    assertKey(0x05fbab60u, 0x814d2f62u, storage);
}

void testLayerExecOpKeyIsNullWithoutAContext(void) {
    rngKey_t storage = {{0xdeadbeefu, 0xdeadbeefu}};

    TEST_ASSERT_NULL(layerExecOpKey(NULL, RNG_TAG_BACKWARD, 3u, &storage));
    assertKey(0xdeadbeefu, 0xdeadbeefu, storage);
}

/* A purpose stream is the stream of the phase key folded with (purpose, 0).
 * Python, tags as numbers (FORWARD 4, BACKWARD 5, CONVERT 12, DROPOUT_MASK 13,
 * DROPOUT_BRIDGE 14, SOFTMAX_SHIFT 15, NORM_DX 16, LOSS 17):
 *   uv run python -c "from examples._shared.keyed_rng import (fold, key_from_seed as kfs,
 *   open_stream as o); k = kfs(0x1a7e); print([hex(o(fold(fold(k, p, 0), t, 0))) for (p, t) in
 *   ((4, 12), (5, 12), (4, 13), (4, 14), (5, 14), (4, 15), (5, 16), (5, 17))])" */
void testLayerExecPurposeStreamOpensThePhaseKeyFoldedWithThePurpose(void) {
    static const struct {
        uint32_t phase;
        uint32_t purpose;
        uint32_t pythonState;
    } cases[] = {
        {RNG_TAG_FORWARD, RNG_TAG_CONVERT, 0x1ccf7b1cu},
        {RNG_TAG_BACKWARD, RNG_TAG_CONVERT, 0x143ac223u},
        {RNG_TAG_FORWARD, RNG_TAG_DROPOUT_MASK, 0x86a64e58u},
        {RNG_TAG_FORWARD, RNG_TAG_DROPOUT_BRIDGE, 0x25c63c10u},
        {RNG_TAG_BACKWARD, RNG_TAG_DROPOUT_BRIDGE, 0xa9448e71u},
        {RNG_TAG_FORWARD, RNG_TAG_SOFTMAX_SHIFT, 0xdfc1f67cu},
        {RNG_TAG_BACKWARD, RNG_TAG_NORM_DX, 0x8683620bu},
        {RNG_TAG_BACKWARD, RNG_TAG_LOSS, 0x7114be1fu},
    };
    layerExec_t exec = {.key = rngKeyFromSeed(LAYER_EXEC_TEST_SEED)};
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        rngKey_t phase;
        rngStream_t storage;

        rngStream_t *stream =
            layerExecPurposeStream(&exec, cases[c].phase, cases[c].purpose, &storage);

        TEST_ASSERT_EQUAL_PTR(&storage, stream);
        rngKey_t purposeKey =
            rngFold(*layerExecPhaseKey(&exec, cases[c].phase, &phase), cases[c].purpose, 0u);
        TEST_ASSERT_EQUAL_HEX32(rngOpen(purposeKey).state, storage.state);
        TEST_ASSERT_EQUAL_HEX32(cases[c].pythonState, storage.state);
    }
}

void testLayerExecPurposeStreamIsNullWithoutAContext(void) {
    rngStream_t storage = {.state = 0xdeadbeefu};

    TEST_ASSERT_NULL(layerExecPurposeStream(NULL, RNG_TAG_BACKWARD, RNG_TAG_LOSS, &storage));
    TEST_ASSERT_EQUAL_HEX32(0xdeadbeefu, storage.state);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerExecPhaseKeyIsNullWithoutAContext);
    RUN_TEST(testLayerExecPhaseKeyFoldsTheForwardPhase);
    RUN_TEST(testLayerExecPhaseKeyFoldsTheBackwardPhase);
    RUN_TEST(testLayerExecOpKeyFoldsThePhaseThenTheOpIndex);
    RUN_TEST(testLayerExecOpKeyIsNullWithoutAContext);
    RUN_TEST(testLayerExecPurposeStreamOpensThePhaseKeyFoldedWithThePurpose);
    RUN_TEST(testLayerExecPurposeStreamIsNullWithoutAContext);
    return UNITY_END();
}
