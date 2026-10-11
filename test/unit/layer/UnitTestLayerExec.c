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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerExecPhaseKeyIsNullWithoutAContext);
    RUN_TEST(testLayerExecPhaseKeyFoldsTheForwardPhase);
    RUN_TEST(testLayerExecPhaseKeyFoldsTheBackwardPhase);
    RUN_TEST(testLayerExecOpKeyFoldsThePhaseThenTheOpIndex);
    RUN_TEST(testLayerExecOpKeyIsNullWithoutAContext);
    return UNITY_END();
}
