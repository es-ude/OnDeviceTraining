#ifndef ODT_TEST_KEYED_DRAW_CHECK_H
#define ODT_TEST_KEYED_DRAW_CHECK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "Layer.h"
#include "LayerExec.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

/* Checks that a layer's draws are a function of its layer key alone and come
 * from the documented key chain. The expected keys spell the chain out with
 * rngFold, so a defect in LayerExec.c does not cancel out of the comparison. */

#define KEYED_DRAW_MAX_BYTES 512
#define KEYED_DRAW_MIN_ELEMENTS 16

/* The state of the stream the funnel opens for the OUT_WRITE epilogue of op
 * opIndex in phase phaseTag of the layer keyed rngKeyFromSeed(seed). */
static inline uint32_t keyedOpOutState(uint64_t seed, uint32_t phaseTag, uint32_t opIndex) {
    rngKey_t phase = rngFold(rngKeyFromSeed(seed), phaseTag, 0u);
    return rngOpen(rngFold(rngFold(phase, RNG_TAG_OP, opIndex), RNG_TAG_OUT, 0u)).state;
}

/* The stream of a draw site outside the funnel (purposeTag) in phase phaseTag
 * of the layer keyed rngKeyFromSeed(seed). */
static inline rngStream_t keyedPurposeStream(uint64_t seed, uint32_t phaseTag,
                                             uint32_t purposeTag) {
    return rngOpen(rngFold(rngFold(rngKeyFromSeed(seed), phaseTag, 0u), purposeTag, 0u));
}

/* ---- a recording wrapper around one conversionMatrix cell ---- */

typedef struct {
    size_t calls;
    bool hasStream; /* of the first call */
    uint32_t state; /* of the first call's stream */
    qtype_t from;
    qtype_t to;
    conversionFunction_t saved;
} keyedCellRecord_t;

static inline keyedCellRecord_t *keyedCellRecord(void) {
    static keyedCellRecord_t record;
    return &record;
}

/* Records the stream of the first call, then runs the real cell. */
static inline void keyedRecordingCell(tensor_t *in, tensor_t *out, rngStream_t *rs) {
    keyedCellRecord_t *r = keyedCellRecord();
    if (r->calls == 0) {
        r->hasStream = rs != NULL;
        r->state = (rs != NULL) ? rs->state : 0u;
    }
    r->calls++;
    r->saved(in, out, rs);
}

static inline void keyedCellInstall(qtype_t from, qtype_t to) {
    keyedCellRecord_t *r = keyedCellRecord();
    *r = (keyedCellRecord_t){.from = from, .to = to, .saved = conversionMatrix[from][to]};
    conversionMatrix[from][to] = keyedRecordingCell;
}

static inline void keyedCellRestore(void) {
    keyedCellRecord_t *r = keyedCellRecord();
    conversionMatrix[r->from][r->to] = r->saved;
}

/* The recorded cell ran once, with the stream whose state the chain gives;
 * pythonState is the same state from the Python mirror. */
static inline void assertKeyedCellSawOneStream(uint32_t chainState, uint32_t pythonState) {
    keyedCellRecord_t *r = keyedCellRecord();
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, r->calls, "the cell runs once");
    TEST_ASSERT_TRUE_MESSAGE(r->hasStream, "the cell gets a stream");
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(pythonState, chainState, "the chain matches the mirror");
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(chainState, r->state, "the cell gets the documented stream");
}

/* ---- purity ---- */

/* One run of the fixture under the key of seed, drawing (stochastic) or
 * rounding deterministically; writes the result bytes to out, returns their
 * count. */
typedef size_t (*keyedRun_t)(void *ctx, uint64_t seed, bool stochastic,
                             uint8_t out[KEYED_DRAW_MAX_BYTES]);

/* The same key repeats the result; another key draws another result; without
 * draws the key changes nothing (so the difference is the draws'). */
static inline void assertKeyedRunIsPure(keyedRun_t run, void *ctx, uint64_t seed) {
    uint8_t first[KEYED_DRAW_MAX_BYTES];
    uint8_t again[KEYED_DRAW_MAX_BYTES];
    uint8_t other[KEYED_DRAW_MAX_BYTES];
    uint8_t fixed[KEYED_DRAW_MAX_BYTES];
    uint8_t fixedOther[KEYED_DRAW_MAX_BYTES];

    size_t n = run(ctx, seed, true, first);
    size_t againBytes = run(ctx, seed, true, again);
    run(ctx, seed + 1u, true, other);
    run(ctx, seed, false, fixed);
    run(ctx, seed + 1u, false, fixedOther);

    TEST_ASSERT_EQUAL_size_t(n, againBytes);
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(first, again, n, "the same key repeats the result");
    TEST_ASSERT_FALSE_MESSAGE(memcmp(first, other, n) == 0, "another key draws otherwise");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(fixed, fixedOther, n, "without draws the key changes nothing");
}

/* ---- funnel layers with a FLOAT32 forward into a SYM_INT32 output ---- */

/* A built layer, its input, the output dims and the rounding mode the run
 * sets: the forward math's when forwardRounding is set, else the output's. */
typedef struct {
    layer_t *layer;
    tensor_t *input;
    size_t outDims[3];
    size_t outRank;
    roundingMode_t *forwardRounding;
} keyedSymForward_t;

/* A keyedRun_t: one forward into a fresh 8-bit SYM_INT32 output; out = the
 * codes, then the scale. 8 bits: few codes sit on the grid, so most elements
 * move with their draw. */
static inline size_t keyedSymForwardRun(void *ctx, uint64_t seed, bool stochastic,
                                        uint8_t out[KEYED_DRAW_MAX_BYTES]) {
    keyedSymForward_t *f = ctx;
    const roundingMode_t mode = stochastic ? SR_HALF_AWAY : HALF_AWAY;
    size_t *dims = reserveMemory(f->outRank * sizeof(size_t));
    memcpy(dims, f->outDims, f->outRank * sizeof(size_t));
    size_t *order = reserveMemory(f->outRank * sizeof(size_t));
    setOrderOfDimsForNewTensor(f->outRank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, f->outRank, order);
    tensor_t *output = initTensor(shape, quantizationInitSymInt32WithBits(mode, 8), NULL);
    if (f->forwardRounding != NULL) {
        *f->forwardRounding = mode;
    }
    const layerExec_t exec = {.key = rngKeyFromSeed(seed)};

    layerFunctions[f->layer->type].forward(f->layer, f->input, output, &exec);

    size_t n = calcNumberOfElementsByTensor(output);
    TEST_ASSERT_TRUE_MESSAGE(n >= KEYED_DRAW_MIN_ELEMENTS, "the fixture rounds >= 16 outputs");
    TEST_ASSERT_TRUE(n * sizeof(int32_t) + sizeof(float) <= KEYED_DRAW_MAX_BYTES);
    memcpy(out, output->data, n * sizeof(int32_t));
    memcpy(out + n * sizeof(int32_t), &((symInt32QConfig_t *)output->quantization->qConfig)->scale,
           sizeof(float));
    freeTensor(output);
    return n * sizeof(int32_t) + sizeof(float);
}

/* The forward's SR epilogue draws from the stream of chainState (pinned to the
 * mirror's pythonState) and is a pure function of the layer key. */
static inline void assertKeyedSymForwardDrawsFrom(keyedSymForward_t *f, uint64_t seed,
                                                  uint32_t chainState, uint32_t pythonState) {
    uint8_t bytes[KEYED_DRAW_MAX_BYTES];
    keyedCellInstall(FLOAT32, SYM_INT32);
    keyedSymForwardRun(f, seed, true, bytes);
    keyedCellRestore();
    assertKeyedCellSawOneStream(chainState, pythonState);
    assertKeyedRunIsPure(keyedSymForwardRun, f, seed);
}

#endif /* ODT_TEST_KEYED_DRAW_CHECK_H */
