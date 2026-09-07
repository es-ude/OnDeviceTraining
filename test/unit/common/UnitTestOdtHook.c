#define SOURCE_FILE "UnitTestOdtHook"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h" /* freeLinearLayerShellOnly (static inline, test support) */
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "OdtHook.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingEpochDefault.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* One shared log for hook events AND trace-sink probes: the interleaving
 * (loss backward inside BACKWARD, layer forwards inside FORWARD) is the
 * contract, not just the event count. Callbacks never assert (Unity longjmps
 * out of a callback would leak the fixture); overflow is caught by the count
 * assertion instead. */
#define MAX_LOG 64
typedef enum { LOG_HOOK, LOG_SINK } logKind_t;
typedef struct {
    logKind_t kind;
    odtEvent_t event; /* LOG_HOOK */
    char phase[16];   /* LOG_SINK */
    void *ctx;        /* LOG_HOOK: the ctx handed back by the hook */
} logEntry_t;
static logEntry_t g_log[MAX_LOG];
static size_t g_logCount;
static int g_ctxToken; /* its address is the opaque ctx under test */

static void resetLog(void) {
    g_logCount = 0;
    memset(g_log, 0, sizeof(g_log));
}

static void recordingHook(void *ctx, odtEvent_t event) {
    if (g_logCount < MAX_LOG) {
        g_log[g_logCount].kind = LOG_HOOK;
        g_log[g_logCount].event = event;
        g_log[g_logCount].ctx = ctx;
    }
    g_logCount++;
}

static void recordingSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                          tensor_t *tensor) {
    (void)ctx;
    (void)layerIdx;
    (void)type;
    (void)tensor;
    if (g_logCount < MAX_LOG) {
        g_log[g_logCount].kind = LOG_SINK;
        snprintf(g_log[g_logCount].phase, sizeof(g_log[g_logCount].phase), "%s", phase);
    }
    g_logCount++;
}

/* Asserts g_log[i] is the hook event `expected` carrying &g_ctxToken. */
static void assertHookAt(size_t i, odtEvent_t expected) {
    TEST_ASSERT_EQUAL_INT_MESSAGE(LOG_HOOK, g_log[i].kind, "expected a hook event at this index");
    TEST_ASSERT_EQUAL_INT(expected, g_log[i].event);
    TEST_ASSERT_EQUAL_PTR(&g_ctxToken, g_log[i].ctx);
}

static void assertSinkAt(size_t i, const char *phase) {
    TEST_ASSERT_EQUAL_INT_MESSAGE(LOG_SINK, g_log[i].kind, "expected a sink probe at this index");
    TEST_ASSERT_EQUAL_STRING(phase, g_log[i].phase);
}

void testFireHandsEventAndCtxToInstalledHook(void) {
    resetLog();
    odtHookSet(recordingHook, &g_ctxToken);
    odtHookFire(ODT_EVENT_OPTIMIZER_END);
    odtHookSet(NULL, NULL);

    TEST_ASSERT_EQUAL_size_t(1, g_logCount);
    assertHookAt(0, ODT_EVENT_OPTIMIZER_END);
}

void testFireIsSilentAfterSetNull(void) {
    resetLog();
    odtHookSet(recordingHook, &g_ctxToken);
    odtHookSet(NULL, NULL); /* the disable path, exercised AFTER a real install */
    odtHookFire(ODT_EVENT_FORWARD_BEGIN);

    TEST_ASSERT_EQUAL_size_t(0, g_logCount);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testFireHandsEventAndCtxToInstalledHook);
    RUN_TEST(testFireIsSilentAfterSetNull);
    return UNITY_END();
}
