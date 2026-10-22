#define SOURCE_FILE "remat_select"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CalculateGradsSequential.h"
#include "LayerConfigAccess.h"
#include "RematScheduler.h"
#include "remat_select.h"

static const char *envValue(const char *name) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? v : NULL;
}

/* The value [plan, end) is exactly word. */
static bool isWord(const char *plan, const char *end, const char *word) {
    size_t len = strlen(word);
    return (size_t)(end - plan) == len && strncmp(plan, word, len) == 0;
}

static bool isDigit(const char *p, const char *end) {
    return p < end && *p >= '0' && *p <= '9';
}

/* <j>[,<j>...] in decimal over [list, end), strictly ascending in 1..n. Every
 * digit is checked against n before the next one, so no value can wrap. */
static bool parseRetainList(const char *list, const char *end, size_t n, bool *retain) {
    size_t last = 0;
    const char *p = list;
    for (;;) {
        if (!isDigit(p, end)) {
            return false;
        }
        size_t j = 0;
        while (isDigit(p, end)) {
            j = j * 10u + (size_t)(*p - '0');
            if (j > n) {
                return false;
            }
            p++;
        }
        if (j <= last) {
            return false; /* 0, repeated, or not ascending */
        }
        retain[j] = true;
        last = j;
        if (p == end) {
            return true;
        }
        if (*p != ',') {
            return false;
        }
        p++;
    }
}

static bool selectRetain(rematSelection_t *sel, const char *what, const char *end, size_t n) {
    sel->plan.policy = REMAT_PLAN_RETAIN_LIST;
    if (isWord(what, end, "sqrt")) {
        sel->retainForm = REMAT_RETAIN_SQRT;
        return true;
    }
    if (isWord(what, end, "speed")) {
        sel->retainForm = REMAT_RETAIN_SPEED;
        return true;
    }
    sel->retainForm = REMAT_RETAIN_LIST;
    if (!parseRetainList(what, end, n, sel->retain)) {
        return false;
    }
    sel->retain[n] = true;
    return true;
}

/* The plan, then optionally "+inplace" (once, at the end): in-place
 * gradients on top of any plan. */
static bool selectPlan(rematSelection_t *sel, const char *plan, size_t modelSize) {
    static const char suffix[] = "+inplace";
    const size_t suffixLen = sizeof suffix - 1u;
    const char *end = plan + strlen(plan);
    if ((size_t)(end - plan) > suffixLen && strcmp(end - suffixLen, suffix) == 0) {
        sel->plan.inPlaceDx = true;
        end -= suffixLen;
    }
    if (isWord(plan, end, "store_all")) {
        sel->plan.policy = REMAT_PLAN_STORE_ALL;
        return true;
    }
    if (isWord(plan, end, "liveness")) {
        sel->plan.policy = REMAT_PLAN_LIVENESS;
        return true;
    }
    if (end - plan >= 7 && strncmp(plan, "retain:", 7) == 0) {
        if (modelSize > REMAT_SELECT_MAX_LAYERS) {
            fprintf(stderr,
                    "ERROR: REMAT_PLAN=%s needs a model of at most %u layers, this one has %zu\n",
                    plan, REMAT_SELECT_MAX_LAYERS, modelSize);
            return false;
        }
        if (selectRetain(sel, plan + 7, end, modelSize)) {
            return true;
        }
    }
    fprintf(stderr,
            "ERROR: REMAT_PLAN=%s (expected store_all|liveness|retain:<j>[,<j>...]|retain:sqrt|"
            "retain:speed, optionally followed by +inplace, 1 <= j <= %zu ascending)\n",
            plan, modelSize);
    return false;
}

/* Decimal digits only, 1 <= value <= SIZE_MAX. Every digit is checked before
 * it is added, so no value can wrap. */
static bool parseBudget(const char *text, size_t *bytes) {
    size_t v = 0;
    for (const char *p = text; *p != '\0'; p++) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        size_t d = (size_t)(*p - '0');
        if (v > (SIZE_MAX - d) / 10u) {
            return false;
        }
        v = v * 10u + d;
    }
    *bytes = v;
    return v != 0u;
}

/* DUMP_PROBLEM=1 alone, and REMAT_BUDGET_B as a byte count on arena storage.
 * The problem does not depend on the storage, the plan or the budget, so any
 * of them next to the dump is a mistake; heap storage reports no arena, so a
 * budget on it would pass for nothing. */
static bool selectDumpAndBudget(rematSelection_t *sel, const char *dump, const char *budget,
                                bool othersSet) {
    if (dump != NULL) {
        if (strcmp(dump, "1") != 0) {
            fprintf(stderr, "ERROR: DUMP_PROBLEM=%s (expected 1)\n", dump);
            return false;
        }
        sel->dumpProblem = true;
    }
    if (budget != NULL && !parseBudget(budget, &sel->budgetBytes)) {
        fprintf(stderr,
                "ERROR: REMAT_BUDGET_B=%s (expected the arena budget in bytes: decimal digits, "
                "1 <= value <= %zu)\n",
                budget, (size_t)SIZE_MAX);
        return false;
    }
    if (dump != NULL && (othersSet || budget != NULL)) {
        fprintf(stderr, "ERROR: DUMP_PROBLEM=1 stands alone: it writes the problem, which no "
                        "storage, plan or budget changes, and stops (unset REMAT_STORAGE, "
                        "REMAT_PLAN, DRY_PLAN and REMAT_BUDGET_B)\n");
        return false;
    }
    if (budget != NULL && sel->storage != REMAT_ARENA) {
        fprintf(stderr, "ERROR: REMAT_BUDGET_B needs REMAT_STORAGE=arena (the budget bounds the "
                        "arena; heap storage has none)\n");
        return false;
    }
    return true;
}

bool rematSelectFromEnv(rematSelection_t *sel, size_t modelSize) {
    const char *storage = envValue("REMAT_STORAGE");
    const char *plan = envValue("REMAT_PLAN");
    const char *dry = envValue("DRY_PLAN");
    const char *dump = envValue("DUMP_PROBLEM");
    const char *budget = envValue("REMAT_BUDGET_B");
    *sel = (rematSelection_t){
        .storage = REMAT_HEAP, .plan = *calculateGradsDefaultPlanSpec(), .modelSize = modelSize};
    if (storage != NULL) {
        if (strcmp(storage, "arena") == 0) {
            sel->storage = REMAT_ARENA;
        } else if (strcmp(storage, "heap") != 0) {
            fprintf(stderr, "ERROR: REMAT_STORAGE=%s (expected arena|heap)\n", storage);
            return false;
        }
    }
    if (plan != NULL && !selectPlan(sel, plan, modelSize)) {
        return false;
    }
    if (dry != NULL) {
        if (strcmp(dry, "1") != 0) {
            fprintf(stderr, "ERROR: DRY_PLAN=%s (expected 1)\n", dry);
            return false;
        }
        sel->dryPlan = true;
    }
    bool othersSet = storage != NULL || plan != NULL || dry != NULL;
    if (!selectDumpAndBudget(sel, dump, budget, othersSet)) {
        return false;
    }
    sel->chosen = othersSet || dump != NULL || budget != NULL;
    return true;
}

static bool isGemm(layerType_t type) {
    return type == LINEAR || type == CONV1D || type == CONV1D_TRANSPOSED;
}

void rematSelectRetainMask(const rematSelection_t *sel, layer_t **model, size_t modelSize,
                           bool *retain) {
    size_t k = 1;
    while (k * k < modelSize) {
        k++;
    }
    for (size_t j = 0; j <= modelSize; j++) {
        switch (sel->retainForm) {
        case REMAT_RETAIN_LIST:
            retain[j] = sel->retain[j];
            break;
        case REMAT_RETAIN_SQRT:
            retain[j] = j >= 1u && j < modelSize && j % k == 0u;
            break;
        case REMAT_RETAIN_SPEED:
            retain[j] = j >= 1u && isGemm(model[j - 1u]->type);
            break;
        }
    }
    retain[modelSize] = true;
}

static const char *storageName(rematSchedulerType_t t) {
    return t == REMAT_ARENA ? "arena" : "heap";
}

/* The plan as selected; a SEQUENCE is never selected here, but has a name. */
static void printPolicyName(FILE *f, const rematSelection_t *sel) {
    switch (sel->plan.policy) {
    case REMAT_PLAN_STORE_ALL:
        fputs("store_all", f);
        return;
    case REMAT_PLAN_LIVENESS:
        fputs("liveness", f);
        return;
    case REMAT_PLAN_SEQUENCE:
        fputs("sequence", f);
        return;
    case REMAT_PLAN_RETAIN_LIST:
        break;
    }
    switch (sel->retainForm) {
    case REMAT_RETAIN_SQRT:
        fputs("retain:sqrt", f);
        return;
    case REMAT_RETAIN_SPEED:
        fputs("retain:speed", f);
        return;
    case REMAT_RETAIN_LIST:
        break;
    }
    fputs("retain:", f);
    const char *separator = "";
    for (size_t j = 1; j <= sel->modelSize; j++) {
        if (sel->retain[j]) {
            fprintf(f, "%s%zu", separator, j);
            separator = ",";
        }
    }
}

static void printPlanName(FILE *f, const rematSelection_t *sel) {
    printPolicyName(f, sel);
    if (sel->plan.inPlaceDx) {
        fputs("+inplace", f);
    }
}

/* The fields the report's flags declare valid are the analytic numbers; a
 * did-not-run point prints the same fields with its flags. The plan is the
 * selected one, also where the report's own policy is not valid yet. */
static void printFields(FILE *f, const char *lead, const char *example, const rematSelection_t *sel,
                        const rematReport_t *r) {
    fprintf(f, "%sexample=%s storage=%s plan=", lead, example, storageName(r->type));
    printPlanName(f, sel);
    fprintf(f,
            " planned=%d placed=%d data_reserved=%d steps=%zu wires_peak_b=%zu "
            "activations_peak_b=%zu arena_b=%zu arena_pad_b=%zu arena_gap_b=%zu "
            "wire_metadata_b=%zu refwd_steps=%zu forward_cost=%zu recompute_cost=%zu\n",
            r->planned ? 1 : 0, r->placed ? 1 : 0, r->dataReserved ? 1 : 0, r->numSteps,
            r->peakLiveBytes, r->activationsPeakBytes, r->arenaBytes, r->arenaPadBytes,
            r->arenaGapBytes, r->metadataBytes, r->refwdSteps, r->forwardCost, r->recomputeCost);
}

bool rematSelectInit(rematScheduler_t *s, const rematSelection_t *sel, const char *example,
                     layer_t **model, size_t modelSize, lossConfig_t loss,
                     const tensor_t *inputLike) {
    rematPlanSpec_t spec = sel->plan;
    bool retain[REMAT_SELECT_MAX_LAYERS + 1u];
    if (spec.policy == REMAT_PLAN_RETAIN_LIST) {
        if (modelSize != sel->modelSize) {
            fprintf(stderr,
                    "ERROR: rematSelectInit: REMAT_PLAN was read for %zu layers, the model "
                    "has %zu\n",
                    sel->modelSize, modelSize);
            return false;
        }
        rematSelectRetainMask(sel, model, modelSize, retain);
        spec.retain = retain;
    }
    bool built = (sel->storage == REMAT_ARENA)
                     ? rematArenaInit(s, model, modelSize, loss, inputLike, &spec)
                     : rematHeapInit(s, model, modelSize, loss, inputLike, &spec);
    if (!built) {
        rematReport_t r;
        rematSchedulerReport(s, &r);
        printFields(stderr, "ERROR: remat scheduler init failed (did not run): ", example, sel, &r);
        rematSchedulerDeinit(s);
    }
    return built;
}

void rematSelectPrintPlan(FILE *f, const char *example, const rematSelection_t *sel,
                          const rematScheduler_t *s) {
    rematReport_t r;
    rematSchedulerReport(s, &r);
    printFields(f, "DRY_PLAN ", example, sel, &r);
}

void rematSelectPrintConfigKeys(FILE *f, const rematSelection_t *sel) {
    if (!sel->chosen) {
        return;
    }
    fprintf(f, ", \"remat_storage\": \"%s\", \"remat_plan\": \"", storageName(sel->storage));
    printPlanName(f, sel);
    fputc('"', f);
}

#define REMAT_PROBLEM_HASH_PRIME UINT64_C(0x100000001b3)
/* The problem's shapes have the input's rank, or 2 after a Flatten. */
#define REMAT_DUMP_MAX_RANK 8u

uint64_t rematSelectProblemHash(uint64_t state, const char *bytes, size_t n) {
    for (size_t i = 0; i < n; i++) {
        state ^= (uint8_t)bytes[i];
        state *= REMAT_PROBLEM_HASH_PRIME;
    }
    return state;
}

/* The layer types of the example trainers, each with a C reference run; NULL
 * for every other type. */
static const char *dumpTypeName(layerType_t type) {
    switch (type) {
    case CONV1D:
        return "CONV1D";
    case RELU:
        return "RELU";
    case MAXPOOL1D:
        return "MAXPOOL1D";
    case AVGPOOL1D:
        return "AVGPOOL1D";
    case FLATTEN:
        return "FLATTEN";
    case LINEAR:
        return "LINEAR";
    case SOFTMAX:
        return "SOFTMAX";
    default:
        return NULL;
    }
}

/* Everything the problem cannot hold is refused before a byte is written, so
 * a refused dump leaves json empty. A Flatten's dtypes are its input's. */
static bool dumpCanWrite(layer_t **model, size_t modelSize, lossConfig_t loss,
                         const tensor_t *inputLike) {
    const shape_t *in = inputLike->shape;
    if (in->numberOfDimensions == 0u || in->numberOfDimensions > REMAT_DUMP_MAX_RANK) {
        fprintf(stderr,
                "ERROR: DUMP_PROBLEM: the input has rank %zu (the problem dump writes ranks 1 to "
                "%u)\n",
                in->numberOfDimensions, REMAT_DUMP_MAX_RANK);
        return false;
    }
    if (in->dimensions[0] != 1u) {
        fprintf(stderr,
                "ERROR: DUMP_PROBLEM: the input's first dimension is %zu (the problem is written "
                "at batch 1)\n",
                in->dimensions[0]);
        return false;
    }
    if (inputLike->quantization->type != FLOAT32) {
        fprintf(stderr,
                "ERROR: DUMP_PROBLEM: the input has qtype_t %d (the problem dump writes FLOAT32 "
                "only)\n",
                (int)inputLike->quantization->type);
        return false;
    }
    if (loss.funcType != CROSS_ENTROPY && loss.funcType != MSE) {
        fprintf(stderr, "ERROR: DUMP_PROBLEM: unknown lossFuncType_t %d\n", (int)loss.funcType);
        return false;
    }
    for (size_t l = 0; l < modelSize; l++) {
        layer_t *layer = model[l];
        if (dumpTypeName(layer->type) == NULL) {
            fprintf(stderr,
                    "ERROR: DUMP_PROBLEM: layer %zu has layerType_t %d, which the problem dump "
                    "does not write (it writes CONV1D, RELU, MAXPOOL1D, AVGPOOL1D, FLATTEN, "
                    "LINEAR and SOFTMAX)\n",
                    l, (int)layer->type);
            return false;
        }
        if (layer->type == FLATTEN) {
            continue;
        }
        if (layerOutputQ(layer)->type != FLOAT32) {
            fprintf(stderr,
                    "ERROR: DUMP_PROBLEM: layer %zu has outputDtype qtype_t %d (the problem dump "
                    "writes FLOAT32 only)\n",
                    l, (int)layerOutputQ(layer)->type);
            return false;
        }
        if (backwardWireQ(layer)->type != FLOAT32) {
            fprintf(stderr,
                    "ERROR: DUMP_PROBLEM: layer %zu has propLossDtype qtype_t %d (the problem dump "
                    "writes FLOAT32 only)\n",
                    l, (int)backwardWireQ(layer)->type);
            return false;
        }
    }
    return true;
}

/* One writer feeds the stream and the hash, so the hash covers exactly the
 * bytes written. */
typedef struct problemWriter {
    FILE *f;
    uint64_t hash;
} problemWriter_t;

static void emit(problemWriter_t *w, const char *text) {
    size_t n = strlen(text);
    (void)fwrite(text, 1u, n, w->f);
    w->hash = rematSelectProblemHash(w->hash, text, n);
}

static void emitSize(problemWriter_t *w, size_t v) {
    char digits[24];
    (void)snprintf(digits, sizeof digits, "%zu", v);
    emit(w, digits);
}

static void emitShape(problemWriter_t *w, const shape_t *s) {
    emit(w, "[");
    for (size_t d = 0; d < s->numberOfDimensions; d++) {
        if (d > 0u) {
            emit(w, ",");
        }
        emitSize(w, s->dimensions[d]);
    }
    emit(w, "]");
}

/* The forward-cost class of these types is linear in the output element
 * count, so the cost of one output element is the operand: Cin/groups * K or
 * inF multiply-accumulates, or the pool window. The other types cost their
 * element count (Flatten 0) and carry no operand. */
static void emitCost(problemWriter_t *w, const layer_t *layer) {
    switch (layer->type) {
    case CONV1D:
    case LINEAR:
        emit(w, ",\"cost\":{\"macPerElement\":");
        break;
    case MAXPOOL1D:
    case AVGPOOL1D:
        emit(w, ",\"cost\":{\"kernel\":");
        break;
    default:
        return;
    }
    emitSize(w, layerForwardCost(layer, 0u, 1u));
    emit(w, "}");
}

/* Only a BatchNorm1d tracks running statistics; the dump writes none. */
static void emitLayer(problemWriter_t *w, const layer_t *layer, const shape_t *out) {
    emit(w, "{\"type\":\"");
    emit(w, dumpTypeName(layer->type));
    emit(w, layerIsFrozen(layer) ? "\",\"frozen\":true" : "\",\"frozen\":false");
    emit(w, ",\"trackRunningStats\":false,\"outShape\":");
    emitShape(w, out);
    emit(w, ",\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"}");
    emitCost(w, layer);
    emit(w, "}");
}

bool rematSelectDumpProblem(FILE *json, FILE *hashLine, layer_t **model, size_t modelSize,
                            lossConfig_t loss, const tensor_t *inputLike) {
    if (!dumpCanWrite(model, modelSize, loss, inputLike)) {
        return false;
    }
    problemWriter_t w = {.f = json, .hash = REMAT_PROBLEM_HASH_BASIS};
    emit(&w, "{\"schema_version\":0,\"input\":{\"shape\":");
    emitShape(&w, inputLike->shape);
    emit(&w, ",\"dtype\":{\"kind\":\"FLOAT32\"}},\"loss\":\"");
    emit(&w, loss.funcType == CROSS_ENTROPY ? "CROSS_ENTROPY" : "MSE");
    emit(&w, "\",\"layers\":[");
    size_t dims[2][REMAT_DUMP_MAX_RANK];
    size_t order[2][REMAT_DUMP_MAX_RANK];
    shape_t scratch[2] = {{.dimensions = dims[0], .orderOfDimensions = order[0]},
                          {.dimensions = dims[1], .orderOfDimensions = order[1]}};
    shape_t *prev = inputLike->shape;
    for (size_t l = 0; l < modelSize; l++) {
        layer_t *layer = model[l];
        shape_t *out = &scratch[l % 2u];
        /* As the remat wire table: a Flatten writes [batch, features]. */
        out->numberOfDimensions = (layer->type == FLATTEN) ? 2u : prev->numberOfDimensions;
        layerFunctions[layer->type].calcOutputShape(layer, prev, out);
        if (l > 0u) {
            emit(&w, ",");
        }
        emitLayer(&w, layer, out);
        prev = out;
    }
    emit(&w, "]}\n");
    if (fflush(json) != 0 || ferror(json)) {
        fprintf(stderr, "ERROR: DUMP_PROBLEM: writing the problem failed\n");
        return false;
    }
    return fprintf(hashLine, "PROBLEM_HASH fnv1a64=%016" PRIx64 "\n", w.hash) > 0 &&
           fflush(hashLine) == 0;
}
