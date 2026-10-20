#define SOURCE_FILE "remat_select"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CalculateGradsSequential.h"
#include "RematScheduler.h"
#include "remat_select.h"

static const char *envValue(const char *name) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? v : NULL;
}

/* <j>[,<j>...] in decimal, strictly ascending in 1..n. Every digit is
 * checked against n before the next one, so no value can wrap. */
static bool parseRetainList(const char *list, size_t n, bool *retain) {
    size_t last = 0;
    const char *p = list;
    for (;;) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        size_t j = 0;
        while (*p >= '0' && *p <= '9') {
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
        if (*p == '\0') {
            return true;
        }
        if (*p != ',') {
            return false;
        }
        p++;
    }
}

static bool selectRetain(rematSelection_t *sel, const char *what, size_t n) {
    sel->plan.policy = REMAT_PLAN_RETAIN_LIST;
    if (strcmp(what, "sqrt") == 0) {
        sel->retainForm = REMAT_RETAIN_SQRT;
        return true;
    }
    if (strcmp(what, "speed") == 0) {
        sel->retainForm = REMAT_RETAIN_SPEED;
        return true;
    }
    sel->retainForm = REMAT_RETAIN_LIST;
    if (!parseRetainList(what, n, sel->retain)) {
        return false;
    }
    sel->retain[n] = true;
    return true;
}

static bool selectPlan(rematSelection_t *sel, const char *plan, size_t modelSize) {
    if (strcmp(plan, "store_all") == 0) {
        sel->plan.policy = REMAT_PLAN_STORE_ALL;
        return true;
    }
    if (strcmp(plan, "liveness") == 0) {
        sel->plan.policy = REMAT_PLAN_LIVENESS;
        return true;
    }
    if (strncmp(plan, "retain:", 7) == 0) {
        if (modelSize > REMAT_SELECT_MAX_LAYERS) {
            fprintf(stderr,
                    "ERROR: REMAT_PLAN=%s needs a model of at most %u layers, this one has %zu\n",
                    plan, REMAT_SELECT_MAX_LAYERS, modelSize);
            return false;
        }
        if (selectRetain(sel, plan + 7, modelSize)) {
            return true;
        }
    }
    fprintf(stderr,
            "ERROR: REMAT_PLAN=%s (expected store_all|liveness|retain:<j>[,<j>...]|retain:sqrt|"
            "retain:speed, 1 <= j <= %zu ascending)\n",
            plan, modelSize);
    return false;
}

bool rematSelectFromEnv(rematSelection_t *sel, size_t modelSize) {
    const char *storage = envValue("REMAT_STORAGE");
    const char *plan = envValue("REMAT_PLAN");
    const char *dry = envValue("DRY_PLAN");
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
    sel->chosen = storage != NULL || plan != NULL || dry != NULL;
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
static void printPlanName(FILE *f, const rematSelection_t *sel) {
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
