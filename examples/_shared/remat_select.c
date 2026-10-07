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

bool rematSelectFromEnv(rematSelection_t *sel) {
    const char *storage = envValue("REMAT_STORAGE");
    const char *plan = envValue("REMAT_PLAN");
    const char *dry = envValue("DRY_PLAN");
    *sel = (rematSelection_t){.storage = REMAT_HEAP, .plan = *calculateGradsDefaultPlanSpec()};
    if (storage != NULL) {
        if (strcmp(storage, "arena") == 0) {
            sel->storage = REMAT_ARENA;
        } else if (strcmp(storage, "heap") != 0) {
            fprintf(stderr, "ERROR: REMAT_STORAGE=%s (expected arena|heap)\n", storage);
            return false;
        }
    }
    if (plan != NULL) {
        if (strcmp(plan, "store_all") == 0) {
            sel->plan.policy = REMAT_PLAN_STORE_ALL;
        } else if (strcmp(plan, "liveness") == 0) {
            sel->plan.policy = REMAT_PLAN_LIVENESS;
        } else {
            fprintf(stderr, "ERROR: REMAT_PLAN=%s (expected store_all|liveness)\n", plan);
            return false;
        }
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

static const char *storageName(rematSchedulerType_t t) {
    return t == REMAT_ARENA ? "arena" : "heap";
}

static const char *planName(rematPlanPolicy_t p) {
    return p == REMAT_PLAN_LIVENESS ? "liveness" : "store_all";
}

/* The fields the report's flags declare valid are the analytic numbers; a
 * did-not-run point prints the same fields with its flags. */
static void printFields(FILE *f, const char *lead, const char *example, const rematReport_t *r) {
    fprintf(f,
            "%sexample=%s storage=%s plan=%s planned=%d placed=%d data_reserved=%d steps=%zu "
            "wires_peak_b=%zu activations_peak_b=%zu arena_b=%zu arena_pad_b=%zu "
            "arena_gap_b=%zu wire_metadata_b=%zu\n",
            lead, example, storageName(r->type), planName(r->policy), r->planned ? 1 : 0,
            r->placed ? 1 : 0, r->dataReserved ? 1 : 0, r->numSteps, r->peakLiveBytes,
            r->activationsPeakBytes, r->arenaBytes, r->arenaPadBytes, r->arenaGapBytes,
            r->metadataBytes);
}

bool rematSelectInit(rematScheduler_t *s, const rematSelection_t *sel, const char *example,
                     layer_t **model, size_t modelSize, lossConfig_t loss,
                     const tensor_t *inputLike) {
    bool built = (sel->storage == REMAT_ARENA)
                     ? rematArenaInit(s, model, modelSize, loss, inputLike, &sel->plan)
                     : rematHeapInit(s, model, modelSize, loss, inputLike, &sel->plan);
    if (!built) {
        rematReport_t r;
        rematSchedulerReport(s, &r);
        if (!r.planned) {
            /* No plan was made, so the report's policy is not valid: name the
             * requested one. */
            r.policy = sel->plan.policy;
        }
        printFields(stderr, "ERROR: remat scheduler init failed (did not run): ", example, &r);
        rematSchedulerDeinit(s);
    }
    return built;
}

void rematSelectPrintPlan(FILE *f, const char *example, const rematScheduler_t *s) {
    rematReport_t r;
    rematSchedulerReport(s, &r);
    printFields(f, "DRY_PLAN ", example, &r);
}

void rematSelectPrintConfigKeys(FILE *f, const rematSelection_t *sel) {
    if (!sel->chosen) {
        return;
    }
    fprintf(f, ", \"remat_storage\": \"%s\", \"remat_plan\": \"%s\"", storageName(sel->storage),
            planName(sel->plan.policy));
}
