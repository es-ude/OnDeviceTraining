#ifndef EXAMPLES_SHARED_REMAT_SELECT_H
#define EXAMPLES_SHARED_REMAT_SELECT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "Layer.h"
#include "LossFunction.h"
#include "RematScheduler.h"
#include "Tensor.h"

/* The examples' start-up choice of the training call's memory scheme (#4).
 * REMAT_STORAGE (arena | heap) picks the storage scheme, REMAT_PLAN
 * (store_all | liveness) the plan, DRY_PLAN=1 prints the plan and stops.
 * With none of them set a trainer passes no scheduler, so the training call
 * runs its own default plan. A trainer keeps the scheduler on main's stack
 * and deinits it at the end. */
typedef struct rematSelection {
    bool chosen;  /* any of the three variables is set */
    bool dryPlan; /* DRY_PLAN=1 */
    rematSchedulerType_t storage;
    rematPlanSpec_t plan;
} rematSelection_t;

/* A missing REMAT_STORAGE means heap, a missing REMAT_PLAN the training
 * call's default plan; an empty value counts as missing, as for LOG_PATH and
 * BIT_PARITY. An unknown value prints the allowed ones to stderr and returns
 * false; call it before any data is loaded. */
bool rematSelectFromEnv(rematSelection_t *sel);

/* Builds the chosen scheduler into s, keyed to inputLike (the [1, ...sample
 * shape] view of one training sample). A failed reservation prints the
 * did-not-run line to stderr, deinits s and returns false. */
bool rematSelectInit(rematScheduler_t *s, const rematSelection_t *sel, const char *example,
                     layer_t **model, size_t modelSize, lossConfig_t loss,
                     const tensor_t *inputLike);

/* The one-line plan report of DRY_PLAN=1. */
void rematSelectPrintPlan(FILE *f, const char *example, const rematScheduler_t *s);

/* The run log's config keys of a chosen scheme, each preceded by ", "
 * ("remat_storage", "remat_plan"); nothing when none was chosen. */
void rematSelectPrintConfigKeys(FILE *f, const rematSelection_t *sel);

#endif // EXAMPLES_SHARED_REMAT_SELECT_H
