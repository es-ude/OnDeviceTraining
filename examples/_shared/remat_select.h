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
 * REMAT_STORAGE (arena | heap) picks the storage scheme, REMAT_PLAN the plan,
 * DRY_PLAN=1 prints the plan and stops. REMAT_PLAN is store_all | liveness, or
 * a recompute plan that keeps some activations and recomputes the others in
 * the backward pass: retain:<j>[,<j>...] keeps the outputs ACT j of the
 * listed layers (1 <= j <= n, ascending); retain:sqrt keeps ACT j for every
 * j < n divisible by k = ceil(sqrt(n)) (Chen et al. 2016, "Training deep nets
 * with sublinear memory cost"); retain:speed keeps the output of every Linear,
 * Conv1d and Conv1dTransposed layer (the outputs of the expensive ops, the
 * "speed" choice of the cybertronai gradient-checkpointing package). A
 * recompute plan always keeps ACT n too: the loss reads it right after the
 * last forward writes it, so dropping it saves nothing. Any of them may end
 * in +inplace (liveness+inplace, retain:3,7+inplace): a ReLU, Dropout or
 * Flatten backward then writes its gradient into the bytes of the gradient
 * it reads, where both have one dtype (FLOAT32 or SYM_INT32) and one size;
 * the gradients stay the same bytes, the peak may fall. With none of the
 * variables set a trainer passes no scheduler, so the training call runs its
 * own default plan. A trainer keeps the scheduler on main's stack and deinits
 * it at the end. */
#define REMAT_SELECT_MAX_LAYERS 64u

typedef enum rematRetainForm {
    REMAT_RETAIN_LIST = 0, /* retain:<j>[,<j>...] */
    REMAT_RETAIN_SQRT,     /* retain:sqrt */
    REMAT_RETAIN_SPEED,    /* retain:speed */
} rematRetainForm_t;

typedef struct rematSelection {
    bool chosen;        /* any of the five variables is set */
    bool dryPlan;       /* DRY_PLAN=1 */
    bool dumpProblem;   /* DUMP_PROBLEM=1 */
    size_t budgetBytes; /* REMAT_BUDGET_B; 0 = unset */
    rematSchedulerType_t storage;
    rematPlanSpec_t plan;         /* plan.retain stays NULL: rematSelectInit resolves the mask */
    size_t modelSize;             /* the n rematSelectFromEnv checked the plan against */
    rematRetainForm_t retainForm; /* REMAT_PLAN_RETAIN_LIST only */
    bool retain[REMAT_SELECT_MAX_LAYERS + 1u]; /* retain:<j>...: the listed ACTs and ACT n */
} rematSelection_t;

/* A missing REMAT_STORAGE means heap, a missing REMAT_PLAN the training
 * call's default plan, a missing REMAT_BUDGET_B no budget; an empty value of
 * any of the five variables counts as missing, as for LOG_PATH and
 * BIT_PARITY. An unknown value, a retain list that is not ascending in
 * 1..modelSize, a recompute plan for a model of more than
 * REMAT_SELECT_MAX_LAYERS layers, a REMAT_BUDGET_B that is not a byte count in
 * 1..SIZE_MAX or comes without REMAT_STORAGE=arena, and a DUMP_PROBLEM=1 next
 * to any other of the variables print the reason to stderr and return false;
 * call it before any data is loaded, with the model's layer count. */
bool rematSelectFromEnv(rematSelection_t *sel, size_t modelSize);

/* The mask a recompute plan gets, modelSize + 1 entries: retain:<j>... as
 * listed, retain:sqrt and retain:speed resolved against the model (speed
 * reads each layer's type only); retain[modelSize] is always true. */
void rematSelectRetainMask(const rematSelection_t *sel, layer_t **model, size_t modelSize,
                           bool *retain);

/* Builds the chosen scheduler into s, keyed to inputLike (the [1, ...sample
 * shape] view of one training sample). A failed reservation prints the
 * did-not-run line to stderr, deinits s and returns false. A recompute plan
 * read for another layer count prints the mismatch and returns false before
 * s is touched. */
bool rematSelectInit(rematScheduler_t *s, const rematSelection_t *sel, const char *example,
                     layer_t **model, size_t modelSize, lossConfig_t loss,
                     const tensor_t *inputLike);

/* The one-line plan report of DRY_PLAN=1: the plan as selected, then the
 * scheduler's numbers. */
void rematSelectPrintPlan(FILE *f, const char *example, const rematSelection_t *sel,
                          const rematScheduler_t *s);

/* The run log's config keys of a chosen scheme, each preceded by ", "
 * ("remat_storage", "remat_plan"); nothing when none was chosen. A retain
 * list is logged with ACT n added (retain:3,7,12); the presets by name. */
void rematSelectPrintConfigKeys(FILE *f, const rematSelection_t *sel);

#endif // EXAMPLES_SHARED_REMAT_SELECT_H
