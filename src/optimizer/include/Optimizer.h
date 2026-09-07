#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Layer.h"
#include "Quantization.h"
#include "Tensor.h"

typedef struct sgd sgd_t;
typedef struct adamW adamW_t;

typedef struct states {
    tensor_t **stateBuffers;
    size_t statesPerParameter;
} states_t;

typedef union optimImpl {
    sgd_t *sgd;
    adamW_t *adamW;
} optimImpl_t;

typedef enum { SGD_M, ADAM_W } optimizerType_t;

typedef struct optimizer {
    optimizerType_t type;
    optimImpl_t *impl;
    parameter_t **parameter;
    states_t **states;
    size_t sizeStates;
    /* #279: rounding for TRAINING write-backs -- the step functions set this
     * as the update ops' operation-owned arithmetic.roundingMode (#282), so
     * every OUT_WRITE requant of param/state storage rounds with THIS mode
     * instead of the tensor's own qConfig roundingMode (which stays
     * authoritative for storage/inference encodes and is what serialization
     * persists). Factories default SR_HALF_AWAY so sub-ULP updates on
     * quantized storage escape the fixed-scale dead-zone in expectation;
     * deterministic HALF_AWAY is an explicit opt-out. Zero-init yields
     * HALF_AWAY (enum value 0, pinned by the serial format) -- hand-assembled
     * optimizers must set the field explicitly. */
    roundingMode_t writeBackRounding;
} optimizer_t;

typedef void (*stepFn_t)(optimizer_t *optim);
typedef void (*zeroFn_t)(optimizer_t *optim);
/* #327: optimizer-agnostic LR access for the scheduler. LR stays in the impl
 * structs; these accessors are the only sanctioned cross-impl path. */
typedef float (*getLrFn_t)(optimizer_t *optim);
typedef void (*setLrFn_t)(optimizer_t *optim, float learningRate);

typedef struct optimizerFunctions {
    stepFn_t step;
    zeroFn_t zero;
    getLrFn_t getLr;
    setLrFn_t setLr;
} optimizerFunctions_t;

extern optimizerFunctions_t optimizerFunctions[];

/* optimizer-agnostic: reads only optim->parameter; every vtable row points
 * here unless an impl needs custom zeroing. */
void optimizerZeroGrad(optimizer_t *optimizer);

/* #279: the explicit opt-out from the factories' seeded-SR training
 * write-back default. Call with HALF_AWAY for deterministic write-backs
 * (bit-parity twins, storage round-trip tests) -- stating it in one call
 * instead of silently inheriting a non-learning default is the point. */
void optimizerSetWriteBackRounding(optimizer_t *optimizer, roundingMode_t writeBackRounding);

/* THE public step entry: runs the vtable step bracketed by the
 * ODT_EVENT_OPTIMIZER_BEGIN/END phase-hook events (OdtHook.h). Reach for
 * this -- not optimizerFunctions[type].step() -- from any training loop or
 * benchmark that an external profiler may observe: the raw vtable call runs
 * the identical update but fires no events. Grad zeroing stays a separate
 * call (optimizerFunctions[type].zero / optimizerZeroGrad); it is not part
 * of the measured phase. trainingEpochDefault (hence trainingRun) steps
 * through here. */
void optimizerStep(optimizer_t *optimizer);

size_t calcTotalNumberOfStates(layer_t **model, size_t sizeModel);

#endif // OPTIMIZER_H
