#ifndef HAR_MEM_INSTRUMENT_H
#define HAR_MEM_INSTRUMENT_H

/* Shared memory instrumentation for the two HAR trainer binaries (train_c.c
 * FLOAT32 + train_c_sym.c packed-SYM). Both share the identical 12-layer model
 * and training loop, so the measurement + analytic sizing + JSON-emit logic
 * lives here ONCE and both binaries call it (no copy-paste).
 *
 * The heap counters (memProfile*Bytes) are no-ops returning 0 unless the whole
 * program is built with -DODT_MEM_PROFILE (PUBLIC on StorageApi). The stack
 * high-water and RSS probes work regardless. Callers gate the whole report on
 * #ifdef ODT_MEM_PROFILE so the CI bit-parity build carries zero instrumentation.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "CalculateGradsSequential.h"
#include "MemProfile.h" /* measurePeakStackBytes, memProfileRssPeakKb */
#include "Optimizer.h"
#include "Quantization.h"
#include "StorageApi.h" /* memProfileReset/Mark/CurrentBytes/PeakBytes */
#include "TrainingLoopApi.h"

/* One honest per-run memory breakdown. All *_b fields are bytes; rss is KiB.
 * A Python consumer (examples/_shared/log_schema.py memory contract) reads the
 * emitted keys verbatim, so field <-> key names must stay in lockstep with
 * memInstrumentEmitJson. */
typedef struct memReport {
    int sym_bits;              /* SYM_BITS for the sym binary; -1 for the float binary */
    const char *storage_dtype; /* "float" | "sym" | "asym" | "bfp" -- every HAR trainer sets it */

    /* Instrumented phase marks (memProfileMark deltas over the heap counter). */
    size_t dataset_b;      /* live bytes after initDataSets */
    size_t params_grads_b; /* delta across buildModel (+ requantize for sym) */
    size_t optstate_b;     /* delta across sgdMCreateOptim */

    /* Analytic categories — what an MCU deployment would actually hold. */
    size_t params_b;            /* sum of weight+bias tensor bytes (dtype-aware) */
    size_t grads_b;             /* sum of grad tensor bytes */
    size_t optstate_analytic_b; /* sum of optimizer momentum-buffer bytes */
    size_t activations_b;       /* peak concurrent activation bytes, one batch */
    size_t io_b;                /* batched input + one-hot label bytes */
    size_t pool_backward_b;     /* persistent MaxPool argmax-index buffers (backward state, #321) */
    size_t dx_peak_b;           /* worst concurrent dx ping-pong pair during backprop (#321) */
    /* Metadata side tables, each its OWN category (integrity rule: never folded
     * into a payload). group_ = the 8 param tensors' scales/exponents (was
     * config-only before PR7); grad_/optstate_ = per-tensor packed grads/states
     * (BFP knobs); wire_ = live forward-wire exponents + the peak dx pair's. */
    size_t group_overhead_b;
    size_t grad_overhead_b;
    size_t optstate_overhead_b;
    size_t wire_overhead_b;
    size_t mcu_total_b; /* sum of the eleven analytic categories */

    /* Instrumented process-level anchors. */
    size_t heap_peak_b;  /* memProfilePeakBytes() */
    size_t stack_peak_b; /* measurePeakStackBytes() on one training step */
    size_t rss_peak_kb;  /* memProfileRssPeakKb() */

    /* heap_peak_b - mcu_total_b. RECORDED, never massaged (integrity rule). */
    long reconciliation_gap_b;
} memReport_t;

/* Analytic byte sums over the optimizer's authoritative trainable-parameter
 * array (optim->parameter / optim->states). Preferred over a traceModel* walk:
 * the optimizer already holds exactly the trainable weight/bias parameters and
 * their grads + momentum buffers, and calcBytesPerTensor is dtype-aware (packed
 * SYM weights count as ceil(qBits*N/8), FLOAT32 grads as 4*N), so this measures
 * the ACTUAL storage whatever it is.
 * CAVEAT (#380): frozen layers are optimizer-invisible, so on a frozen-layer
 * model memInstrumentParamBytes UNDER-COUNTS resident params (it sums only
 * trainable ones) — for the resident total, walk traceModelWeights instead
 * (see train_c_finetune.c's sumParamBytesSink). Grad/OptState sums stay
 * correct: frozen layers genuinely have neither. */
size_t memInstrumentParamBytes(optimizer_t *optim);
size_t memInstrumentGradBytes(optimizer_t *optim);
size_t memInstrumentOptStateBytes(optimizer_t *optim);

/* One wire's storage as the trainer resolved it (spec §7.2). NULL profile
 * arrays = every wire FLOAT32 (the float/SYM/AdamW/finetune trainers). */
#define HAR_NUM_LAYERS 12
typedef struct harWireProfile {
    bool present; /* false = this wire is never allocated (conv1.dx, softmax.dx under CE) */
    qtype_t type;
    uint8_t bits;     /* 32 for FLOAT32, mantissaBits for BFP */
    size_t numGroups; /* 0 for FLOAT32 */
} harWireProfile_t;

/* HAR-classifier-specific analytic activation / IO sizing for one MICRO-batch.
 * Encodes the fixed 12-layer topology (see mem_instrument.c); the wire dtypes
 * come from the caller's profile array (NULL = every wire FLOAT32).
 *
 * Pass the MICRO-batch (concurrent samples per forward/backward), NOT the loader
 * macro-batch: trainingBatchDefault streams the macro-batch one sample at a time
 * (loss.md: dimensions[0]=B, today B=1) and accumulates grads at the optimizer,
 * so only one sample's activations are ever live. Passing the macro-batch would
 * over-count activations + IO by that factor. */
size_t memInstrumentHarActivationBytes(size_t microBatch,
                                       const harWireProfile_t out[HAR_NUM_LAYERS]);
size_t memInstrumentHarIoBytes(size_t microBatch);

/* #321: backward-only on-device state that activations_b/params_b do NOT count.
 *  - PoolBackward: the persistent INT32 MaxPool argmax-index buffers (required for
 *    the backward pass, allocated per-layer at build time). Walks the model for
 *    MAXPOOL1D layers; dtype-aware via calcBytesPerTensor.
 *  - HarDxPeak: the transient dx ping-pong — during backprop gradNext + gradCurr
 *    coexist with every forward wire; the worst concurrent PAIR of resolved dx
 *    wires, payload only (FLOAT32: relu1/pool1 = 2 x [16,128] = 16,384 B for
 *    HAR). Pass the MICRO-batch.
 *  - HarWireOverhead: the metadata (group exponents/scales) of the 12 forward
 *    wires plus that of the same peak dx pair HarDxPeak models; 0 for FLOAT32. */
size_t memInstrumentPoolBackwardBytes(layer_t **model, size_t modelSize);
size_t memInstrumentHarDxPeakBytes(size_t microBatch, const harWireProfile_t dx[HAR_NUM_LAYERS]);
size_t memInstrumentHarWireOverheadBytes(size_t microBatch,
                                         const harWireProfile_t out[HAR_NUM_LAYERS],
                                         const harWireProfile_t dx[HAR_NUM_LAYERS]);

/* Everything the stack thunk needs to run one representative training step
 * (zeroGrad -> calculateGradsSequential -> step -> zeroGrad) on one sample. */
typedef struct memStepCtx {
    layer_t **model;
    size_t modelSize;
    lossConfig_t lossConfig;
    tensor_t *input; /* one dataset sample in its natural shape; the thunk adds */
    tensor_t *label; /* the batch axis with batchViewOf, as trainingBatchDefault does */
    optimizer_t *optim;
} memStepCtx_t;

/* Runs one training step on a painted pthread stack and returns the high-water
 * bytes touched. NOTE: performs a REAL gradient + optimizer step, mutating the
 * model and momentum state — call it AFTER any output the run must preserve. */
size_t memInstrumentStackPeakBytes(memStepCtx_t *ctx, size_t stackBytes);

/* Fill mcu_total_b (sum of the eleven categories) and reconciliation_gap_b
 * from the populated fields. Fails fast if storage_dtype was never set. */
void memInstrumentFinalize(memReport_t *r);

/* Emit the bare "memory" object body: {"storage_dtype": ..., ...}. sym_bits is
 * written only when storage_dtype is not "bfp". The caller places the object
 * (writes the "memory": key and any surrounding commas). */
void memInstrumentEmitJson(FILE *f, const memReport_t *r);

/* Print the reconciliation line to stdout. Integrity: prints the gap as-is. */
void memInstrumentPrintReconciliation(const memReport_t *r);

#endif /* HAR_MEM_INSTRUMENT_H */
