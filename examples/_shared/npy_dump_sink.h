#ifndef EXAMPLES_SHARED_NPY_DUMP_SINK_H
#define EXAMPLES_SHARED_NPY_DUMP_SINK_H

#include <stddef.h>

#include "Layer.h"
#include "Tensor.h"

/* Context for npyDumpSink. probeNames[layerIdx] gives the manifest probe name;
 * layerIdx == numProbes (the loss-gradient probe) is named "loss". Files are
 * written to <dir>/<probe>.<phase>.npy as FLOAT32, or
 * <dir>/<probe>.<phase>.s<NN>.npy when sampleIdx != NPY_DUMP_NO_SAMPLE (used for
 * the per-sample activation / act-grad tiers). The harness sets sampleIdx before
 * each per-sample tracedGrads call and resets it to NPY_DUMP_NO_SAMPLE before the
 * batch-level param/grad dumps. */
#define NPY_DUMP_NO_SAMPLE ((size_t)-1)

typedef struct npyDumpCtx {
    const char *dir;
    const char **probeNames;
    size_t numProbes;
    size_t sampleIdx; /* NPY_DUMP_NO_SAMPLE for batch-level (param/grad) dumps */
} npyDumpCtx_t;

/* Matches traceSink_t. Writes float32 .npy for ANY storage dtype except BOOL:
 * FLOAT32 verbatim, everything else (SYM / ASYM / SYM_INT32 / INT32 / BFP)
 * dequantized through convertTensor into a sink-owned FLOAT32 scratch first
 * (#417). BOOL hard-errors (exit 1) -- no float dequant exists for it. */
void npyDumpSink(void *ctx, size_t layerIdx, layerType_t layerType, const char *phase,
                 tensor_t *tensor);

#endif
