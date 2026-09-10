#define SOURCE_FILE "npy_dump_sink"

#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "npy_dump_sink.h"
#include "npy_writer.h"

void npyDumpSink(void *ctxV, size_t layerIdx, layerType_t layerType, const char *phase,
                 tensor_t *tensor) {
    (void)layerType;
    npyDumpCtx_t *ctx = (npyDumpCtx_t *)ctxV;

    if (tensor->quantization->type == BOOL) {
        fprintf(stderr, "npyDumpSink: BOOL tensors have no float dequant (probe %zu, phase %s)\n",
                layerIdx, phase);
        exit(1);
    }

    const char *probe = (layerIdx < ctx->numProbes) ? ctx->probeNames[layerIdx] : "loss";

    char path[512];
    if (ctx->sampleIdx == NPY_DUMP_NO_SAMPLE) {
        snprintf(path, sizeof(path), "%s/%s.%s.npy", ctx->dir, probe, phase);
    } else {
        snprintf(path, sizeof(path), "%s/%s.%s.s%03zu.npy", ctx->dir, probe, phase, ctx->sampleIdx);
    }

    /* Non-FLOAT32 storage (SYM / ASYM / SYM_INT32 / INT32 / BFP) is dequantized
     * through the conversion matrix into a sink-owned FLOAT32 scratch for the
     * duration of the write -- the sink never reads packed bytes itself
     * (arithmetic-bfp.md §5.7 packed-walk rule), and every dtype the matrix
     * learns is dumpable without touching this file. */
    tensor_t *floatScratch = NULL;
    const float *data = (const float *)tensor->data;
    if (tensor->quantization->type != FLOAT32) {
        floatScratch = initTensor(getShapeLike(tensor->shape), quantizationInitFloat(), NULL);
        convertTensor(tensor, floatScratch);
        data = (const float *)floatScratch->data;
    }

    int rc =
        npyWriteFloat32(path, data, tensor->shape->dimensions, tensor->shape->numberOfDimensions);
    if (floatScratch != NULL) {
        freeTensor(floatScratch);
    }
    if (rc != 0) {
        fprintf(stderr, "npyDumpSink: write failed for %s (rc=%d)\n", path, rc);
        exit(1);
    }
}
