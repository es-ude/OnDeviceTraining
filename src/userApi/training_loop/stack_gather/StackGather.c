#define SOURCE_FILE "STACK_GATHER"

#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "StackGather.h"
#include "StorageApi.h"

uint8_t *stackGatherReserveBuffer(const char *caller, size_t m, size_t perSampleBytes,
                                  const char *what) {
    if (perSampleBytes != 0 && m > SIZE_MAX / perSampleBytes) {
        PRINT_ERROR("%s: the %s gather buffer (microBatchSize %zu x %zu bytes "
                    "per sample) overflows size_t",
                    caller, what, m, perSampleBytes);
        exit(1);
    }
    uint8_t *buffer = reserveMemory(m * perSampleBytes);
    if (buffer == NULL) {
        PRINT_ERROR("%s: reserving the %s gather buffer failed (microBatchSize "
                    "%zu x %zu bytes per sample)",
                    caller, what, m, perSampleBytes);
        exit(1);
    }
    return buffer;
}

void stackGatherRequireFloat32Model(const char *caller, layer_t **model, size_t modelSize, size_t m,
                                    nonFloat32FieldFn_t fieldFn) {
    for (size_t i = 0; i < modelSize; i++) {
        const char *field = fieldFn(model[i]);
        if (field != NULL) {
            PRINT_ERROR("%s: microBatchSize %zu > 1 is FLOAT32-only, but layer "
                        "%zu (layerType_t %d) has a non-FLOAT32 %s",
                        caller, m, i, (int)model[i]->type, field);
            exit(1);
        }
    }
}

void stackGatherRequireStackable(const char *caller, tensor_t *reference, tensor_t *t,
                                 const char *what, size_t sampleIndex, size_t m) {
    if (t->quantization->type != FLOAT32) {
        PRINT_ERROR("%s: microBatchSize %zu > 1 is FLOAT32-only, but the %s of "
                    "sample %zu has dtype %d",
                    caller, m, what, sampleIndex, (int)t->quantization->type);
        exit(1);
    }
    if (t->sparsity != NULL) {
        PRINT_ERROR("%s: microBatchSize %zu > 1 cannot stack the %s of sample "
                    "%zu: it carries sparsity",
                    caller, m, what, sampleIndex);
        exit(1);
    }
    size_t rank = reference->shape->numberOfDimensions;
    if (t->shape->numberOfDimensions != rank) {
        PRINT_ERROR("%s: the %s of sample %zu has rank %zu, sample 0 has rank "
                    "%zu -- a stacked chunk needs shape-identical samples",
                    caller, what, sampleIndex, t->shape->numberOfDimensions, rank);
        exit(1);
    }
    for (size_t d = 0; d < rank; d++) {
        if (t->shape->dimensions[d] != reference->shape->dimensions[d] ||
            t->shape->orderOfDimensions[d] != reference->shape->orderOfDimensions[d]) {
            PRINT_ERROR("%s: the %s of sample %zu differs from sample 0 in "
                        "dimension %zu (size %zu vs %zu, order %zu vs %zu) -- a stacked chunk "
                        "needs shape-identical samples",
                        caller, what, sampleIndex, d, t->shape->dimensions[d],
                        reference->shape->dimensions[d], t->shape->orderOfDimensions[d],
                        reference->shape->orderOfDimensions[d]);
            exit(1);
        }
    }
}
