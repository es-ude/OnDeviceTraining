#define SOURCE_FILE "BATCH_VIEW"

#include <stdlib.h>

#include "BatchView.h"
#include "Common.h"

tensor_t *batchViewOf(batchView_t *view, tensor_t *sample) {
    size_t rank = sample->shape->numberOfDimensions;
    /* `rank >= MAX` is `rank + 1 > MAX` without the size_t wrap on a garbage rank. */
    if (rank >= BATCH_VIEW_MAX_RANK) {
        PRINT_ERROR("batchViewOf: sample rank %zu plus the batch axis exceeds "
                    "BATCH_VIEW_MAX_RANK (%d)",
                    rank, BATCH_VIEW_MAX_RANK);
        exit(1);
    }

    view->dimensions[0] = 1;
    view->orderOfDimensions[0] = 0;
    for (size_t i = 0; i < rank; i++) {
        view->dimensions[i + 1] = sample->shape->dimensions[i];
        view->orderOfDimensions[i + 1] = sample->shape->orderOfDimensions[i] + 1;
    }
    view->shape.numberOfDimensions = rank + 1;
    view->shape.dimensions = view->dimensions;
    view->shape.orderOfDimensions = view->orderOfDimensions;

    view->tensor.data = sample->data;
    view->tensor.shape = &view->shape;
    view->tensor.quantization = sample->quantization;
    view->tensor.sparsity = sample->sparsity;
    return &view->tensor;
}
