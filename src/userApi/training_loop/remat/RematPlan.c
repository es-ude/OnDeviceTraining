#define SOURCE_FILE "REMAT_PLAN"

#include <stddef.h>

#include "Layer.h"
#include "LossFunction.h"
#include "RematPlan.h"

void rematBackwardRange(layer_t **model, size_t n, lossFuncType_t lt, size_t *deepest,
                        ptrdiff_t *top) {
    *deepest = deepestTrainableIndex(model, n);
    *top = (ptrdiff_t)n - 1 - (lt == CROSS_ENTROPY ? 1 : 0);
}
