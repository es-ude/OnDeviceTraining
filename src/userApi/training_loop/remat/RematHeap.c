#define SOURCE_FILE "REMAT_HEAP"

#include <stdbool.h>
#include <stddef.h>

#include "Layer.h"
#include "LossFunction.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "Tensor.h"

bool rematHeapInit(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss,
                   const tensor_t *inputLike, const rematPlanSpec_t *spec) {
    *s = (rematScheduler_t){.type = REMAT_HEAP};
    return rematWireTableInit(&s->wires, model, n, loss, inputLike) &&
           rematPlanBuild(&s->plan, s->wires, model, spec);
}
