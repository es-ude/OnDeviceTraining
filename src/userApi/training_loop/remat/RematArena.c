#define SOURCE_FILE "REMAT_ARENA"

#include <stdbool.h>
#include <stddef.h>

#include "Layer.h"
#include "LossFunction.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "Tensor.h"

bool rematArenaInit(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss,
                    const tensor_t *inputLike, const rematPlanSpec_t *spec) {
    *s = (rematScheduler_t){.type = REMAT_ARENA};
    if (!rematWireTableInit(&s->wires, model, n, loss, inputLike)) {
        return false;
    }
    /* The identical model reaches both: rematPlanBuild checks it against the
     * table's key. */
    return rematPlanBuild(&s->plan, s->wires, model, spec);
}
