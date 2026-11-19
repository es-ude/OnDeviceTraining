#define SOURCE_FILE "REMAT_DERIVE"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Common.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LossFunction.h"
#include "Model.h"
#include "OpTable.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"

static void setSlot(uint16_t *dxWire, uint8_t *dxMode, size_t slot, uint16_t wire, opMode_t mode) {
    if (dxWire != NULL) {
        dxWire[slot] = wire;
        dxMode[slot] = (uint8_t)mode;
    }
}

/* The seed GRAD n: under CrossEntropy with a BFP-declaring head
 * (crossEntropyHeadDeclaresBfp) it IS the last Softmax's dx wire (the combined gradient replaces
 * its backward), so its template is the Softmax's propLossQ, type-derived. Every other seed -- MSE,
 * and CrossEntropy on a FLOAT32 / SYM head, whose propLossQ stays unused -- inherits ACT n, the
 * model output it is the gradient of. A Flatten passes its dx through, so its GRAD inherits the
 * ACT it is the gradient of. */
size_t rematDeriveGradWires(const model_t *m, lossFuncType_t lt, const uint8_t *needsGrad,
                            const uint8_t *runsBackward, rematWireFact_t *facts, uint16_t *dxWire,
                            uint8_t *dxMode) {
    size_t n = m->numLayers;
    size_t count = 0;
    bool hasBackward = needsGrad[lt == CROSS_ENTROPY ? n - 1u : n] != 0;
    if (hasBackward) {
        if (facts != NULL) {
            bool seedTakesPropLossQ =
                lt == CROSS_ENTROPY && crossEntropyHeadDeclaresBfp(m->layers[n - 1u]);
            facts[n + 1u] =
                (rematWireFact_t){.kind = REMAT_WIRE_GRAD,
                                  .index = (uint16_t)n,
                                  .inheritFrom = seedTakesPropLossQ ? REMAT_NONE : (uint16_t)n};
        }
        count++;
    }
    for (size_t j = n; j-- > 0;) {
        uint16_t ids[MODEL_MAX_ARITY];
        uint8_t k = modelNodeInputs(m, j, ids);
        for (size_t i = 0; i < MODEL_MAX_ARITY; i++) {
            size_t slot = j * MODEL_MAX_ARITY + i;
            if (i >= k || !runsBackward[j]) {
                setSlot(dxWire, dxMode, slot, REMAT_NONE, OP_MODE_READ);
                continue;
            }
            uint16_t b = ids[i];
            if (!needsGrad[b]) {
                setSlot(dxWire, dxMode, slot, REMAT_NONE, OP_MODE_WRITE);
                continue;
            }
            size_t id = n + 1u + count++;
            if (facts != NULL) {
                bool passThrough = m->layers[j]->type == FLATTEN;
                facts[id] = (rematWireFact_t){.kind = REMAT_WIRE_GRAD,
                                              .index = b,
                                              .inheritFrom = passThrough ? b : REMAT_NONE};
            }
            setSlot(dxWire, dxMode, slot, (uint16_t)id, OP_MODE_WRITE);
        }
    }
    return count;
}
