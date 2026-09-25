#ifndef ODT_REMAT_PLAN_H
#define ODT_REMAT_PLAN_H

#include <stddef.h>

#include "Layer.h"
#include "LossFunction.h"

/* Internal to the remat libraries (#4, spec §3-§4): the row inits, the
 * dispatch, RematCheck and tests include it. Placement-free: no byte offset
 * lives here (ARENA keeps its offsets privately, spec §5.5). */

/* deepest = deepestTrainableIndex (n = nothing trains); top = n-1, or n-2
 * under CROSS_ENTROPY: the positional rule of CalculateGradsSequential.c:77-80
 * in SIGNED arithmetic (n == 1 under CE gives -1, D20). One shared function:
 * the plan uses it on the built model, the checker on the live one. */
void rematBackwardRange(layer_t **model, size_t n, lossFuncType_t lt, size_t *deepest,
                        ptrdiff_t *top);

#endif // ODT_REMAT_PLAN_H
