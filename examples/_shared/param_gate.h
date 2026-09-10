#ifndef EXAMPLES_SHARED_PARAM_GATE_H
#define EXAMPLES_SHARED_PARAM_GATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Quantization.h"
#include "Tensor.h"

/* Storage gate for sweep binaries (#417): after a trainer requantizes its
 * parameters, every trainable weight/bias (and grad) tensor is checked
 * against the storage the run's knobs should have produced. Hosts the HAR
 * trainer's group-shape vocabulary (moved out of train_c_sym.c) so the epic
 * #410 PR7 BFP trainer shares one checker instead of a third static copy.
 * Enum names carry a _MODE_ prefix to avoid colliding with qtype_t's own
 * SYM/ASYM enumerators -- C enums share one flat namespace. */
typedef enum groupModeSweep {
    GROUP_MODE_TENSOR, /* per-tensor {1,0} */
    GROUP_MODE_CHANNEL,
    GROUP_MODE_SIZE
} groupModeSweep_t;

/* Resolved {numGroups, groupSize} for one tensor -- the shape grammar from
 * Quantization.h: per-tensor = {1,0}; grouped = {k>1, g>0, k*g==N}. {1,N}
 * is never valid and is never emitted by resolveGroupShape. */
typedef struct groupShape {
    size_t numGroups;
    size_t groupSize;
} groupShape_t;

/* dtype-generic read of a packed tensor's width + group shape. qBits is the
 * SYM/ASYM code width or the BFP mantissa width; exponentBits is BFP-only
 * (0 otherwise). */
typedef struct qShapeView {
    uint8_t qBits;
    uint8_t exponentBits;
    size_t numGroups;
    size_t groupSize;
} qShapeView_t;

/* What one tensor's storage is expected to be. type FLOAT32 = type-only
 * check (bits/exponentBits/shape ignored); SYM/ASYM = bits + shape; BFP =
 * mantissaBits (bits) + exponentBits + shape. */
typedef struct paramGateExpect {
    qtype_t type;
    uint8_t bits;
    uint8_t exponentBits;
    groupShape_t shape;
} paramGateExpect_t;

/* Per-tensor group-shape policy of the HAR sweep (#300), N = element count,
 * outCh = dim-0 size:
 *   mode=tensor:  always {1,0}.
 *   mode=channel: one group per output channel (groupSize = N/outCh).
 *   mode=size:    groupSize = groupSizeEnv if it evenly divides N, else FALL
 *                 BACK to the per-channel size for that tensor.
 * Either grouped branch collapses to {1,0} when it would leave numGroups
 * <= 1. Worked HAR table lives in the .c. */
groupShape_t resolveGroupShape(size_t N, size_t outCh, groupModeSweep_t mode, int groupSizeEnv);

/* SYM / ASYM / BFP only; any other dtype fails fast (exit 1) -- it carries no
 * group shape. */
qShapeView_t viewQShape(const quantization_t *q);

/* true when `tensor`'s storage matches `expect`; on false, `msg` (msgLen
 * bytes, always NUL-terminated) carries the first mismatch, e.g.
 * "expected BFP, got SYM" / "expected mantissaBits 8, got 4" /
 * "expected group shape {4,3}, got {1,0}". Expectation dtypes without an
 * arm (INT32, SYM_INT32, BOOL) fail fast (exit 1). */
bool paramGateCheck(const tensor_t *tensor, const paramGateExpect_t *expect, char *msg,
                    size_t msgLen);

#endif
