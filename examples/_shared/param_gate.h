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
 * <= 1. Worked HAR table lives in the .c.
 * Precondition (fail-fast, exit 1): N > 0, outCh > 0, N % outCh == 0 -- N is
 * the element count and outCh the dim-0 size of one non-empty tensor, which
 * holds by construction for every real weight tensor. */
groupShape_t resolveGroupShape(size_t N, size_t outCh, groupModeSweep_t mode, int groupSizeEnv);

/* Wire-block policy of the BFP sweep (spec 2026-09-14 §3.2). One knob for ALL
 * wires; every forward output wire and every dx wire resolves against its OWN
 * element count, so a layer's out and dx (its input size) may resolve
 * differently and two wires with equal N always resolve identically (the
 * packed-transparent layers' grid checks rely on that).
 *   FLOAT:  caller keeps a FLOAT32 template; returns {1,0} for uniformity.
 *   TENSOR: {1,0}.
 *   SIZE g: {N/g, g} when g divides N and N/g > 1; {1,0} when g == N (one
 *           group IS per-tensor, {1,N} is not a valid spelling); {1,0}
 *           FALLBACK when g does not divide N -- the HAR head wires (6
 *           elements) hit this for every int block; callers record it.
 * Fail-fast (exit 1): N == 0, or size <= 0 under SIZE. */
typedef enum wireBlockSweep {
    WIRE_BLOCK_FLOAT,
    WIRE_BLOCK_TENSOR,
    WIRE_BLOCK_SIZE
} wireBlockSweep_t;

groupShape_t resolveWireShape(size_t N, wireBlockSweep_t mode, int size);

/* SYM / ASYM / BFP only; any other dtype fails fast (exit 1) -- it carries no
 * group shape. */
qShapeView_t viewQShape(const quantization_t *q);

/* true when `tensor`'s storage matches `expect`; on false, `msg` (msgLen > 0
 * bytes; always NUL-terminated) carries the first mismatch, e.g.
 * "expected BFP, got SYM" / "expected mantissaBits 8, got 4" /
 * "expected group shape {4,3}, got {1,0}". Expectation dtypes without an
 * arm (INT32, SYM_INT32, BOOL) fail fast (exit 1) -- this is checked FIRST,
 * before comparing against `tensor`'s actual type, so an unsupported
 * `expect->type` always exits even when the tensor's actual type differs
 * from it (the common case for this exact programmer error). */
bool paramGateCheck(const tensor_t *tensor, const paramGateExpect_t *expect, char *msg,
                    size_t msgLen);

/* Byte accounting for the sweep's memory report (spec §7.1). Payload mirrors
 * calcNumberOfBytesForData (packed widths round up per TENSOR); metadata is
 * the per-group side table the qconfig carries: BFP one u8 exponent, SYM one
 * float scale, ASYM scale + u16 zero-point. FLOAT32 carries neither. Any
 * other dtype fails fast -- it has no sweep meaning. */
size_t packedPayloadBytes(qtype_t type, uint8_t bits, size_t N);
size_t packedMetadataBytes(qtype_t type, size_t numGroups);

/* The BFP sweep's knob set (spec §3.1), parsed from the environment by
 * bfpSweepConfigFromEnv so the trainer's CONFIG line, gates and log all read
 * ONE struct. Returns NULL on success; otherwise a static message naming the
 * offending knob and its legal values (the trainer prints it and exits 1).
 * The six SYM-only knobs are not errors -- a set one is a silent
 * misconfiguration, so each sets its LEGACY_KNOB_* bit for the trainer to WARN. */
typedef enum bfpMathSweep { BFP_MATH_NATIVE, BFP_MATH_FQ } bfpMathSweep_t;
typedef enum bfpRoundingSweep { BFP_ROUNDING_SR, BFP_ROUNDING_DET } bfpRoundingSweep_t;
enum {
    LEGACY_KNOB_SYM_BITS = 1u << 0,
    LEGACY_KNOB_SYM_WIRES = 1u << 1,
    LEGACY_KNOB_WEIGHT_DTYPE = 1u << 2,
    LEGACY_KNOB_GROUP_MODE = 1u << 3,
    LEGACY_KNOB_GROUP_SIZE = 1u << 4,
    LEGACY_KNOB_SYM_ROUNDING = 1u << 5,
    LEGACY_KNOB_ODTS_ROUNDTRIP = 1u << 6
};
typedef struct bfpSweepConfig {
    uint8_t mantissaBits;
    uint8_t exponentBits;
    groupModeSweep_t weightMode;
    int weightSize;
    wireBlockSweep_t wireMode;
    int wireSize;
    bfpMathSweep_t math;
    bool bfpGrads;
    bool bfpState;
    bfpRoundingSweep_t rounding;
    unsigned ignoredLegacyKnobs;
    char weightBlockStr[16];
    char wireBlockStr[16];
} bfpSweepConfig_t;

const char *bfpSweepConfigFromEnv(bfpSweepConfig_t *out);

#endif
