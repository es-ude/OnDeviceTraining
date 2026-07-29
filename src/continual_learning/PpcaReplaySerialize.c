#define SOURCE_FILE "PPCA_REPLAY_SERIALIZE"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Deserialize.h"
#include "PpcaReplaySerialize.h"
#include "SerialWire.h"
#include "Serialize.h"
#include "Tensor.h"

#define PPCA_SERIALIZE_MAGIC "ODTR"
/* v2 (#370): embedded ODTS tensor records switched to fixed-width LE fields
 * and the ODTR scalars are LE-pinned via SerialWire — v1 checkpoints were
 * host-local artifacts, no back-compat shim. The ODTR container version
 * itself does NOT track the embedded ODTS tensor record's own wire version
 * (group-quant PR1 bumped that to v4, Task 4 to v5) --
 * peekValidateThenDeserializeTensor's SYM and ASYM arms below duplicate (a
 * subset of) Serialize.c/Deserialize.c's qConfig layouts and must stay in
 * lockstep with them by hand; see each arm's comment. */
#define PPCA_SERIALIZE_FORMAT_VERSION 2u
/* PPCA state tensors are rank 1 or 2 by construction (mean/eigvals, basis). */
#define PPCA_MAX_TENSOR_RANK 2
/* Lockstep literal with Deserialize.c's SERIAL_MAX_QCONFIG_GROUPS (same
 * value, same rationale) -- this module has no allocation sized by the
 * file's numGroups (the peek below only fseek-skips past scales[]/
 * zeroPoints[], never reserves memory for them), but an untrusted, unbounded
 * numGroups still drives an unbounded serialReadF32LE/serialReadU16LE loop
 * on a corrupt/foreign file, so the same sanity cap applies for uniform
 * reader hardening. Keep this literal equal to Deserialize.c's by hand;
 * there is no shared header both modules already depend on to hoist it into
 * (see final-review Fix 7). Renamed from PPCA_MAX_SYM_GROUPS (group-quant
 * PR4 Task 4): the v5 ASYM peek arm below reuses it too. */
#define PPCA_MAX_QCONFIG_GROUPS 65536u

void ppcaReplaySetSerialize(const ppcaReplaySet_t *set, FILE *f) {
    serialWriteBytes(PPCA_SERIALIZE_MAGIC, 4, f);
    serialWriteU32LE(PPCA_SERIALIZE_FORMAT_VERSION, f);
    serialWriteSizeAsU32LE(set->numClasses, f);
    for (size_t c = 0; c < set->numClasses; c++) {
        const ppcaReplay_t *g = set->generators[c];
        serialWriteSizeAsU32LE(g->dim, f);
        serialWriteSizeAsU32LE(g->rank, f);
        serialWriteU32LE(g->count, f);
        serialWriteF32LE(g->sigma2, f);
        serialWriteF32LE(g->totalVar, f);
        serializeTensor(g->mean, f);
        serializeTensor(g->basis, f);
        serializeTensor(g->eigvals, f);
    }
}

/* Peek one tensor record's header (shape + quantization) into locals,
 * validate against the skeleton BEFORE any overwrite, rewind, then let the
 * public deserializeTensor consume the (now trusted) record. Mirrors the
 * Serialize.c record layout field-for-field — testRoundTripPacked is the
 * drift alarm for this coupling. */
static void peekValidateThenDeserializeTensor(tensor_t *skeleton, FILE *f, const char *what) {
    long recordStart = ftell(f);
    if (recordStart < 0) {
        PRINT_ERROR("ppcaReplaySetDeserialize: stream not seekable (%s)", what);
        exit(1);
    }

    uint32_t nd = serialReadU32LE(f);
    if ((size_t)nd != skeleton->shape->numberOfDimensions || nd > PPCA_MAX_TENSOR_RANK) {
        PRINT_ERROR("ppcaReplaySetDeserialize: %s rank mismatch (file %u, skeleton %zu)", what,
                    (unsigned)nd, skeleton->shape->numberOfDimensions);
        exit(1);
    }
    size_t dims[PPCA_MAX_TENSOR_RANK];
    for (size_t i = 0; i < nd; i++) {
        dims[i] = (size_t)serialReadU32LE(f);
    }
    for (size_t i = 0; i < nd; i++) {
        (void)serialReadU32LE(f); /* orderOfDimensions: consumed, not validated */
    }
    for (size_t i = 0; i < nd; i++) {
        if (dims[i] != skeleton->shape->dimensions[i]) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s dim[%zu] mismatch (file %zu, skeleton %zu)",
                        what, i, dims[i], skeleton->shape->dimensions[i]);
            exit(1);
        }
    }

    uint8_t fileType = serialReadU8(f);
    if ((qtype_t)fileType != skeleton->quantization->type) {
        PRINT_ERROR("ppcaReplaySetDeserialize: %s dtype mismatch (file %u, skeleton %u) — "
                    "#316 guard: rebuild the skeleton with the checkpoint's storage config",
                    what, (unsigned)fileType, (unsigned)skeleton->quantization->type);
        exit(1);
    }
    switch ((qtype_t)fileType) {
    case FLOAT32:
    case INT32:
    case BOOL:
        break;
    case SYM: {
        /* v4 (group-quant PR1/PR2): numGroups/groupSize precede the scales
         * array -- duplicates (a subset of) Deserialize.c's deserializeQConfig
         * SYM arm layout (numGroups u32, groupSize u32, f32 scales[numGroups],
         * qBits u8, roundingMode u8) since this peek runs BEFORE the trusted
         * deserializeTensor call below, which is where the REAL group-aware
         * read (reallocating the skeleton's scales[] to the file's numGroups,
         * Task 5) happens. This peek never writes the skeleton, only reads
         * past the record to size the post-peek length check further down --
         * so unlike Deserialize.c it has no numGroups to reallocate, and a
         * file numGroups that differs from the skeleton's own is NOT an
         * error here (dropped, PR2): the scales loop already iterates the
         * FILE's numGroups regardless. The sentinel invariant
         * (numGroups==1 <=> groupSize==0, Quantization.h) is still checked,
         * mirroring Deserialize.c's own (unrelaxed) sentinel guard. qBits
         * MUST still match the skeleton's -- same-dtype width mismatch is
         * the #316 2x-overflow case, independent of group shape. */
        uint32_t numGroups = serialReadU32LE(f);
        symQConfig_t *skelQc = skeleton->quantization->qConfig;
        /* Task-5-style review fix: bound the untrusted file numGroups before
         * the skip loop below reads that many floats off the stream (mirrors
         * Deserialize.c's deserializeQConfig guard -- see PPCA_MAX_SYM_GROUPS
         * above). */
        if (numGroups == 0 || numGroups > PPCA_MAX_QCONFIG_GROUPS) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s SYM file numGroups %u is zero or exceeds "
                        "the %u-group sanity cap",
                        what, (unsigned)numGroups, (unsigned)PPCA_MAX_QCONFIG_GROUPS);
            exit(1);
        }
        uint32_t groupSize = serialReadU32LE(f);
        if ((numGroups == 1) != (groupSize == 0)) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s SYM file violates the "
                        "numGroups==1<=>groupSize==0 sentinel invariant (numGroups=%u, "
                        "groupSize=%u)",
                        what, (unsigned)numGroups, (unsigned)groupSize);
            exit(1);
        }
        for (uint32_t g = 0; g < numGroups; g++) {
            (void)serialReadF32LE(f); /* scales[g] */
        }
        uint8_t qBits = serialReadU8(f);
        (void)serialReadU8(f); /* roundingMode */
        if (qBits != skelQc->qBits) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s SYM qBits mismatch (file %u, skeleton "
                        "%u) — same-dtype width mismatch is the #316 2x-overflow case",
                        what, (unsigned)qBits, (unsigned)skelQc->qBits);
            exit(1);
        }
        break;
    }
    case ASYM: {
        /* v5 (group-quant PR4, Task 4): mirrors the SYM peek arm above --
         * numGroups/groupSize precede the two per-group arrays (scales THEN
         * zeroPoints, u16 LE) -- duplicates (a subset of) Deserialize.c's
         * deserializeQConfig ASYM arm layout, since this peek runs BEFORE
         * the trusted deserializeTensor call below, which does the real
         * group-aware reallocation of BOTH arrays. A file numGroups that
         * differs from the skeleton's own is NOT an error here, mirroring
         * the SYM peek's relax -- only qBits must still match the skeleton's
         * (same-dtype width mismatch is the #316 2x-overflow case). */
        uint32_t numGroups = serialReadU32LE(f);
        asymQConfig_t *skelQc = skeleton->quantization->qConfig;
        if (numGroups == 0 || numGroups > PPCA_MAX_QCONFIG_GROUPS) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s ASYM file numGroups %u is zero or exceeds "
                        "the %u-group sanity cap",
                        what, (unsigned)numGroups, (unsigned)PPCA_MAX_QCONFIG_GROUPS);
            exit(1);
        }
        uint32_t groupSize = serialReadU32LE(f);
        if ((numGroups == 1) != (groupSize == 0)) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s ASYM file violates the "
                        "numGroups==1<=>groupSize==0 sentinel invariant (numGroups=%u, "
                        "groupSize=%u)",
                        what, (unsigned)numGroups, (unsigned)groupSize);
            exit(1);
        }
        for (uint32_t g = 0; g < numGroups; g++) {
            (void)serialReadF32LE(f); /* scales[g] */
        }
        for (uint32_t g = 0; g < numGroups; g++) {
            (void)serialReadU16LE(f); /* zeroPoints[g] */
        }
        uint8_t qBits = serialReadU8(f);
        (void)serialReadU8(f); /* roundingMode */
        if (qBits != skelQc->qBits) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s ASYM qBits mismatch (file %u, skeleton %u)",
                        what, (unsigned)qBits, (unsigned)skelQc->qBits);
            exit(1);
        }
        break;
    }
    default:
        /* SYM_INT32 etc.: create() forbids these as PPCA state, so a file
         * claiming them is corrupt or foreign. */
        PRINT_ERROR("ppcaReplaySetDeserialize: %s carries non-PPCA storage dtype %u", what,
                    (unsigned)fileType);
        exit(1);
    }

    /* Post-read record-length check, sized from the PRE-overwrite skeleton.
     * Since #370 deserializeTensor fails fast on short reads itself; the
     * position arithmetic remains as the wire-drift alarm in case the peek
     * ever stops mirroring the record layout deserializeTensor consumes. */
    size_t elems = calcNumberOfElementsByShape(skeleton->shape);
    size_t expectedBytes = calcNumberOfBytesForData(skeleton->quantization, elems);
    long headerBytes = ftell(f) - recordStart;

    if (fseek(f, recordStart, SEEK_SET) != 0) {
        PRINT_ERROR("ppcaReplaySetDeserialize: rewind failed (%s)", what);
        exit(1);
    }
    deserializeTensor(skeleton, f);
    if (ftell(f) - recordStart != headerBytes + (long)expectedBytes) {
        PRINT_ERROR("ppcaReplaySetDeserialize: %s record length mismatch — truncated payload "
                    "or wire-layout drift",
                    what);
        exit(1);
    }
}

void ppcaReplaySetDeserialize(ppcaReplaySet_t *skeleton, FILE *f) {
    char magic[4];
    serialReadBytes(magic, 4, f);
    if (memcmp(magic, PPCA_SERIALIZE_MAGIC, 4) != 0) {
        PRINT_ERROR("ppcaReplaySetDeserialize: bad magic (not an ODTR checkpoint)");
        exit(1);
    }
    uint32_t version = serialReadU32LE(f);
    if (version != PPCA_SERIALIZE_FORMAT_VERSION) {
        PRINT_ERROR("ppcaReplaySetDeserialize: unsupported version %u", version);
        exit(1);
    }
    uint32_t numClasses = serialReadU32LE(f);
    if (numClasses != (uint32_t)skeleton->numClasses) {
        PRINT_ERROR("ppcaReplaySetDeserialize: numClasses mismatch (file %u, skeleton %zu)",
                    numClasses, skeleton->numClasses);
        exit(1);
    }
    for (size_t c = 0; c < skeleton->numClasses; c++) {
        ppcaReplay_t *g = skeleton->generators[c];
        uint32_t dim = serialReadU32LE(f);
        uint32_t rank = serialReadU32LE(f);
        uint32_t count = serialReadU32LE(f);
        float sigma2 = serialReadF32LE(f);
        float totalVar = serialReadF32LE(f);
        if (dim != (uint32_t)g->dim || rank != (uint32_t)g->rank) {
            PRINT_ERROR("ppcaReplaySetDeserialize: class %zu dim/rank mismatch "
                        "(file %u/%u, skeleton %zu/%zu)",
                        c, dim, rank, g->dim, g->rank);
            exit(1);
        }
        peekValidateThenDeserializeTensor(g->mean, f, "mean");
        peekValidateThenDeserializeTensor(g->basis, f, "basis");
        peekValidateThenDeserializeTensor(g->eigvals, f, "eigvals");
        /* Scalars land only after every guard for this class has passed. */
        g->count = count;
        g->sigma2 = sigma2;
        g->totalVar = totalVar;
    }
}
