#define SOURCE_FILE "DESERIALIZE"

#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Deserialize.h"
#include "DeserializeInternal.h"
#include "Dropout.h"
#include "GroupNorm.h"
#include "Kernel.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Rounding.h"
#include "SerialWire.h"
#include "Softmax.h"
#include "StorageApi.h"
#include "Tensor.h"

/* Mirrors Serialize.c's locked format v2 constants (#370): fixed-width
 * little-endian scalars via the checked SerialWire primitives; no v1
 * back-compat shim — v1 files were host-local artifacts.
 * v3: parameter records carry a grad-presence byte (#380). The reader is
 * TOLERANT of a presence/skeleton mismatch (#380 PR3, superseding PR1's
 * fail-fast): see deserializeParameter / skipSerializedTensor below.
 * v4 (group-quant PR1/PR2, spec
 * docs/superpowers/specs/2026-07-28-group-quantization-design.md §6): the SYM
 * qConfig record grows `u32 numGroups`, `u32 groupSize` ahead of the scales
 * array. A file whose numGroups does not match the skeleton's own (PR1:
 * always 1, from initSymQConfig) REALLOCATES the skeleton's scales[] to the
 * file's numGroups (PR2, Task 5) rather than failing fast; the #316
 * no-silent-misparse discipline now lives in validateSymQConfigShape's
 * post-reallocation divisibility check (Quantization.h) instead. The
 * sentinel invariant (numGroups==1 <=> groupSize==0, Quantization.h) is
 * checked on the FILE's values and is untouched by the relax; a violation is
 * a corrupt or from-the-future record. v4's ASYM record was an INTRA-BRANCH
 * BRIDGE (old per-tensor shape, i32 zeroPoint slot repurposed to carry a
 * code-domain uint16 value) -- superseded by v5 below.
 * v5 (group-quant PR4, Task 4): the ASYM qConfig record gets the SAME
 * numGroups/groupSize-prefixed, reallocate-on-mismatch treatment the v4 SYM
 * record (and PR2/Task 5's relax) already gave SYM, PLUS a second per-group
 * array -- `u16 zeroPoints[numGroups]` (LE) after `f32 scales[numGroups]` --
 * both reallocated together whenever the file's numGroups differs from the
 * skeleton's own. No migration path from v4: that record's ASYM value
 * decoded WRONG under the code-domain grid by design of the interim bridge,
 * so a v4 file (including ones this branch's own Tasks 1-3 produced) now
 * fails cleanly at the version check below -- consistent with the v1->v4
 * no-back-compat-shim policy. See Serialize.c's v5 comment for the
 * BFP-coordination note. */
#define SERIALIZE_MAGIC "ODTS"
#define SERIALIZE_FORMAT_VERSION 5u

void deserializeTensor(tensor_t *tensor, FILE *f) {
    /* #316: capture the skeleton's expected payload size BEFORE the shape /
     * quantization overwrites. tensor->data was sized by initTensor from this
     * build-time shape + quantization; a file record with a larger element count
     * or packed width (which the dtype check alone cannot see) would otherwise
     * fread past that allocation. */
    size_t expectedBytes =
        calcNumberOfBytesForData(tensor->quantization, calcNumberOfElementsByShape(tensor->shape));

    deserializeShape(tensor->shape, f);
    /* Group-quant PR2 (Task 5): computed here (not after deserializeQuantization,
     * as before) because a SYM record's divisibility validate needs it — shape
     * already holds the FILE's dimensions at this point (deserializeShape wrote
     * them), so this is the tensor's true element count, not the skeleton's
     * pre-overwrite one. */
    size_t numberOfValues = calcNumberOfElementsByShape(tensor->shape);
    deserializeQuantization(tensor->quantization, f, numberOfValues);

    /* Mirrors Serialize.c: payload length is the packed size. */
    size_t dataBytes = calcNumberOfBytesForData(tensor->quantization, numberOfValues);

    if (dataBytes != expectedBytes) {
        PRINT_ERROR("deserializeTensor: file payload %zu bytes does not match the skeleton's "
                    "allocated %zu bytes (shape/qBits mismatch)",
                    dataBytes, expectedBytes);
        exit(1);
    }

    serialReadBytes(tensor->data, dataBytes, f);
    deserializeSparsity();
}

void deserializeParameter(parameter_t *parameter, FILE *f) {
    uint8_t hasGrad = serialReadU8(f);
    deserializeTensor(parameter->param, f);
    if (hasGrad && parameter->grad != NULL) {
        deserializeTensor(parameter->grad, f);
    } else if (hasGrad) {
        /* Frozen skeleton, full checkpoint (#380 PR3): grads are transient
         * (zeroed before every batch) and a frozen layer allocates none -
         * parse past the record so the stream stays synced. */
        skipSerializedTensor(f);
    }
    /* hasGrad == 0 with a trainable skeleton: grads stay as built (zeroed);
     * optimizerZeroGrad re-zeros before every batch anyway. */
}

void deserializeModel(layer_t **model, size_t sizeModel, FILE *f) {
    char magic[4];
    serialReadBytes(magic, 4, f);
    if (memcmp(magic, SERIALIZE_MAGIC, 4) != 0) {
        PRINT_ERROR("deserializeModel: bad magic bytes (expected \"ODTS\")");
        exit(1);
    }

    uint32_t version = serialReadU32LE(f);
    if (version != SERIALIZE_FORMAT_VERSION) {
        PRINT_ERROR("deserializeModel: unsupported format version %u (expected %u)",
                    (unsigned)version, (unsigned)SERIALIZE_FORMAT_VERSION);
        exit(1);
    }

    uint32_t layerCount = serialReadU32LE(f);
    if (layerCount != (uint32_t)sizeModel) {
        PRINT_ERROR("deserializeModel: layerCount mismatch (file has %u, caller expects %zu)",
                    (unsigned)layerCount, sizeModel);
        exit(1);
    }

    for (size_t i = 0; i < sizeModel; i++) {
        /* Tag byte = layerType_t enum position (append-only wire contract; see
         * Layer.h and the pins in test/unit/serial/UnitTestSerialize.c). */
        uint8_t tag = serialReadU8(f);
        if (tag != (uint8_t)model[i]->type) {
            PRINT_ERROR("deserializeModel: record tag %u does not match expected layer type "
                        "%u at index %zu",
                        (unsigned)tag, (unsigned)model[i]->type, i);
            exit(1);
        }
        deserializeLayer(model[i], f);
    }
}

// Helper Functions

static void deserializeShape(shape_t *shape, FILE *f) {
    uint32_t fileRank = serialReadU32LE(f);
    /* The skeleton's dimensions/orderOfDimensions arrays were sized by the
     * build-time rank; a mismatched record would otherwise write file dims
     * past them (and an equal-element-count rank change would slip past the
     * #316 payload-size check below). */
    if ((size_t)fileRank != shape->numberOfDimensions) {
        PRINT_ERROR("deserializeShape: file rank %u does not match the skeleton rank %zu",
                    (unsigned)fileRank, shape->numberOfDimensions);
        exit(1);
    }
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        shape->dimensions[d] = (size_t)serialReadU32LE(f);
    }
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        shape->orderOfDimensions[d] = (size_t)serialReadU32LE(f);
    }
}

static void deserializeQuantization(quantization_t *q, FILE *f, size_t numberOfElements) {
    uint8_t type = serialReadU8(f);
    /* #316: the skeleton was built with a fixed dtype whose qConfig struct (or
     * NULL, for FLOAT32/INT32/BOOL) is fixed. A file record claiming a different
     * dtype would make deserializeQConfig write scale/qBits/... through a
     * mismatched or NULL qConfig — segfault on host, silent low-address writes on
     * an MMU-less MCU. Reject the mismatch before the overwrite; running here (not
     * in deserializeQConfig) forecloses the NULL-qConfig deref. */
    if ((qtype_t)type != q->type) {
        PRINT_ERROR("deserializeQuantization: file dtype %u does not match the skeleton dtype %u",
                    (unsigned)type, (unsigned)q->type);
        exit(1);
    }
    q->type = (qtype_t)type;
    deserializeQConfig(q, f, numberOfElements);
}

static void deserializeArithmetic(arithmetic_t *arithmetic, FILE *f) {
    arithmetic->type = (arithmeticType_t)serialReadU8(f);
    arithmetic->roundingMode = (roundingMode_t)serialReadU8(f);
}

static void deserializeKernel(kernel_t *kernel, FILE *f) {
    kernel->size = (size_t)serialReadU32LE(f);
    kernel->paddingType = (paddingType_t)serialReadU8(f);
    kernel->stride = (size_t)serialReadU32LE(f);
    kernel->dilation = (size_t)serialReadU32LE(f);
    kernel->padding = (size_t)serialReadU32LE(f);
}

/* Task-5 review fix (Critical): sanity cap on the untrusted wire
 * fileNumGroups (SYM qConfig record) read below, mirroring
 * SKIP_TENSOR_MAX_DIMS's role for the untrusted wire rank further down --
 * generous against any real model (a group count is bounded by the element
 * count of the largest parameter tensor it groups, never in the tens of
 * thousands), so this is headroom, not a real limit. It exists purely to
 * foreclose the allocation-size trust boundary a raw u32 * sizeof(float)
 * would otherwise cross: on a 32-bit size_t (MCU target), a value like
 * 0x40000001 makes fileNumGroups * sizeof(float) wrap around to 4 before it
 * ever reaches reserveMemory, handing back a 1-float buffer while the
 * scales-read loop below still iterates the file's full (unwrapped) count --
 * a heap overflow; on a 64-bit host the multiply does not wrap, but a
 * multi-gigabyte request makes reserveMemory's calloc return NULL, and nothing
 * downstream expects that. The cap bounds the allocation at 256 KiB and rules
 * out both.
 * group-quant PR4 (Task 4): renamed from SERIAL_MAX_SYM_GROUPS -- the v5
 * ASYM arm below reuses this exact cap for its own untrusted fileNumGroups
 * (same rationale, same value; ASYM's second array (zeroPoints, u16) is even
 * smaller per-group than SYM's, so the 256 KiB sizing headroom above is
 * conservative for it too). Keep PpcaReplaySerialize.c's PPCA_MAX_QCONFIG_GROUPS
 * literal equal to this by hand (see that file's lockstep comment). */
#define SERIAL_MAX_QCONFIG_GROUPS 65536u

static void deserializeQConfig(quantization_t *q, FILE *f, size_t numberOfElements) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32: {
        symInt32QConfig_t *symIntQC = q->qConfig;
        symIntQC->scale = serialReadF32LE(f);
        symIntQC->roundingMode = (roundingMode_t)serialReadU8(f);
        symIntQC->qMaxBits = serialReadU8(f);
        break;
    }
    case SYM: {
        symQConfig_t *symQC = q->qConfig;
        /* v4 (group-quant PR1/PR2): symQC->scales must already point at a
         * valid (>=1-element) array -- callers build the skeleton via
         * initSymQConfig/initSymQConfigGrouped or the stack-fixture idiom
         * before deserializing into it. A file numGroups that differs from
         * the skeleton's own is no longer a fail-fast (PR1's #316
         * no-silent-misparse discipline): REALLOCATE the scales array to the
         * file's shape instead -- the resulting shape is validated below
         * (validateSymQConfigShape), which is the discipline's PR2 form. */
        size_t fileNumGroups = (size_t)serialReadU32LE(f);
        /* Task-5 review fix (Critical): fileNumGroups is untrusted wire input
         * about to size an allocation -- bound it BEFORE touching
         * symQC->scales at all. Zero is never valid (numGroups==1 is the
         * per-tensor floor; the sentinel check below would catch it too, but
         * only after a pointless free+realloc(0) round trip), and
         * SERIAL_MAX_QCONFIG_GROUPS forecloses the size_t-wrap-on-32-bit /
         * NULL-calloc-on-64-bit pair a multi-GB value would otherwise invite
         * (see the macro's own comment above). */
        if (fileNumGroups == 0 || fileNumGroups > SERIAL_MAX_QCONFIG_GROUPS) {
            PRINT_ERROR("deserializeQConfig: SYM file numGroups %zu is zero or exceeds the "
                        "%u-group sanity cap",
                        fileNumGroups, (unsigned)SERIAL_MAX_QCONFIG_GROUPS);
            exit(1);
        }
        /* Whenever a live tensor backs this config (numberOfElements != 0 --
         * the skip path now passes its own real count too, see
         * skipSerializedTensor below), a config cannot have more groups than
         * elements; reject that here, before the realloc, so the later
         * validateSymQConfigShape call downstream keeps its own job
         * (divisibility) uncomplicated by an out-of-range numGroups it would
         * otherwise have to multiply against. numberOfElements == 0 (the
         * layer outputQ/propLossQ wire-config call sites) has no tensor to
         * bound against -- SERIAL_MAX_QCONFIG_GROUPS above is the only guard
         * there, and is sized generously enough to cover it alone. */
        if (numberOfElements != 0 && fileNumGroups > numberOfElements) {
            PRINT_ERROR("deserializeQConfig: SYM file numGroups %zu exceeds the %zu-element "
                        "tensor it attaches to",
                        fileNumGroups, numberOfElements);
            exit(1);
        }
        if (fileNumGroups != symQC->numGroups) {
            freeReservedMemory(symQC->scales);
            symQC->scales = reserveMemory(fileNumGroups * sizeof(float));
            symQC->numGroups = fileNumGroups;
        }
        size_t fileGroupSize = (size_t)serialReadU32LE(f);
        /* Sentinel invariant (Quantization.h): numGroups == 1 <=> groupSize
         * == 0. Checked on the FILE values -- a violation means a corrupt
         * record or one written by a future (group-aware) format this build
         * cannot interpret. Untouched by the reallocation relax above. */
        if ((fileNumGroups == 1) != (fileGroupSize == 0)) {
            PRINT_ERROR("deserializeQConfig: SYM file violates the numGroups==1<=>groupSize==0 "
                        "sentinel invariant (numGroups=%zu, groupSize=%zu)",
                        fileNumGroups, fileGroupSize);
            exit(1);
        }
        symQC->groupSize = fileGroupSize;
        for (size_t g = 0; g < fileNumGroups; g++) {
            symQC->scales[g] = serialReadF32LE(f);
        }
        symQC->qBits = serialReadU8(f);
        symQC->roundingMode = (roundingMode_t)serialReadU8(f);
        /* numberOfElements == 0 marks ONLY the layer outputQ/propLossQ
         * wire-config call sites in deserializeLayer -- no live tensor backs
         * q there (group-quant PR2's carrier gate keeps those per-tensor
         * anyway, so skipping this validate there costs nothing). Every
         * other caller reaches this validate, INCLUDING skipSerializedTensor's
         * grad-skip path (Task-5 review fix: it now threads the real element
         * count it just parsed off the wire, not 0) -- a grouped grad record
         * whose numGroups*groupSize does not divide its own element count is
         * corrupt whether or not the resulting scratch qConfig ever attaches
         * to a live tensor, so it fails fast here too. This is the choke
         * point that turns "the reallocation succeeded" into "the resulting
         * shape actually describes this record's own element count". */
        if (numberOfElements != 0) {
            validateSymQConfigShape(symQC, numberOfElements);
        }
        break;
    }
    case ASYM: {
        asymQConfig_t *asymQC = q->qConfig;
        /* v5 (group-quant PR4, Task 4): the ASYM twin of the SYM arm above --
         * same caps-before-allocation discipline, same reallocate-on-mismatch
         * relax, PLUS a second array (zeroPoints) reallocated in lockstep
         * with scales. Replaces Task 1's v4 bridge (which rejected any
         * grouped skeleton outright) entirely. */
        size_t fileNumGroups = (size_t)serialReadU32LE(f);
        /* Untrusted wire input about to size TWO allocations -- bound it
         * BEFORE touching asymQC's arrays at all (mirrors the SYM arm's
         * Task-5 review fix). */
        if (fileNumGroups == 0 || fileNumGroups > SERIAL_MAX_QCONFIG_GROUPS) {
            PRINT_ERROR("deserializeQConfig: ASYM file numGroups %zu is zero or exceeds the "
                        "%u-group sanity cap",
                        fileNumGroups, (unsigned)SERIAL_MAX_QCONFIG_GROUPS);
            exit(1);
        }
        /* Whenever a live tensor backs this config (numberOfElements != 0),
         * a config cannot have more groups than elements; reject before the
         * realloc, mirroring the SYM arm's identical guard. */
        if (numberOfElements != 0 && fileNumGroups > numberOfElements) {
            PRINT_ERROR("deserializeQConfig: ASYM file numGroups %zu exceeds the %zu-element "
                        "tensor it attaches to",
                        fileNumGroups, numberOfElements);
            exit(1);
        }
        if (fileNumGroups != asymQC->numGroups) {
            /* Both arrays are reallocated together -- a mismatch always
             * resizes the whole config, never just one array (the mutation
             * this guards against: sizing only scales[] leaves zeroPoints[]
             * stale, an ASan-visible heap-buffer-overflow the moment the
             * zeroPoints loop below writes past it). */
            freeReservedMemory(asymQC->scales);
            freeReservedMemory(asymQC->zeroPoints);
            asymQC->scales = reserveMemory(fileNumGroups * sizeof(float));
            asymQC->zeroPoints = reserveMemory(fileNumGroups * sizeof(uint16_t));
            asymQC->numGroups = fileNumGroups;
        }
        size_t fileGroupSize = (size_t)serialReadU32LE(f);
        /* Sentinel invariant (Quantization.h): numGroups == 1 <=> groupSize
         * == 0. Checked on the FILE values, mirroring the SYM arm. */
        if ((fileNumGroups == 1) != (fileGroupSize == 0)) {
            PRINT_ERROR("deserializeQConfig: ASYM file violates the numGroups==1<=>groupSize==0 "
                        "sentinel invariant (numGroups=%zu, groupSize=%zu)",
                        fileNumGroups, fileGroupSize);
            exit(1);
        }
        asymQC->groupSize = fileGroupSize;
        /* scales THEN zeroPoints, matching Serialize.c's write order exactly
         * (the mutation this order guards against: swapping the two loops
         * decodes every scale as a zeroPoint's bit pattern and vice versa --
         * an immediate golden-bytes and round-trip failure, never a silent
         * pass). */
        for (size_t g = 0; g < fileNumGroups; g++) {
            asymQC->scales[g] = serialReadF32LE(f);
        }
        for (size_t g = 0; g < fileNumGroups; g++) {
            asymQC->zeroPoints[g] = serialReadU16LE(f);
        }
        uint8_t fileQBits = serialReadU8(f);
        if (fileQBits == 0 || fileQBits > 16) {
            /* D6: uint16 code-domain zp requires qBits <= 16; a wider record
             * is corrupt or written by an incompatible/future build. Checked
             * unconditionally (independent of numberOfElements) since it
             * does not depend on a live tensor's element count -- same
             * immediacy as the pre-v5 bridge's inline check. */
            PRINT_ERROR("deserializeQConfig: ASYM file qBits %u outside [1, 16] (D6)",
                        (unsigned)fileQBits);
            exit(1);
        }
        asymQC->qBits = fileQBits;
        asymQC->roundingMode = (roundingMode_t)serialReadU8(f);
        /* numberOfElements == 0 marks ONLY the layer outputQ/propLossQ
         * wire-config call sites (mirrors the SYM arm's identical note). */
        if (numberOfElements != 0) {
            validateAsymQConfigShape(asymQC, numberOfElements);
        }
        break;
    }
    default:
        PRINT_ERROR("Unknown qType!");
        exit(1);
    }
}

/* Sanity cap on skipSerializedTensor's untrusted file rank: no shipped tensor
 * exceeds rank 8, so a larger value is a corrupt/malicious record. Bounds the
 * dims-read loop below -- no dims[] array is sized off this value (#380 PR3
 * review: the array was write-only, dropped in favor of an inline product). */
#define SKIP_TENSOR_MAX_DIMS 8

static void skipSerializedTensor(FILE *f) {
    long recordStart = ftell(f);
    if (recordStart < 0) {
        PRINT_ERROR("skipSerializedTensor: stream not seekable (grad-record skip requires "
                    "fseek/ftell support, #380 PR3)");
        exit(1);
    }

    uint32_t numDims = serialReadU32LE(f);
    if (numDims > SKIP_TENSOR_MAX_DIMS) {
        PRINT_ERROR("skipSerializedTensor: rank %u exceeds the %d-dim skip-helper cap",
                    (unsigned)numDims, SKIP_TENSOR_MAX_DIMS);
        exit(1);
    }
    size_t numberOfElements = 1;
    for (uint32_t d = 0; d < numDims; d++) {
        numberOfElements *= (size_t)serialReadU32LE(f);
    }
    /* orderOfDimensions: positional only, irrelevant to a discarded record. */
    if (fseek(f, (long)numDims * (long)sizeof(uint32_t), SEEK_CUR) != 0) {
        PRINT_ERROR("skipSerializedTensor: seek past orderOfDimensions failed");
        exit(1);
    }

    uint8_t type = serialReadU8(f);
    quantization_t scratchQ = {.type = (qtype_t)type, .qConfig = NULL};
    symInt32QConfig_t symIntScratch;
    /* v4/group-quant PR2 (Task 5): unlike the OTHER scratch qConfigs here
     * (symIntScratch/asymScratch, plain stack locals never freed), symScratch
     * needs a REAL heap-allocated scales[1] from the start, not a stack
     * array -- a GROUPED skipped record (file numGroups > 1) is no longer
     * rejected, and deserializeQConfig's SYM arm unconditionally
     * freeReservedMemory()s the OLD scales pointer before reserveMemory()ing
     * a differently-sized one whenever the file's numGroups differs from
     * the qConfig's own (the same contract every live tensor's heap-owned
     * symQConfig relies on, Quantization.h's ownership note). A stack-backed
     * initial array would make that free() undefined behavior -- verified:
     * SIGABRT with no PRINT_ERROR on this host's allocator before this fix,
     * exactly the "drop the free = leak" mutation's mirror image. Freed
     * unconditionally below regardless of whether a reallocation actually
     * happened: this whole qConfig is discarded at function exit, never
     * attached to a real tensor, so nothing else owns whatever it ends up
     * pointing at. Task-5 review fix: the record's own numberOfElements
     * (computed above from the dims it just read, not a hardcoded 0) is
     * threaded into deserializeQConfig below, so a grouped record whose
     * numGroups*groupSize does not divide its own element count fails fast
     * here too -- the discarded scratch qConfig no longer exempts a
     * corrupt/malformed grouped grad record from the divisibility check a
     * live tensor's qConfig would get. */
    symQConfig_t symScratch = {0};
    /* asymScratch (group-quant PR4, Task 4): now the ASYM twin of symScratch
     * above -- the v5 ASYM arm of deserializeQConfig unconditionally
     * freeReservedMemory()s BOTH of asymScratch's current arrays before
     * reserveMemory()ing differently-sized ones whenever the file's
     * numGroups differs from the qConfig's own, exactly the SYM contract. A
     * stack-backed initial array would make that free() undefined behavior
     * (the same SIGABRT hazard symScratch's comment documents). Freed
     * unconditionally below regardless of whether a reallocation actually
     * happened, mirroring symScratch's disposal. */
    asymQConfig_t asymScratch = {0};
    switch (scratchQ.type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32:
        scratchQ.qConfig = &symIntScratch;
        break;
    case SYM:
        symScratch.scales = reserveMemory(sizeof(float));
        symScratch.numGroups = 1;
        symScratch.groupSize = 0;
        scratchQ.qConfig = &symScratch;
        break;
    case ASYM:
        asymScratch.scales = reserveMemory(sizeof(float));
        asymScratch.zeroPoints = reserveMemory(sizeof(uint16_t));
        asymScratch.numGroups = 1;
        asymScratch.groupSize = 0;
        asymScratch.qBits = 8;
        asymScratch.roundingMode = HALF_AWAY;
        scratchQ.qConfig = &asymScratch;
        break;
    default:
        PRINT_ERROR("skipSerializedTensor: unknown qtype %u", (unsigned)type);
        exit(1);
    }
    deserializeQConfig(&scratchQ, f, numberOfElements);
    if (scratchQ.type == SYM) {
        freeReservedMemory(symScratch.scales);
    }
    if (scratchQ.type == ASYM) {
        /* Task 4 mutation guard: dropping either free here leaks the
         * reallocated array whenever a grouped grad's numGroups differs from
         * this scratch's initial 1 -- LSan/ASan-visible (see the report's
         * mutation transcript), never a functional test failure on its own
         * since the scratch is discarded either way. */
        freeReservedMemory(asymScratch.scales);
        freeReservedMemory(asymScratch.zeroPoints);
    }

    size_t payloadBytes = calcNumberOfBytesForData(&scratchQ, numberOfElements);
    if (fseek(f, (long)payloadBytes, SEEK_CUR) != 0) {
        PRINT_ERROR("skipSerializedTensor: seek past payload failed");
        exit(1);
    }
    /* deserializeSparsity() is a zero-byte TODO stub -- nothing to skip. */

    /* Post-skip truncation guard (mirrors the ODTR/PPCA ftell precedent,
     * #316 wave): fseek past a genuinely truncated file's real end succeeds
     * silently (POSIX allows seeking beyond EOF), so only comparing against
     * the actual on-disk size catches it. */
    long recordEnd = ftell(f);
    if (recordEnd < 0 || fseek(f, 0, SEEK_END) != 0) {
        PRINT_ERROR("skipSerializedTensor: position query failed after skip");
        exit(1);
    }
    long fileSize = ftell(f);
    if (fileSize < 0 || recordEnd > fileSize) {
        PRINT_ERROR("skipSerializedTensor: skipped record extends past end of file (truncated?)");
        exit(1);
    }
    if (fseek(f, recordEnd, SEEK_SET) != 0) {
        PRINT_ERROR("skipSerializedTensor: reposition after truncation check failed");
        exit(1);
    }
}

// TODO
static void deserializeSparsity() {}

static void deserializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR: {
        linearConfig_t *linearConfig = layer->config->linear;
        deserializeParameter(linearConfig->weights, f);
        deserializeParameter(linearConfig->bias, f);
        deserializeArithmetic(&linearConfig->forwardMath, f);
        deserializeArithmetic(&linearConfig->weightGradMath, f);
        deserializeArithmetic(&linearConfig->biasGradMath, f);
        deserializeArithmetic(&linearConfig->propLossMath, f);
        deserializeQuantization(linearConfig->outputQ, f, 0);
        deserializeQuantization(linearConfig->propLossQ, f, 0);
        break;
    }
    case RELU: {
        reluConfig_t *reluConfig = layer->config->relu;
        deserializeArithmetic(&reluConfig->forwardMath, f);
        deserializeArithmetic(&reluConfig->propLossMath, f);
        deserializeQuantization(reluConfig->outputQ, f, 0);
        deserializeQuantization(reluConfig->propLossQ, f, 0);
        break;
    }
    case CONV1D: {
        conv1dConfig_t *conv1dConfig = layer->config->conv1d;
        deserializeKernel(conv1dConfig->kernel, f);
        conv1dConfig->groups = (size_t)serialReadU32LE(f);
        deserializeParameter(conv1dConfig->weights, f);
        uint8_t conv1dHasBias = serialReadU8(f);
        if (conv1dHasBias) {
            deserializeParameter(conv1dConfig->bias, f);
        }
        deserializeArithmetic(&conv1dConfig->forwardMath, f);
        deserializeArithmetic(&conv1dConfig->weightGradMath, f);
        deserializeArithmetic(&conv1dConfig->biasGradMath, f);
        deserializeArithmetic(&conv1dConfig->propLossMath, f);
        deserializeQuantization(conv1dConfig->outputQ, f, 0);
        deserializeQuantization(conv1dConfig->propLossQ, f, 0);
        break;
    }
    case CONV1D_TRANSPOSED: {
        conv1dTransposedConfig_t *conv1dTransposedConfig = layer->config->conv1dTransposed;
        deserializeKernel(conv1dTransposedConfig->kernel, f);
        conv1dTransposedConfig->groups = (size_t)serialReadU32LE(f);
        conv1dTransposedConfig->outputPadding = (size_t)serialReadU32LE(f);
        deserializeParameter(conv1dTransposedConfig->weights, f);
        uint8_t conv1dTransposedHasBias = serialReadU8(f);
        if (conv1dTransposedHasBias) {
            deserializeParameter(conv1dTransposedConfig->bias, f);
        }
        deserializeArithmetic(&conv1dTransposedConfig->forwardMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->weightGradMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->biasGradMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->propLossMath, f);
        deserializeQuantization(conv1dTransposedConfig->outputQ, f, 0);
        deserializeQuantization(conv1dTransposedConfig->propLossQ, f, 0);
        break;
    }
    case MAXPOOL1D: {
        maxPool1dConfig_t *maxPool1dConfig = layer->config->maxPool1d;
        deserializeKernel(maxPool1dConfig->kernel, f);
        deserializeArithmetic(&maxPool1dConfig->forwardMath, f);
        deserializeArithmetic(&maxPool1dConfig->propLossMath, f);
        deserializeQuantization(maxPool1dConfig->outputQ, f, 0);
        deserializeQuantization(maxPool1dConfig->propLossQ, f, 0);
        break;
    }
    case AVGPOOL1D: {
        avgPool1dConfig_t *avgPool1dConfig = layer->config->avgPool1d;
        deserializeKernel(avgPool1dConfig->kernel, f);
        deserializeArithmetic(&avgPool1dConfig->forwardMath, f);
        deserializeArithmetic(&avgPool1dConfig->propLossMath, f);
        deserializeQuantization(avgPool1dConfig->outputQ, f, 0);
        deserializeQuantization(avgPool1dConfig->propLossQ, f, 0);
        break;
    }
    case SOFTMAX: {
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        deserializeArithmetic(&softmaxConfig->forwardMath, f);
        deserializeArithmetic(&softmaxConfig->propLossMath, f);
        deserializeQuantization(softmaxConfig->outputQ, f, 0);
        deserializeQuantization(softmaxConfig->propLossQ, f, 0);
        break;
    }
    case FLATTEN:
        // Flatten carries no state (no parameters, no quantization).
        break;
    case QUANTIZATION: {
        quantizationConfig_t *quantizationConfig = layer->config->quantization;
        deserializeQuantization(quantizationConfig->outputQ, f, 0);
        deserializeQuantization(quantizationConfig->propLossQ, f, 0);
        break;
    }
    case ADAPTIVE_AVGPOOL1D: {
        adaptiveAvgPool1dConfig_t *adaptiveAvgPool1dConfig = layer->config->adaptiveAvgPool1d;
        adaptiveAvgPool1dConfig->outputSize = (size_t)serialReadU32LE(f);
        deserializeArithmetic(&adaptiveAvgPool1dConfig->forwardMath, f);
        deserializeArithmetic(&adaptiveAvgPool1dConfig->propLossMath, f);
        deserializeQuantization(adaptiveAvgPool1dConfig->outputQ, f, 0);
        deserializeQuantization(adaptiveAvgPool1dConfig->propLossQ, f, 0);
        break;
    }
    case DROPOUT: {
        dropoutConfig_t *dropoutConfig = layer->config->dropout;
        dropoutConfig->p = serialReadF32LE(f);
        deserializeArithmetic(&dropoutConfig->forwardMath, f);
        deserializeArithmetic(&dropoutConfig->propLossMath, f);
        deserializeQuantization(dropoutConfig->outputQ, f, 0);
        deserializeQuantization(dropoutConfig->propLossQ, f, 0);
        break;
    }
    case LAYERNORM: {
        layerNormConfig_t *layerNormConfig = layer->config->layerNorm;
        uint32_t layerNormNumNormDims = serialReadU32LE(f);
        /* numNormDims drives the normalizedShape read count; the skeleton's
         * array was sized by the build-time value — reject a mismatch before
         * file entries could be written past it. */
        if ((size_t)layerNormNumNormDims != layerNormConfig->numNormDims) {
            PRINT_ERROR("deserializeLayer: LayerNorm numNormDims mismatch (file %u, skeleton "
                        "%zu)",
                        (unsigned)layerNormNumNormDims, layerNormConfig->numNormDims);
            exit(1);
        }
        for (size_t d = 0; d < layerNormConfig->numNormDims; d++) {
            layerNormConfig->normalizedShape[d] = (size_t)serialReadU32LE(f);
        }
        layerNormConfig->eps = serialReadF32LE(f);
        deserializeParameter(layerNormConfig->gamma, f);
        deserializeParameter(layerNormConfig->beta, f);
        deserializeArithmetic(&layerNormConfig->forwardMath, f);
        deserializeArithmetic(&layerNormConfig->propLossMath, f);
        deserializeQuantization(layerNormConfig->outputQ, f, 0);
        deserializeQuantization(layerNormConfig->propLossQ, f, 0);
        break;
    }
    case GROUPNORM: {
        groupNormConfig_t *groupNormConfig = layer->config->groupNorm;
        groupNormConfig->numGroups = (size_t)serialReadU32LE(f);
        groupNormConfig->numChannels = (size_t)serialReadU32LE(f);
        groupNormConfig->eps = serialReadF32LE(f);
        deserializeParameter(groupNormConfig->gamma, f);
        deserializeParameter(groupNormConfig->beta, f);
        deserializeArithmetic(&groupNormConfig->forwardMath, f);
        deserializeArithmetic(&groupNormConfig->propLossMath, f);
        deserializeQuantization(groupNormConfig->outputQ, f, 0);
        deserializeQuantization(groupNormConfig->propLossQ, f, 0);
        break;
    }
    default:
        PRINT_ERROR("Unsupported layer type!\n");
        exit(1);
    }
}
