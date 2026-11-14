#define SOURCE_FILE "PPCA_REPLAY_SERIALIZE"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Deserialize.h"
#include "PpcaReplaySerialize.h"
#include "SerialWire.h"
#include "Serialize.h"
#include "Tensor.h"
#include "TensorRecord.h"

#define PPCA_SERIALIZE_MAGIC "ODTR"
#define PPCA_SERIALIZE_FORMAT_VERSION 3u

void ppcaReplaySetSerialize(const ppcaReplaySet_t *set, FILE *f) {
    serialWriteBytes(PPCA_SERIALIZE_MAGIC, 4, f);
    serialWriteU32LE(PPCA_SERIALIZE_FORMAT_VERSION, f);
    serialWriteU32LE(ODTS_TENSOR_RECORD_VERSION, f);
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

/* Equal packed byte counts can hide width mismatches. Group counts may
 * differ because loading reallocates group arrays. */
static void validateRecordAgainstSkeleton(const tensorRecordHeader_t *header,
                                          const tensor_t *skeleton, const char *what) {
    size_t nd = header->numberOfDimensions;
    if (nd != skeleton->shape->numberOfDimensions) {
        PRINT_ERROR("ppcaReplaySetDeserialize: %s rank mismatch (file %zu, skeleton %zu)", what, nd,
                    skeleton->shape->numberOfDimensions);
        exit(1);
    }
    for (size_t i = 0; i < nd; i++) {
        if (header->dimensions[i] != skeleton->shape->dimensions[i]) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s dim[%zu] mismatch (file %zu, skeleton %zu)",
                        what, i, header->dimensions[i], skeleton->shape->dimensions[i]);
            exit(1);
        }
    }

    qtype_t fileType = header->quantization.type;
    if (fileType != skeleton->quantization->type) {
        PRINT_ERROR("ppcaReplaySetDeserialize: %s dtype mismatch (file %u, skeleton %u) — "
                    "rebuild the skeleton with the checkpoint's storage config",
                    what, (unsigned)fileType, (unsigned)skeleton->quantization->type);
        exit(1);
    }
    switch (fileType) {
    case SYM: {
        const symQConfig_t *fileQc = &header->qConfigStorage.sym;
        const symQConfig_t *skelQc = skeleton->quantization->qConfig;
        if (fileQc->qBits != skelQc->qBits) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s SYM qBits mismatch (file %u, skeleton "
                        "%u)",
                        what, (unsigned)fileQc->qBits, (unsigned)skelQc->qBits);
            exit(1);
        }
        break;
    }
    case ASYM: {
        const asymQConfig_t *fileQc = &header->qConfigStorage.asym;
        const asymQConfig_t *skelQc = skeleton->quantization->qConfig;
        if (fileQc->qBits != skelQc->qBits) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s ASYM qBits mismatch (file %u, skeleton %u)",
                        what, (unsigned)fileQc->qBits, (unsigned)skelQc->qBits);
            exit(1);
        }
        break;
    }
    case BFP: {
        const bfpQConfig_t *fileQc = &header->qConfigStorage.bfp;
        const bfpQConfig_t *skelQc = skeleton->quantization->qConfig;
        if (fileQc->mantissaBits != skelQc->mantissaBits ||
            fileQc->exponentBits != skelQc->exponentBits) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s BFP width mismatch (file m=%u/e=%u, "
                        "skeleton m=%u/e=%u)",
                        what, (unsigned)fileQc->mantissaBits, (unsigned)fileQc->exponentBits,
                        (unsigned)skelQc->mantissaBits, (unsigned)skelQc->exponentBits);
            exit(1);
        }
        break;
    }
    default:
        break;
    }
}

/* Validate the record header against the skeleton before loading that tensor. */
static void peekValidateThenDeserializeTensor(tensor_t *skeleton, FILE *f, const char *what) {
    long recordStart = ftell(f);
    if (recordStart < 0) {
        PRINT_ERROR("ppcaReplaySetDeserialize: stream not seekable (%s)", what);
        exit(1);
    }

    tensorRecordHeader_t header;
    readTensorRecordHeader(&header, f);
    validateRecordAgainstSkeleton(&header, skeleton, what);
    freeTensorRecordHeader(&header);

    if (fseek(f, recordStart, SEEK_SET) != 0) {
        PRINT_ERROR("ppcaReplaySetDeserialize: rewind failed (%s)", what);
        exit(1);
    }
    deserializeTensor(skeleton, f);
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
    uint32_t recordVersion = serialReadU32LE(f);
    if (recordVersion != ODTS_TENSOR_RECORD_VERSION) {
        PRINT_ERROR("ppcaReplaySetDeserialize: tensor-record version %u, this build reads %u",
                    (unsigned)recordVersion, (unsigned)ODTS_TENSOR_RECORD_VERSION);
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
