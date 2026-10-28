#ifndef ODT_PPCA_REPLAY_SERIALIZE_H
#define ODT_PPCA_REPLAY_SERIALIZE_H

#include <stdio.h>

#include "PpcaReplay.h"

/* Standalone "ODTR" v3 checkpoint for a ppcaReplaySet_t — the repo's first
 * non-layer serialized state. Wire format (every scalar little-endian):
 *   "ODTR" | u32 version=3 | u32 tensorRecordVersion | u32 numClasses
 *   per class: u32 dim | u32 rank | u32 count | f32 sigma2 | f32 totalVar
 *              | serializeTensor(mean) | serializeTensor(basis)
 *              | serializeTensor(eigvals)
 * tensorRecordVersion is ODTS_TENSOR_RECORD_VERSION (TensorRecord.h): a file
 * written under another tensor-record layout fails at load by name. A change
 * to the bytes serializeTensor writes is caught by the tensor golden-bytes
 * tests going RED; the fix is to bump ODTS_TENSOR_RECORD_VERSION together
 * with SERIALIZE_FORMAT_VERSION and regenerate the goldens.
 *
 * Deserialize fills a pre-built matching skeleton IN PLACE and guards
 * BEFORE any overwrite via peek-validate-rewind (the #316 guard pattern:
 * the public deserializeTensor trusts file bytes, so every record header
 * is first parsed by the shared ODTS header parse (readTensorRecordHeader,
 * TensorRecord.h, which applies deserializeTensor's own qConfig checks),
 * compared with the skeleton — rank, dims, dtype, SYM/ASYM qBits, BFP
 * widths — then the stream is rewound and deserializeTensor consumes the
 * validated record).
 * Requires a SEEKABLE stream. Fail-fast on any mismatch. */
void ppcaReplaySetSerialize(const ppcaReplaySet_t *set, FILE *f);
void ppcaReplaySetDeserialize(ppcaReplaySet_t *skeleton, FILE *f);

#endif // ODT_PPCA_REPLAY_SERIALIZE_H
