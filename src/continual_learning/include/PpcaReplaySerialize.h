#ifndef ODT_PPCA_REPLAY_SERIALIZE_H
#define ODT_PPCA_REPLAY_SERIALIZE_H

#include <stdio.h>

#include "PpcaReplay.h"

/* Standalone "ODTR" v1 checkpoint for a ppcaReplaySet_t — the repo's first
 * non-layer serialized state. Wire format:
 *   "ODTR" | u32 version=1 | u32 numClasses
 *   per class: u32 dim | u32 rank | u32 count | f32 sigma2 | f32 totalVar
 *              | serializeTensor(mean) | serializeTensor(basis)
 *              | serializeTensor(eigvals)
 * Host-native widths/endianness (inherited from the model format).
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
