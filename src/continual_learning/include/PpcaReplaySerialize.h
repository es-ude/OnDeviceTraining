#ifndef ODT_PPCA_REPLAY_SERIALIZE_H
#define ODT_PPCA_REPLAY_SERIALIZE_H

#include <stdio.h>

#include "PpcaReplay.h"

/* "ODTR" v3 checkpoint for a ppcaReplaySet_t. Wire format (every scalar
 * little-endian):
 *   "ODTR" | u32 version=3 | u32 tensorRecordVersion | u32 numClasses
 *   per class: u32 dim | u32 rank | u32 count | f32 sigma2 | f32 totalVar
 *              | serializeTensor(mean) | serializeTensor(basis)
 *              | serializeTensor(eigvals)
 * tensorRecordVersion is ODTS_TENSOR_RECORD_VERSION (TensorRecord.h), so a
 * file written under another tensor-record layout fails at load by name.
 *
 * Deserialize into a matching prebuilt skeleton using a seekable stream. Each
 * tensor header (rank, dims, dtype, SYM/ASYM qBits, BFP widths) is checked
 * before that tensor is overwritten; loading is not transactional across the
 * set, so on a mismatch earlier tensors and classes may already be loaded.
 * Any mismatch exits. */
void ppcaReplaySetSerialize(const ppcaReplaySet_t *set, FILE *f);
void ppcaReplaySetDeserialize(ppcaReplaySet_t *skeleton, FILE *f);

#endif // ODT_PPCA_REPLAY_SERIALIZE_H
