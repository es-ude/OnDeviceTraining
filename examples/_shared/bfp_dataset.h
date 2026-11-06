#ifndef EXAMPLES_SHARED_BFP_DATASET_H
#define EXAMPLES_SHARED_BFP_DATASET_H

#include <stdbool.h>
#include <stddef.h>

#include "Dataset.h"
#include "Quantization.h"
#include "Tensor.h"

/* Pre-converted BFP datasets for the full-integer training path: the samples
 * are converted from FLOAT32 to the model's input-wire BFP format ONCE,
 * before training, and the one-hot labels to BFP, so a training step touches
 * no float data. The conversion runs the library's own FLOAT32 -> BFP cell,
 * the one the input Quantization layer runs online, so the stored codes are
 * byte for byte what the online layer would write (that layer then passes a
 * BFP input of its own format straight through).
 *
 * File format (ODTA, "ODT tensor array"): bytes "ODTA", u32 LE tensor-record
 * version (ODTS_TENSOR_RECORD_VERSION), u32 LE tensor count, then that many
 * ODTS tensor records exactly as serializeTensor writes them (shape, dtype,
 * qConfig incl. BFP group exponents, packed payload), nothing after. */

/* Every FLOAT32 tensor of src converted into a fresh tensor of templateQ's
 * storage format (same shape). templateQ must be BFP with HALF_AWAY rounding:
 * the conversion uses no RNG stream, and an SR template would need the
 * per-call stream only the online layer has. Fails fast (exit 1) otherwise,
 * or on a non-FLOAT32 source tensor. */
tensorArray_t *bfpDatasetConvert(const tensorArray_t *src, const quantization_t *templateQ);

/* One-hot labels of numClasses elements (shape [numClasses]) from INT32
 * class indices (one element each), in templateQ's BFP format (converted
 * like bfpDatasetConvert from a FLOAT32 one-hot; 0 and 1.0 are exact).
 * Fails fast on a class out of range or a label that does not come out as
 * an exact one-hot (bfpOneHotClass). */
tensorArray_t *bfpDatasetOneHotLabels(const tensorArray_t *classes, size_t numClasses,
                                      const quantization_t *templateQ);

/* true iff label is BFP and exactly one element is exactly 1.0 (code 2^-E on
 * its group's 2^E, E <= 0) and every other is code 0; *cls = that element.
 * Integer only. */
bool bfpOneHotClass(const tensor_t *label, size_t *cls);

/* Writes / reads an ODTA file (format above). The reader builds a fresh
 * tensor per record (identity dimension order; FLOAT32, INT32 and BFP) and
 * fails fast on a missing file, a foreign magic, another record version,
 * a truncated record or trailing bytes. The writer fails fast on a tensor
 * whose dimension order is not the identity. Free with freeTensorArray. */
void tensorArrayWriteFile(const char *path, const tensorArray_t *arr);
tensorArray_t *tensorArrayReadFile(const char *path);

#endif /* EXAMPLES_SHARED_BFP_DATASET_H */
