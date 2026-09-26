#ifndef STATE_DICT_API_H
#define STATE_DICT_API_H

#include "Layer.h"
#include <stddef.h>
#include <stdint.h>

typedef struct stateDictEntry {
    const char *name; /* OPTIONAL — used only in error messages */
    float *weightData;
    float *biasData; /* NULL allowed iff the corresponding layer has no bias */
} stateDictEntry_t;

/*! Load weights/biases from entries[] into the parameter layers of model,
 *  in the order they appear. Param-less layers are skipped.
 *
 *  Errors (PRINT_ERROR + exit):
 *   - numEntries != count of param layers in model
 *   - any entry's weightData == NULL
 *   - bias presence in entry does not match bias presence in the corresponding layer
 *
 *  Error messages include entries[i].name if non-NULL, otherwise the
 *  param-layer index (0-based). */
void modelLoadStateDict(layer_t **model, size_t numLayers, stateDictEntry_t *entries,
                        size_t numEntries);

typedef struct stateDictBuffers {
    const char *name;         /* OPTIONAL -- error messages only */
    const float *runningMean; /* [C], required; every value finite */
    const float *runningVar;  /* [C], required; every value finite and >= 0 */
    uint64_t numBatchesTracked;
} stateDictBuffers_t;

/*! Load BatchNorm1d running buffers (PyTorch running_mean / running_var /
 *  num_batches_tracked) into the model's BATCHNORM1D layers that track
 *  running stats, in model order; every other layer (incl. a BN with
 *  noRunningStats) is skipped. Copies the values (caller keeps ownership).
 *  Parameters (gamma/beta) load through modelLoadStateDict.
 *
 *  Errors (PRINT_ERROR + exit): numEntries != count of buffer-bearing
 *  layers; a NULL runningMean/runningVar; a non-finite running mean; a
 *  negative or non-finite running variance. Messages name entries[i].name
 *  if non-NULL, else the buffer-layer index (0-based). */
void modelLoadStateDictBuffers(layer_t **model, size_t numLayers, const stateDictBuffers_t *entries,
                               size_t numEntries);

#endif /* STATE_DICT_API_H */
