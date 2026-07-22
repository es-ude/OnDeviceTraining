#ifndef DESERIALIZE_H
#define DESERIALIZE_H

#include "Layer.h"

/*! Deserializes tensor from a given file.
 *
 * \param tensor: Pointer to tensor to write deserialized data to
 * \param f: Pointer of file to deserialize from
 */
void deserializeTensor(tensor_t *tensor, FILE *f);

/*! Deserializes parameter from a given file.
 *
 *  Grad-presence TOLERANT (#380 PR3): the file's per-parameter hasGrad byte
 *  may legitimately disagree with the skeleton's construction for a
 *  full-model checkpoint that mixes frozen and trainable layers. File
 *  hasGrad=1 into a frozen skeleton (parameter->grad == NULL) parses and
 *  discards the grad record (no allocation; the stream stays positionally
 *  in sync for the next record). File hasGrad=0 into a trainable skeleton
 *  leaves the skeleton's already-zero-initialized grad untouched
 *  (optimizerZeroGrad re-zeros it before every batch regardless). The
 *  discard path requires a seekable stream (fseek/ftell).
 *
 * \param parameter: Pointer to parameter to write deserialized data to
 * \param f: Pointer of file to deserialize from
 */
void deserializeParameter(parameter_t *parameter, FILE *f);

/*! Deserializes model from a given file.
 *
 *  Requires a seekable stream (fseek/ftell) — see deserializeParameter's
 *  grad-presence tolerance (#380 PR3), which discards a mismatched grad
 *  record via fseek rather than allocating a scratch buffer for it.
 *
 * \param model: Pointer to model to write deserialized data to
 * \param sizeModel: Size of model
 * \param f: Pointer of file to deserialize from
 */
void deserializeModel(layer_t **model, size_t sizeModel, FILE *f);

#endif // DESERIALIZE_H
