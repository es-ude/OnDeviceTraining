#ifndef DESERIALIZE_H
#define DESERIALIZE_H

#include "Layer.h"
#include "Model.h"

/* Readers for the ODTS format. Each loads into an existing compatible
 * skeleton and fails fast on a record that does not fit it. */
void deserializeTensor(tensor_t *tensor, FILE *f);

/*! Present gradients are discarded for frozen parameters; absent gradients
 *  leave destination gradients unchanged. A trainable parameter's grad must
 *  end with the loaded param's dims and axis order, or the load exits.
 *  Discarding gradients requires a seekable stream. */
void deserializeParameter(parameter_t *parameter, FILE *f);

/*! m must match the file's layer count and wiring. Requires a seekable stream
 *  whenever a grad record is discarded (see deserializeParameter). */
void deserializeModel(const model_t *m, FILE *f);

#endif // DESERIALIZE_H
