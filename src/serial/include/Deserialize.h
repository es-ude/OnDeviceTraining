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
 * \param parameter: Pointer to parameter to write deserialized data to
 * \param f: Pointer of file to deserialize from
 */
void deserializeParameter(parameter_t *parameter, FILE *f);

/*! Deserializes model from a given file.
 *
 * \param model: Pointer to model to write deserialized data to
 * \param sizeModel: Size of model
 * \param f: Pointer of file to deserialize from
 */
void deserializeModel(layer_t **model, size_t sizeModel, FILE *f);

#endif // DESERIALIZE_H
