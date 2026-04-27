#ifndef MODELSTORAGE_H
#define MODELSTORAGE_H

#include "Layer.h"

/*! Serializes tensor to a given file.
 *
 * \param tensor: Pointer to tensor
 * \param f: Pointer of file to serialize to
 */
void serializeTensor(tensor_t *tensor, FILE *f);

/*! Serializes parameter to a given file.
 *
 * \param parameter: Pointer to parameter
 * \param f: Pointer of file to serialize to
 */
void serializeParameter(parameter_t *parameter, FILE *f);

/*! Serializes model to a given file.
 *
 * \param model: Pointer to model
 * \param sizeModel: Size of model
 * \param f: Pointer of file to serialize to
 */
void serializeModel(layer_t **model, size_t sizeModel, FILE *f);

#endif // MODELSTORAGE_H
