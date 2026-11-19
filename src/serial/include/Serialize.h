#ifndef MODELSTORAGE_H
#define MODELSTORAGE_H

#include "Layer.h"
#include "Model.h"

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
 * \param m: Model to serialize
 * \param f: Pointer of file to serialize to
 */
void serializeModel(const model_t *m, FILE *f);

#endif // MODELSTORAGE_H
