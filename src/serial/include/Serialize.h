#ifndef MODELSTORAGE_H
#define MODELSTORAGE_H

#include <stdio.h>

#include "Layer.h"
#include "Model.h"

/* Writers for the ODTS format (layout: Serialize.c). Every write is checked;
 * a short write fails fast. */
void serializeTensor(tensor_t *tensor, FILE *f);
void serializeParameter(parameter_t *parameter, FILE *f);
void serializeModel(const model_t *m, FILE *f);

#endif // MODELSTORAGE_H
