#ifndef ODT_MODEL_VALIDATION_API_H
#define ODT_MODEL_VALIDATION_API_H

#include <stdbool.h>
#include <stddef.h>

#include "Model.h"

/*! Opt-in; not wired into the training loop. Checks that the model array and
 *  its elements are non-NULL. Logs violations and returns false; never exits. */
bool validateModelQuantization(const model_t *m);

#endif // ODT_MODEL_VALIDATION_API_H
