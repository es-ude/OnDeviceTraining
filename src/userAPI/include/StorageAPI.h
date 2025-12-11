#ifndef STORAGEAPI_H
#define STORAGEAPI_H

#include <stddef.h>
#include <stdlib.h>

#include "Tensor.h"

void **reserveMemory(size_t numberOfBytes);

void freeReservedMemory(void *ptr);



#endif //STORAGEAPI_H
