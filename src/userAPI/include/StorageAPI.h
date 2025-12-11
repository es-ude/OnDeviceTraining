#ifndef STORAGEAPI_H
#define STORAGEAPI_H

#include <stddef.h>


void **reserveMemory(size_t numberOfBytes);

void freeReservedMemory(void *ptr);



#endif //STORAGEAPI_H
