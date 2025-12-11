#include "StorageAPI.h"

void **reserveMemory(size_t numberOfBytes) {
    void *ptr = calloc(1, numberOfBytes);
    void **handle = malloc(sizeof(void *));
    *handle = ptr;
    return handle;
}

void freeReservedMemory(void *ptr) {
    free(ptr);
}

