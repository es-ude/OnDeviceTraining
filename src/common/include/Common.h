#ifndef ODT_COMMON_H
#define ODT_COMMON_H

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#ifndef SOURCE_FILE
#define SOURCE_FILE "no Source file defined!"
#endif

#define DLEVEL 0

#ifdef DEBUG_MODE_DEBUG
#define DLEVEL 3
#endif

#ifdef DEBUG_MODE_INFO
#define DLEVEL 2
#endif

#ifdef DEBUG_MODE_ERROR
#define DLEVEL 1
#endif

#define PRINT_DEBUG(str, ...) \
do { \
if (DLEVEL >= 3) { \
printf("\033[0;33m[%s: %s] ", SOURCE_FILE, __FUNCTION__); \
printf(str, ##__VA_ARGS__); \
printf("\033[0m\n"); \
} \
} while (false)

#define PRINT_INFO(str, ...)                                                    \
    do {                                                                        \
        if (DLEVEL >= 2) {                                                 \
            printf("[%s: %s] ", SOURCE_FILE, __FUNCTION__); \
            printf(str, ##__VA_ARGS__); \
            printf("\n"); \
        } \
    } while (false)

#define PRINT_ERROR(str, ...) \
    do { \
        if (DLEVEL >= 1) { \
            printf("\033[0;31m[%s: %s] ", SOURCE_FILE, __FUNCTION__); \
            printf(str, ##__VA_ARGS__); \
            printf("\033[0m\n"); \
        } \
    } while (false)                                      \


// TODO

#define PRINT_BYTE_ARRAY(prefix, byteArray, numberOfBytes)                                         \
    do {                                                                                           \
        printf("[%s: %s] ", SOURCE_FILE, __FUNCTION__);                                            \
        printf(prefix);                                                                            \
        for (size_t index = 0; index < numberOfBytes; index++) {                                   \
            printf("0x%02X ", byteArray[index]);                                                   \
        }                                                                                          \
        printf("\n");                                                                              \
    } while (false)


#endif
