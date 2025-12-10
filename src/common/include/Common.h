#ifndef ODT_COMMON_H
#define ODT_COMMON_H

#include <stdbool.h>
#include <stdio.h>
#include <string.h>


#ifndef SOURCE_FILE
#define SOURCE_FILE "no Source file defined!"
#endif

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#define PRINT_INFO(str, ...)                                                    \
    do {                                                                        \
        if (DEBUG_LEVEL >= 3) {                                                 \
            printf("[%s: %s] ", SOURCE_FILE, __FUNCTION__); \
            printf(str, ##__VA_ARGS__); \
            printf("\n"); \
        } \
    } while (false)

#define PRINT_WARN(str, ...) \
    do { \
        if (DEBUG_LEVEL >= 2) { \
            printf("\033[0;33m[%s: %s] ", SOURCE_FILE, __FUNCTION__); \
            printf(str, ##__VA_ARGS__); \
            printf("\033[0m\n"); \
        } \
    } while (false)

#define PRINT_ERROR(str, ...) \
    do { \
        if (DEBUG_LEVEL >= 1) { \
            printf("\033[0;31m[%s: %s] ", SOURCE_FILE, __FUNCTION__); \
            printf(str, ##__VA_ARGS__); \
            printf("\033[0m\n"); \
        } \
    } while (false)                                      \


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
