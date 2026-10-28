#define SOURCE_FILE "NPY_LOADER"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "DataLoader.h"
#include "NPYLoader.h"
#include "StorageApi.h"

FILE *openNPYFile(char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        PRINT_ERROR("File doesn't exist!");
        exit(1);
    }
    return f;
}

char *npyReadHeaderText(FILE *f, const char *path) {
    static const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    uint8_t preamble[6];
    if (fread(preamble, 1, sizeof(preamble), f) != sizeof(preamble) ||
        memcmp(preamble, magic, sizeof(magic)) != 0) {
        PRINT_ERROR("npyReadHeaderText: %s: is not a .npy file", path);
        exit(1);
    }

    uint8_t version[2];
    if (fread(version, 1, sizeof(version), f) != sizeof(version)) {
        PRINT_ERROR("npyReadHeaderText: %s: truncated preamble", path);
        exit(1);
    }
    unsigned major = version[0];
    unsigned minor = version[1];
    if ((major < 1u || major > 3u) || minor != 0u) {
        PRINT_ERROR("npyReadHeaderText: %s: unsupported .npy version %u.%u", path, major, minor);
        exit(1);
    }

    /* 1.x stores a u16 length, 2.x/3.x a u32; both little-endian. */
    size_t lengthBytes = (major == 1u) ? 2u : 4u;
    uint8_t lengthField[4] = {0};
    if (fread(lengthField, 1, lengthBytes, f) != lengthBytes) {
        PRINT_ERROR("npyReadHeaderText: %s: truncated preamble", path);
        exit(1);
    }
    uint32_t length = (uint32_t)lengthField[0] | ((uint32_t)lengthField[1] << 8) |
                      ((uint32_t)lengthField[2] << 16) | ((uint32_t)lengthField[3] << 24);

    if (length > NPY_MAX_HEADER_BYTES) {
        PRINT_ERROR("npyReadHeaderText: %s: header length %u exceeds the %u-byte cap", path,
                    (unsigned)length, (unsigned)NPY_MAX_HEADER_BYTES);
        exit(1);
    }

    /* Measured before reserving so a lying length never reaches the allocator
     * or a short read. length <= the cap keeps the comparison 32-bit-long safe. */
    long headerStart = ftell(f);
    if (headerStart < 0 || fseek(f, 0, SEEK_END) != 0) {
        PRINT_ERROR("npyReadHeaderText: %s: cannot determine the file size", path);
        exit(1);
    }
    long fileSize = ftell(f);
    if (fileSize < headerStart || fseek(f, headerStart, SEEK_SET) != 0) {
        PRINT_ERROR("npyReadHeaderText: %s: cannot determine the file size", path);
        exit(1);
    }
    if (fileSize - headerStart < (long)length) {
        PRINT_ERROR("npyReadHeaderText: %s: header length %u runs past the end of the file", path,
                    (unsigned)length);
        exit(1);
    }

    char *header = reserveMemory((size_t)length + 1u);
    if (header == NULL) {
        PRINT_ERROR("npyReadHeaderText: %s: cannot reserve %u bytes for the header", path,
                    (unsigned)length + 1u);
        exit(1);
    }
    if (fread(header, 1, length, f) != length) {
        PRINT_ERROR("npyReadHeaderText: %s: short read of the %u-byte header", path,
                    (unsigned)length);
        exit(1);
    }
    header[length] = '\0';
    return header;
}

static dtype_t parseHeaderDescription(char *s) {
    if (strcmp(s, "<f4") == 0) {
        return FLOAT_32;
    }
    if (strcmp(s, "<i4") == 0) {
        return INT_32;
    }

    fprintf(stderr, "Unsupported dtype: %s\n", s);
    exit(1);
}

dtype_t getDTypeFromHeader(char *header) {
    char descr[8];
    sscanf(strstr(header, "'descr'"), "'descr': '%7[^']'", descr);
    dtype_t dtype = parseHeaderDescription(descr);
    return dtype;
}

size_t getNumberOfDimsFromHeader(char *header) {
    char *p = strstr(header, "shape");
    p = strchr(p, '(') + 1;

    size_t numberOfDims = 0;

    while (*p != ')') {
        while (*p == ' ') {
            p++;
        }

        if (*p >= '0' && *p <= '9') {
            numberOfDims++;
            while (*p >= '0' && *p <= '9') {
                p++;
            }
        }

        if (*p == ',') {
            p++;
        }
    }

    return numberOfDims;
}

void getShapeFromHeader(shape_t *shape, size_t *dims, size_t *orderOfDims, char *header,
                        size_t numberOfDims) {
    shape->numberOfDimensions = numberOfDims;
    shape->dimensions = dims;
    shape->orderOfDimensions = orderOfDims;

    char *p = strstr(header, "shape");
    p = strchr(p, '(');
    p++;

    for (size_t i = 0; i < numberOfDims; i++) {
        while (*p == ' ') {
            p++;
        }

        char *end;
        size_t value = strtoull(p, &end, 10);

        shape->dimensions[i] = value;
        shape->orderOfDimensions[i] = i;

        p = end;

        if (*p == ',') {
            p++;
        }
    }
}
