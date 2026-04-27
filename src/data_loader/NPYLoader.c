#define SOURCE_FILE "NPY_LOADER"

#include "string.h"
#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "DataLoader.h"
#include "NPYLoader.h"

FILE *openNPYFile(char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        PRINT_ERROR("File doesn't exist!");
        exit(1);
    }
    return f;
}

void checkMagic(FILE *f) {
    char magic[7] = {0};
    fread(magic, 1, 6, f);
    if (strcmp(magic, "\x93NUMPY") != 0) {
        fprintf(stderr, "Not a .npy file\n");
        exit(1);
    }
}

uint32_t readHeaderSize(FILE *f) {
    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);

    uint32_t header_len;
    if (major == 1) {
        uint16_t hl16;
        fread(&hl16, 2, 1, f);
        header_len = hl16;
    } else {
        fread(&header_len, 4, 1, f);
    }

    return header_len;
}

void readHeader(char *header, uint32_t headerSize, FILE *f) {
    fread(header, 1, headerSize, f);
    header[headerSize] = 0;
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
