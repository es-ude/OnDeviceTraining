#define SOURCE_FILE "NPY_LOADER"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CheckedSize.h"
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

static char *npyReadHeaderText(FILE *f, const char *path) {
    static const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    uint8_t preamble[6];
    if (fread(preamble, 1, sizeof(preamble), f) != sizeof(preamble) ||
        memcmp(preamble, magic, sizeof(magic)) != 0) {
        PRINT_ERROR("npyReadHeader: %s: is not a .npy file", path);
        exit(1);
    }

    uint8_t version[2];
    if (fread(version, 1, sizeof(version), f) != sizeof(version)) {
        PRINT_ERROR("npyReadHeader: %s: truncated preamble", path);
        exit(1);
    }
    unsigned major = version[0];
    unsigned minor = version[1];
    if ((major < 1u || major > 3u) || minor != 0u) {
        PRINT_ERROR("npyReadHeader: %s: unsupported .npy version %u.%u", path, major, minor);
        exit(1);
    }

    /* 1.x stores a u16 length, 2.x/3.x a u32; both little-endian. */
    size_t lengthBytes = (major == 1u) ? 2u : 4u;
    uint8_t lengthField[4] = {0};
    if (fread(lengthField, 1, lengthBytes, f) != lengthBytes) {
        PRINT_ERROR("npyReadHeader: %s: truncated preamble", path);
        exit(1);
    }
    uint32_t length = (uint32_t)lengthField[0] | ((uint32_t)lengthField[1] << 8) |
                      ((uint32_t)lengthField[2] << 16) | ((uint32_t)lengthField[3] << 24);

    if (length > NPY_MAX_HEADER_BYTES) {
        PRINT_ERROR("npyReadHeader: %s: header length %u exceeds the %u-byte cap", path,
                    (unsigned)length, (unsigned)NPY_MAX_HEADER_BYTES);
        exit(1);
    }

    /* Measured before reserving so a lying length never reaches the allocator
     * or a short read. length <= the cap keeps the comparison 32-bit-long safe. */
    long headerStart = ftell(f);
    if (headerStart < 0 || fseek(f, 0, SEEK_END) != 0) {
        PRINT_ERROR("npyReadHeader: %s: cannot determine the file size", path);
        exit(1);
    }
    long fileSize = ftell(f);
    if (fileSize < headerStart || fseek(f, headerStart, SEEK_SET) != 0) {
        PRINT_ERROR("npyReadHeader: %s: cannot determine the file size", path);
        exit(1);
    }
    if (fileSize - headerStart < (long)length) {
        PRINT_ERROR("npyReadHeader: %s: header length %u runs past the end of the file", path,
                    (unsigned)length);
        exit(1);
    }

    char *header = reserveMemory((size_t)length + 1u);
    if (header == NULL) {
        PRINT_ERROR("npyReadHeader: %s: cannot reserve %u bytes for the header", path,
                    (unsigned)length + 1u);
        exit(1);
    }
    if (fread(header, 1, length, f) != length) {
        PRINT_ERROR("npyReadHeader: %s: short read of the %u-byte header", path, (unsigned)length);
        exit(1);
    }
    header[length] = '\0';
    return header;
}

#define NPY_HEADER_FAIL(path, fmt, ...)                                                            \
    do {                                                                                           \
        PRINT_ERROR("npyReadHeader: %s: " fmt, path, ##__VA_ARGS__);                               \
        exit(1);                                                                                   \
    } while (0)

static const char *skipSpaces(const char *p) {
    while (*p == ' ') {
        p++;
    }
    return p;
}

static bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

/* Returns the start of the key's value; the caller parses the value. A second
 * occurrence is rejected: numpy's literal_eval keeps the LAST duplicate, so a
 * first-match read could disagree with numpy (e.g. fortran_order) silently. */
static const char *findValue(const char *header, const char *key, const char *path) {
    const char *p = strstr(header, key);
    if (p == NULL) {
        NPY_HEADER_FAIL(path, "no %s key", key);
    }
    p += strlen(key);
    if (strstr(p, key) != NULL) {
        NPY_HEADER_FAIL(path, "duplicate %s key", key);
    }
    p = skipSpaces(p);
    if (*p != ':') {
        NPY_HEADER_FAIL(path, "malformed header dict: no ':' after %s", key);
    }
    return skipSpaces(p + 1);
}

/* Both supported descrs, '<f4' and '<i4', are 4 bytes wide. */
#define NPY_ELEMENT_BYTES 4u

static dtype_t parseDescr(const char *p, const char *path) {
    if (strncmp(p, "'<f4'", 5) == 0) {
        return FLOAT_32;
    }
    if (strncmp(p, "'<i4'", 5) == 0) {
        return INT_32;
    }
    NPY_HEADER_FAIL(path, "unsupported dtype (only '<f4' and '<i4' are supported)");
}

static void parseFortranOrder(const char *p, const char *path) {
    if (strncmp(p, "True", 4) == 0) {
        NPY_HEADER_FAIL(path, "fortran_order True is not supported");
    }
    if (strncmp(p, "False", 5) != 0 || (p[5] != ',' && p[5] != ' ' && p[5] != '}')) {
        NPY_HEADER_FAIL(path, "fortran_order must be False");
    }
}

/* shape := '(' ws* [ item ( ws* ',' ws* item )* [ ws* ',' ] ] ws* ')' with item := digit+:
 * numpy writes (2, 3) and (3,), examples/_shared/npy_writer.c writes (2,3,). */
static void parseShape(const char *p, const char *path, npyHeader_t *out) {
    if (*p != '(') {
        NPY_HEADER_FAIL(path, "malformed shape tuple");
    }
    p = skipSpaces(p + 1);
    size_t numberOfDims = 0;
    while (*p != ')') {
        if (!isDigit(*p)) {
            NPY_HEADER_FAIL(path, "malformed shape tuple");
        }
        if (numberOfDims == NPY_MAX_DIMS) {
            NPY_HEADER_FAIL(path, "shape has more than %u dimensions", (unsigned)NPY_MAX_DIMS);
        }
        size_t value = 0;
        while (isDigit(*p)) {
            if (!checkedMulSize(value, 10u, &value) ||
                !checkedAddSize(value, (size_t)(*p - '0'), &value)) {
                NPY_HEADER_FAIL(path, "shape dimension overflows size_t");
            }
            p++;
        }
        out->dims[numberOfDims++] = value;
        p = skipSpaces(p);
        if (*p == ',') {
            p = skipSpaces(p + 1);
        } else if (*p != ')') {
            NPY_HEADER_FAIL(path, "malformed shape tuple");
        }
    }
    out->numberOfDims = numberOfDims;
}

void npyReadHeader(FILE *f, const char *path, npyHeader_t *out) {
    memset(out, 0, sizeof(*out));
    char *header = npyReadHeaderText(f, path);

    const char *descr = findValue(header, "'descr'", path);
    const char *fortranOrder = findValue(header, "'fortran_order'", path);
    const char *shape = findValue(header, "'shape'", path);

    out->dtype = parseDescr(descr, path);
    parseFortranOrder(fortranOrder, path);
    parseShape(shape, path, out);
    freeReservedMemory(header);

    size_t numberOfElements = 1;
    for (size_t i = 0; i < out->numberOfDims; i++) {
        if (!checkedMulSize(numberOfElements, out->dims[i], &numberOfElements)) {
            NPY_HEADER_FAIL(path, "element count of the shape overflows size_t");
        }
    }
    size_t payloadBytes;
    if (!checkedMulSize(numberOfElements, NPY_ELEMENT_BYTES, &payloadBytes)) {
        NPY_HEADER_FAIL(path, "payload size of the shape overflows size_t");
    }
    out->numberOfElements = numberOfElements;
    out->payloadBytes = payloadBytes;

    /* Both loaders fread right after this returns, so f goes back to the payload start. */
    long payloadStart = ftell(f);
    if (payloadStart < 0 || fseek(f, 0, SEEK_END) != 0) {
        NPY_HEADER_FAIL(path, "cannot determine the file size");
    }
    long fileSize = ftell(f);
    if (fileSize < payloadStart || fseek(f, payloadStart, SEEK_SET) != 0) {
        NPY_HEADER_FAIL(path, "cannot determine the file size");
    }
    /* Equality, not >=: np.save and npy_writer.c never pad, so surplus bytes mean a
     * descr or shape that does not describe the data. */
    size_t remaining = (size_t)(fileSize - payloadStart);
    if (remaining != payloadBytes) {
        NPY_HEADER_FAIL(path, "payload is %zu bytes, shape needs %zu", remaining, payloadBytes);
    }
}
