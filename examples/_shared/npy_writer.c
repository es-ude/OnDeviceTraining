#define SOURCE_FILE "npy_writer"

#include "npy_writer.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Append a printf-formatted string into `buf` starting at `offset`. Returns
 * the new offset on success, or -1 on truncation, encoding error, or if the
 * caller passed an already-out-of-range offset. Avoids the unsigned underflow
 * trap of computing `buf_size - offset` when `offset >= buf_size`. */
static int appendBounded(char *buf, size_t buf_size, int offset, const char *fmt, ...) {
    if (offset < 0 || (size_t)offset >= buf_size) {
        return -1;
    }
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(buf + offset, buf_size - (size_t)offset, fmt, args);
    va_end(args);
    if (written < 0 || (size_t)written >= buf_size - (size_t)offset) {
        return -1;
    }
    return offset + written;
}

static int writeNpy(const char *path, const void *data, const size_t *shape, size_t ndim,
                    const char *dtype_descr, /* "<f4" or "<i4" */
                    size_t elem_size) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return 1;
    }

    /* Build dict header: {'descr': '<f4', 'fortran_order': False, 'shape': (a, b, c), } */
    char shape_buf[256];
    int shape_len = 0;
    shape_len = appendBounded(shape_buf, sizeof(shape_buf), shape_len, "(");
    for (size_t i = 0; i < ndim; ++i) {
        shape_len = appendBounded(shape_buf, sizeof(shape_buf), shape_len, "%zu,", shape[i]);
    }
    if (ndim == 0) {
        shape_len = appendBounded(shape_buf, sizeof(shape_buf), shape_len, ",");
    }
    /* ndim == 1: already has trailing comma from the loop above */
    /* ndim > 1:  numpy accepts trailing comma in tuple literal */
    shape_len = appendBounded(shape_buf, sizeof(shape_buf), shape_len, ")");
    if (shape_len < 0) {
        fclose(f);
        return 6;
    }

    char dict[512];
    int dict_len =
        snprintf(dict, sizeof(dict), "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
                 dtype_descr, shape_buf);
    if (dict_len < 0 || dict_len >= (int)sizeof(dict)) {
        fclose(f);
        return 2;
    }

    /* npy v1.0: prefix (magic 6 + version 2 + header_len 2 = 10 bytes) + header_text
     * must be a multiple of 16. Header text = dict + padding spaces + newline terminator.
     */
    const int prefix = 10;
    /* Round up so that prefix + padded_len is a multiple of 16. */
    int padded = ((prefix + dict_len + 1 + 15) / 16) * 16 - prefix;
    /* padded = header_len field value = len(dict) + spaces + newline */
    int pad_spaces = padded - dict_len - 1; /* -1 for the trailing newline */
    if (pad_spaces < 0) {
        fclose(f);
        return 3;
    }

    char *padded_dict = malloc((size_t)padded + 1);
    if (!padded_dict) {
        fclose(f);
        return 4;
    }
    memcpy(padded_dict, dict, (size_t)dict_len);
    memset(padded_dict + dict_len, ' ', (size_t)pad_spaces);
    padded_dict[dict_len + pad_spaces] = '\n';

    /* Write magic + version + header_len (little-endian u16) + header + data */
    const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const unsigned char ver[2] = {0x01, 0x00};
    unsigned char hdrlen[2];
    hdrlen[0] = (unsigned char)(padded & 0xff);
    hdrlen[1] = (unsigned char)((padded >> 8) & 0xff);

    if (fwrite(magic, 1, 6, f) != 6) {
        goto fail;
    }
    if (fwrite(ver, 1, 2, f) != 2) {
        goto fail;
    }
    if (fwrite(hdrlen, 1, 2, f) != 2) {
        goto fail;
    }
    if (fwrite(padded_dict, 1, (size_t)padded, f) != (size_t)padded) {
        goto fail;
    }

    size_t total_elems = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elems *= shape[i];
    }
    if (fwrite(data, elem_size, total_elems, f) != total_elems) {
        goto fail;
    }

    free(padded_dict);
    fclose(f);
    return 0;
fail:
    free(padded_dict);
    fclose(f);
    return 5;
}

int npyWriteFloat32(const char *path, const float *data, const size_t *shape, size_t ndim) {
    return writeNpy(path, data, shape, ndim, "<f4", sizeof(float));
}

int npyWriteInt32(const char *path, const int32_t *data, const size_t *shape, size_t ndim) {
    return writeNpy(path, data, shape, ndim, "<i4", sizeof(int32_t));
}
