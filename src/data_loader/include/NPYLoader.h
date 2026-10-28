#ifndef NPYLOADER_H
#define NPYLOADER_H

#include <stddef.h>
#include <stdio.h>

#include "Dataset.h"

/*! Opens a given path to .npy file.
 *
 * \param path: Path to .npy file
 *
 * \returns Pointer to file
 */
FILE *openNPYFile(char *path);

#define NPY_MAX_HEADER_BYTES 10000u /* numpy np.load max_header_size default */
#define NPY_MAX_DIMS 32u            /* numpy 1.x NPY_MAXDIMS */

typedef struct {
    dtype_t dtype;
    size_t numberOfDims;
    size_t dims[NPY_MAX_DIMS];
    size_t numberOfElements;
    size_t payloadBytes;
} npyHeader_t;

/*! Reads the preamble and the header dict of a .npy file: magic, version 1.0/2.0/3.0,
 * a header length within NPY_MAX_HEADER_BYTES and the file size, then the keys
 * 'descr' ('<f4' or '<i4'), 'fortran_order' (False) and 'shape' (at most NPY_MAX_DIMS
 * dimensions, each fitting size_t), then checks that the element count and payload byte
 * count fit size_t and that the bytes after the header equal the payload exactly. Leaves
 * f at the payload. Every failure prints the path and the reason and exits(1).
 *
 * \param f: File opened at offset 0
 * \param path: Path of f, for the error messages
 * \param out: Parsed header
 */
void npyReadHeader(FILE *f, const char *path, npyHeader_t *out);

#endif // NPYLOADER_H
