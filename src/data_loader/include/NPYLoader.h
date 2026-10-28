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

/*! Reads magic, version and header length, checks the length against
 * NPY_MAX_HEADER_BYTES and the file size, and returns the NUL-terminated header
 * text (reserveMemory; the caller frees it with freeReservedMemory). Leaves f at
 * the payload. Every failure prints the path and the reason and exits(1).
 *
 * \param f: File opened at offset 0
 * \param path: Path of f, for the error messages
 *
 * \returns Header text
 */
char *npyReadHeaderText(FILE *f, const char *path);

/*! Gets datatype from header.
 *
 * \param header: String of header
 *
 * \returns Enum of datatype
 */
dtype_t getDTypeFromHeader(char *header);

/*! Get number of dims from header
 *
 * \param header: String of header
 *
 * \returns Number of dims
 */
size_t getNumberOfDimsFromHeader(char *header);

/*! Fills given shape with correct number of dims, dims and orderOfDims.
 *
 * \param shape: Shape buffer
 * \param dims: Dims buffer
 * \param orderOfDims: Order of dims buffer
 * \param header: String of header
 * \param numberOfDims: Number of dims
 */
void getShapeFromHeader(shape_t *shape, size_t *dims, size_t *orderOfDims, char *header,
                        size_t numberOfDims);

#endif // NPYLOADER_H
