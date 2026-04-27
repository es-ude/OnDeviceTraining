#ifndef NPYLOADER_H
#define NPYLOADER_H

/*! Opens a given path to .npy file.
 *
 * \param path: Path to .npy file
 *
 * \returns Pointer to file
 */
FILE *openNPYFile(char *path);

/*! Check magic if file is actually a .npy file.
 *
 * \param f: Pointer to file
 */
void checkMagic(FILE *f);

/*! Read size of header.
 *
 * \param f: Pointer to file
 *
 * \returns Size of header
 */
uint32_t readHeaderSize(FILE *f);

/*! Read header of .npy file with given header size.
 *
 * \param header: String of header
 * \param headerSize: Size of header
 * \param f: Pointer to file
 */
void readHeader(char *header, uint32_t headerSize, FILE *f);

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
