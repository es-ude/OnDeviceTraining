#ifndef INTERNALNPYLOADERAPI_H
#define INTERNALNPYLOADERAPI_H

/*!
 * Fills rowShape with all dimensions of totalShape, except the first one.
 * The first dimensions describes the total amount of rows in the dataset.
 *
 * \param totalShape: Pointer to total shape of dataset
 * \param rowShape: Pointer to row shape to be filled
 * \return void
 */
static void getRowShape(shape_t *totalShape, shape_t *rowShape);

/*!
 * Initializes quantization by given datatype.
 *
 * \param dtype: Enum of accepted datatypes
 * \return Pointer to quantization
 */
static quantization_t *initQByDType(dtype_t dtype);

#endif // INTERNALNPYLOADERAPI_H
