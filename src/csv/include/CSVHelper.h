#ifndef CSVHELPER_H
#define CSVHELPER_H

#define MAX_ROW_SIZE 1000

#include <stddef.h>

typedef struct csvData {
    char **rows;          /*!< Array of row pointers*/
    size_t numberOfRows;  /*!< Number of rows*/
    size_t *entriesInRow; /*!< Number of entries in each row*/
} csvData_t;

/*!
 * Sets values of given csvData.
 *
 * \param csvData: Pointer to csvData struct
 * \param rows: Pointer to array of row pointers
 * \param numberOfRows: Number of rows
 * \param entriesInRow: Number of entries in each row
 */
void setCSVData(csvData_t *csvData, char **rows, size_t numberOfRows, size_t *entriesInRow);

/*!
 * Reads file until buffer is full.
 *
 * \param filePath: Path to file
 * \param csvData: Pointer to csvData buffer
 */
void csvReadRowsByBufferSize(char *filePath, csvData_t *csvData);

/*!
 * Parses csvData buffer as floats.
 *
 * \param csvData: Pointer to buffer
 * \param output: Pointer to float array for outputs
 */
void csvParseBufferAsFloat(csvData_t *csvData, float **output);

/*!
 * Writes buffer to file.
 *
 * \param filePath: Path to file
 * \param csvData: Pointer to csvData buffer
 * \param mode: Mode of write (a = append, w = write)
 */
void csvWriteRowsByBufferSize(char *filePath, csvData_t *csvData, char *mode);

#endif // CSVHELPER_H
