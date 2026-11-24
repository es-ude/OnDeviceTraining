#ifndef CSVHELPER_H
#define CSVHELPER_H

#define MAX_ROW_SIZE 1000

#include <stddef.h>

typedef struct csvData {
    char **rows;          /*!< Array of row pointers*/
    size_t numberOfRows;  /*!< Number of rows*/
    size_t *entriesInRow; /*!< Number of entries in each row*/
} csvData_t;

/*! Borrows caller-owned row buffers and entry-count storage. */
void setCSVData(csvData_t *csvData, char **rows, size_t numberOfRows, size_t *entriesInRow);

/*! Reads numberOfRows rows into buffers of at least MAX_ROW_SIZE bytes and records each
 * row's comma-delimited field count. A missing file or a file with fewer rows exits(1). */
void csvReadRowsByBufferSize(char *filePath, csvData_t *csvData);

/*! Consumes the row strings in place (strtok); output rows must have entriesInRow
 * capacity. A non-empty field without a leading number exits(1); text after the number is
 * ignored. Empty fields are skipped (strtok merges adjacent commas): later fields shift left
 * and the row's trailing output slots stay unwritten. */
void csvParseBufferAsFloat(csvData_t *csvData, float **output);

/*! Writes the caller-supplied row text verbatim; callers supply separators and newlines.
 * mode is the fopen mode ("w" or "a"). */
void csvWriteRowsByBufferSize(char *filePath, csvData_t *csvData, char *mode);

#endif // CSVHELPER_H
