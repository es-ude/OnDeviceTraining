#define SOURCE_FILE "CSV_HELPER"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CSVHelper.h"
#include "Common.h"

void setCSVData(csvData_t *csvData, char **rows, size_t numberOfRows, size_t *entriesInRow) {
    csvData->rows = rows;
    csvData->entriesInRow = entriesInRow;
    csvData->numberOfRows = numberOfRows;
}

static size_t countCsvEntries(const char *row) {
    size_t count = 1;
    for (size_t i = 0; row[i] != '\0'; i++) {
        if (row[i] == ',') {
            count++;
        }
    }
    return count;
}

void csvReadRowsByBufferSize(char *filePath, csvData_t *csvData) {
    FILE *fp;
    fp = fopen(filePath, "r");
    if (fp == NULL) {
        PRINT_ERROR("Could not open file %s", filePath);
        exit(1);
    }

    for (size_t i = 0; i < csvData->numberOfRows; i++) {
        char *ptr = fgets(csvData->rows[i], MAX_ROW_SIZE, fp);
        if (ptr == NULL) {
            PRINT_ERROR("%s ended after %zu of %zu rows", filePath, i, csvData->numberOfRows);
            exit(1);
        }
        csvData->entriesInRow[i] = countCsvEntries(ptr);
    }

    fclose(fp);
}

void csvParseBufferAsFloat(csvData_t *csvData, float **output) {
    for (size_t i = 0; i < csvData->numberOfRows; i++) {
        char *token = strtok(csvData->rows[i], ",");
        size_t index = 0;

        while (token != NULL && index < csvData->entriesInRow[i]) {
            char *end;
            output[i][index] = strtof(token, &end);
            if (end == token) {
                PRINT_ERROR("row %zu field %zu is not a number: '%s'", i, index, token);
                exit(1);
            }
            index++;
            token = strtok(NULL, ",");
        }
    }
}

void csvWriteRowsByBufferSize(char *filePath, csvData_t *csvData, char *mode) {
    FILE *fp = fopen(filePath, mode);
    if (fp == NULL) {
        PRINT_ERROR("Could not open file %s", filePath);
        exit(1);
    }

    for (size_t i = 0; i < csvData->numberOfRows; i++) {
        fputs(csvData->rows[i], fp);
    }

    fclose(fp);
}
