#include "CSVHelper.h"
#include "DeathTest.h"
#include "unity.h"
#include <stdlib.h>
#include <string.h>

#define CSV_INPUT_PATH "../../../../../test/unit/csv/input.csv"
#define CSV_OUTPUT_PATH "../../../../../test/unit/csv/output.csv"

void testCSVReadRowsByBufferSize() {
    char *filePath = CSV_INPUT_PATH;

    size_t numberOfRows = 1;

    csvData_t csvData;

    char row[MAX_ROW_SIZE];
    char *rows[1];
    rows[0] = row;
    size_t entriesInRow[1];

    setCSVData(&csvData, rows, numberOfRows, entriesInRow);

    csvReadRowsByBufferSize(filePath, &csvData);

    TEST_ASSERT_EQUAL_STRING("1, 2, 3, 4, 5, 6", csvData.rows[0]);
    TEST_ASSERT_EQUAL_size_t(6, csvData.entriesInRow[0]);
}

void testCSVParseBufferAsFloat() {
    char *filePath = CSV_INPUT_PATH;

    size_t numberOfRows = 1;

    csvData_t csvData;

    char row[MAX_ROW_SIZE];
    char *rows[numberOfRows];
    rows[0] = row;
    size_t entriesInRow[1];

    setCSVData(&csvData, rows, numberOfRows, entriesInRow);

    csvReadRowsByBufferSize(filePath, &csvData);

    float *rowsAsFloat[numberOfRows];
    float rowFloat[6];
    rowsAsFloat[0] = rowFloat;

    csvParseBufferAsFloat(&csvData, rowsAsFloat);
    float expected[] = {1, 2, 3, 4, 5, 6};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, rowsAsFloat[0], 6);
}

void testCSVWriteRowsByBufferSize() {
    char *filePath = CSV_OUTPUT_PATH;

    size_t numberOfRows = 1;

    csvData_t csvData;

    char row[] = {"1, 2, 3, 4, 5, 6"};
    char *rows[1];
    rows[0] = row;
    size_t entriesInRow[] = {6};

    setCSVData(&csvData, rows, numberOfRows, entriesInRow);

    csvWriteRowsByBufferSize(filePath, &csvData, "w");

    char actual[MAX_ROW_SIZE];

    FILE *fp;
    fp = fopen(filePath, "r");
    char *ptr = fgets(actual, MAX_ROW_SIZE, fp);
    fclose(fp);

    TEST_ASSERT_EQUAL_STRING("1, 2, 3, 4, 5, 6", ptr);
}

static void readRows(char *filePath, size_t numberOfRows) {
    char row0[MAX_ROW_SIZE];
    char row1[MAX_ROW_SIZE];
    char *rows[2] = {row0, row1};
    size_t entriesInRow[2];
    csvData_t csvData;
    setCSVData(&csvData, rows, numberOfRows, entriesInRow);
    csvReadRowsByBufferSize(filePath, &csvData);
}

void testCSVReadExitsOnMissingFile() {
    ASSERT_EXITS_WITH_OUTPUT(1, "Could not open file", readRows("no/such/file.csv", 1));
}

/* input.csv holds one row; asking for two must stop instead of counting entries of a
 * NULL row. */
void testCSVReadExitsOnPrematureEof() {
    ASSERT_EXITS_WITH_OUTPUT(1, "ended after 1 of 2 rows", readRows(CSV_INPUT_PATH, 2));
}

static void parseRow(char *row) {
    char *rows[1] = {row};
    size_t entriesInRow[1] = {3};
    csvData_t csvData;
    setCSVData(&csvData, rows, 1, entriesInRow);
    float values[3];
    float *output[1] = {values};
    csvParseBufferAsFloat(&csvData, output);
}

void testCSVParseExitsOnFieldWithoutANumber() {
    char row[] = "1, x, 3";
    ASSERT_EXITS_WITH_OUTPUT(1, "row 0 field 1 is not a number", parseRow(row));
}

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testCSVReadRowsByBufferSize);
    RUN_TEST(testCSVParseBufferAsFloat);
    RUN_TEST(testCSVWriteRowsByBufferSize);
    RUN_TEST(testCSVReadExitsOnMissingFile);
    RUN_TEST(testCSVReadExitsOnPrematureEof);
    RUN_TEST(testCSVParseExitsOnFieldWithoutANumber);
    return UNITY_END();
}
