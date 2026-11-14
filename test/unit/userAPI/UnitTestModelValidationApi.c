#define SOURCE_FILE "UNIT_TEST_MODEL_VALIDATION_API"

#include <stdbool.h>

#include "Layer.h"
#include "ModelValidationApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* The validator reads only the array and its element pointers, so config-less
 * stack layers suffice. */

void testValidatorAcceptsEmptyModel(void) {
    layer_t *model[1] = {NULL}; /* pointer needed; element never read */
    bool valid = validateModelQuantization(&MODEL_CHAIN(model, 0));
    TEST_ASSERT_TRUE_MESSAGE(valid, "empty model (size 0) should be accepted");
}

void testValidatorRejectsNullModel(void) {
    bool valid = validateModelQuantization(NULL);
    TEST_ASSERT_FALSE_MESSAGE(valid, "NULL model pointer should be rejected");
}

void testValidatorRejectsNullLayerArray(void) {
    bool valid = validateModelQuantization(&MODEL_CHAIN(NULL, 0));
    TEST_ASSERT_FALSE_MESSAGE(valid, "NULL layer array should be rejected");
}

void testValidatorRejectsNullElementMidArray(void) {
    layer_t linear = {.type = LINEAR, .config = NULL};
    layer_t *model[2] = {&linear, NULL};
    bool valid = validateModelQuantization(&MODEL_CHAIN(model, 2));
    TEST_ASSERT_FALSE_MESSAGE(valid, "NULL element in model array should be rejected");
}

void testValidatorAcceptsNonNullModel(void) {
    layer_t linear = {.type = LINEAR, .config = NULL};
    layer_t norm = {.type = LAYERNORM, .config = NULL};
    layer_t *model[2] = {&linear, &norm};
    bool valid = validateModelQuantization(&MODEL_CHAIN(model, 2));
    TEST_ASSERT_TRUE(valid);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testValidatorAcceptsEmptyModel);
    RUN_TEST(testValidatorRejectsNullModel);
    RUN_TEST(testValidatorRejectsNullLayerArray);
    RUN_TEST(testValidatorRejectsNullElementMidArray);
    RUN_TEST(testValidatorAcceptsNonNullModel);
    return UNITY_END();
}
