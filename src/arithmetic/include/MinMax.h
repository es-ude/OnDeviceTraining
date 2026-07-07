#ifndef MINMAX_H
#define MINMAX_H

#include <stddef.h>
#include <stdint.h>

float absFloat32(float a);
float maxFloat32s(float a, float b);

float findAbsMaxFloat(uint8_t *bytes, size_t numberOfElements);
float findMaxFloat(uint8_t *bytes, size_t numberOfElements);
float findMinFloat(uint8_t *bytes, size_t numberOfElements);

int32_t findMaxInt32(uint8_t *bytes, size_t numberOfElements);
int32_t findMinInt32(uint8_t *bytes, size_t numberOfElements);

#endif // MINMAX_H
