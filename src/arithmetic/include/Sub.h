#ifndef ENV5_RUNTIME_SUB_H
#define ENV5_RUNTIME_SUB_H

#include <stddef.h>

float subFloat32s(float a, float b);

/*! Number of subtractions counted so far; only counts when built with TRACK_INSTRUCTIONS. */
size_t getSubInstructionCounter();

#endif // ENV5_RUNTIME_SUB_H
