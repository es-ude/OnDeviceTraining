#ifndef ENV5_RUNTIME_SQUARE_H
#define ENV5_RUNTIME_SQUARE_H

#include <stddef.h>

float squareFloat32(float a);

/*! Number of squares counted so far; only counts when built with TRACK_INSTRUCTIONS. */
size_t getSquareInstructionCounter();

#endif // ENV5_RUNTIME_SQUARE_H
