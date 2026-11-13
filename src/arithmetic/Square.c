#define SOURCE_FILE "SQUARE"

#ifdef TRACK_INSTRUCTIONS
#define SQUARE_FUNC_FLOAT squareFloatWithInstructionCounter
#else
#define SQUARE_FUNC_FLOAT squareFloat
#endif

#include "Square.h"

size_t squareInstructionCounter = 0;

float squareFloat(float a) {
    return a * a;
}

float squareFloatWithInstructionCounter(float a) {
    ++squareInstructionCounter;
    return a * a;
}

float squareFloat32(float a) {
    return SQUARE_FUNC_FLOAT(a);
}

size_t getSquareInstructionCounter() {
    return squareInstructionCounter;
}
