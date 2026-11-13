#define SOURCE_FILE "SUB"

#ifdef TRACK_INSTRUCTIONS
#define SUB_FUNC_FLOAT subFloatsWithInstructionCounter
#else
#define SUB_FUNC_FLOAT subFloats
#endif

#include "Sub.h"
#include "Common.h"

size_t subInstructionCounter = 0;

float subFloats(float a, float b) {
    return a - b;
}

float subFloatsWithInstructionCounter(float a, float b) {
    ++subInstructionCounter;
    return a - b;
}

float subFloat32s(float a, float b) {
    return SUB_FUNC_FLOAT(a, b);
}

size_t getSubInstructionCounter() {
    return subInstructionCounter;
}
