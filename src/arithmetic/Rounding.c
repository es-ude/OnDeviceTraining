#define SOURCE_FILE "ROUNDING"

#include <math.h>
#include <stdio.h>

#include "RNG.h"
#include "Rounding.h"

// C round(): rounds half away from zero (C17 7.12.9.6)
int32_t roundHalfAway(float input) {
    return round(input);
}

float randfloat() {
    return rngNextFloat();
}

int32_t roundSRHalfAway(const float input) {
    return roundHalfAway(input + randfloat() - 0.5f);
}

int32_t roundByMode(const float input, const roundingMode_t roundingMode) {
    switch (roundingMode) {
    case HALF_AWAY:
        return roundHalfAway(input);
    case SR_HALF_AWAY:
        return roundSRHalfAway(input);
    }
    return 0;
}

float clamp(float input, float min, float max) {
    if (input < min) {
        return min;
    }
    if (input > max) {
        return max;
    }
    return input;
}
