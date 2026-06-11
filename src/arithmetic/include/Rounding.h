#ifndef ROUNDING_H
#define ROUNDING_H
#include <stdint.h>

/*! @brief Describes rounding
 * HALF_AWAY = round half away from zero (C round(), C17 7.12.9.6)
 * SR_HALF_AWAY = stochastic rounding: uniform jitter in [-0.5, 0.5) is added
 *                before rounding half away from zero
 */
typedef enum roundingMode { HALF_AWAY, SR_HALF_AWAY } roundingMode_t;

int32_t roundByMode(float input, roundingMode_t roundingMode);

float clamp(float input, float min, float max);

#endif // ROUNDING_H
