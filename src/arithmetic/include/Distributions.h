#ifndef ODT_DISTRIBUTIONS_H
#define ODT_DISTRIBUTIONS_H

#include <stddef.h>

#include "RNG.h"

typedef enum {
    ZEROS,
    ONES,
    UNIFORM,
    NORMAL,
    XAVIER_UNIFORM,
    XAVIER_NORMAL,
    KAIMING_UNIFORM,
    KAIMING_NORMAL
} distributionType_t;

/*! An initialization recipe for initDistribution() (TensorApi.h). Read the union
 * member matching `type`; ZEROS and ONES take no params. */
typedef struct {
    distributionType_t type;
    union {
        struct {
            float min, max;
        } uniform;
        struct {
            float mean, stddev;
        } normal;
        struct {
            float gain;
            size_t fanIn, fanOut;
        } xavier;
        struct {
            float gain;
            size_t fanMode;
        } kaiming;
    } params;
} distribution_t;

/* Single draws from a caller-owned, non-NULL stream (initDistribution checks it).
 * Normal: Box-Muller on two unit floats, u1 redrawn while <= 1e-7. Uniform: [min, max).
 * Kaiming/Xavier use torch's std/limit formulas; gain is the nonlinearity gain. */
float randomNormalCtx(rngStream_t *rs, float mean, float standardDeviation);
float randomUniformCtx(rngStream_t *rs, float min, float max);
float kaimingNormal(float gain, size_t fanMode, rngStream_t *rs);
float kaimingUniform(float gain, size_t fanMode, rngStream_t *rs);
float xavierNormal(float gain, size_t fanIn, size_t fanOut, rngStream_t *rs);
float xavierUniform(float gain, size_t fanIn, size_t fanOut, rngStream_t *rs);

#endif // ODT_DISTRIBUTIONS_H
