#ifndef ODT_DISTRIBUTIONS_H
#define ODT_DISTRIBUTIONS_H

#include <stddef.h>

#include "RNG.h"

/* Forward declaration to avoid circular includes with Tensor.h */
typedef struct tensor tensor_t;

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

/*! Carries both the kind of distribution and the kind-specific parameters
 * needed to draw values. Used by initDistribution() (TensorApi.h) and by
 * future layer factories that take an initialization recipe.
 *
 * Discriminant: `type`. Read the union member matching that discriminant.
 * ZEROS and ONES need no params; their union slot is unused.
 */
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

/*! A normal draw from a caller-owned stream: the Box-Muller transform (cosine
 * branch) on the stream's unit floats (u1, then u2; u1 is redrawn while
 * <= 1e-7). A NULL stream stops the program with a named error. */
float randomNormalCtx(rngStream_t *rs, float mean, float standardDeviation);

/*! Fills a FLOAT32 tensor with i.i.d. N(mean, stddev^2) draws from the given
 * stream, element-storage order. Fails fast on non-FLOAT32 tensors; an empty
 * tensor draws nothing and needs no stream. */
void fillNormalFloat32Tensor(tensor_t *out, rngStream_t *rs, float mean, float stddev);

/*! A uniform draw in [min, max) from a caller-owned stream: min + u * (max - min)
 * with u the stream's next unit float. A NULL stream stops the program with a
 * named error. */
float randomUniformCtx(rngStream_t *rs, float min, float max);

/*! Gets random value from Kaiming distribution using normal distribution.
 *
 * @param gain Optional scaling factor for non-linearity
 * @param fanMode: Number of input/output features
 * @param rs: Stream the draw comes from (randomNormalCtx; NULL stops the program)
 *
 * @returns Float value
 */
float kaimingNormal(float gain, size_t fanMode, rngStream_t *rs);

/*! Gets random value from Kaiming distribution using uniform distribution.
 *
 * @param gain Optional scaling factor for non-linearity
 * @param fanMode: Number of input/output features
 * @param rs: Stream the draw comes from (randomUniformCtx; NULL stops the program)
 *
 * @returns Float value
 */
float kaimingUniform(float gain, size_t fanMode, rngStream_t *rs);

/*! Gets random value from Xavier distribution using the normal distribution.
 *
 * @param gain Optional scaling factor for non-linearity
 * @param fanIn: Number of input features
 * @param fanOut: Number of output features
 * @param rs: Stream the draw comes from (randomNormalCtx; NULL stops the program)
 *
 * @returns Float value
 */
float xavierNormal(float gain, size_t fanIn, size_t fanOut, rngStream_t *rs);

/*! Gets random value from Xavier distribution using the uniform distribution.
 *
 * @param gain Optional scaling factor for non-linearity
 * @param fanIn: Number of input features
 * @param fanOut: Number of output features
 * @param rs: Stream the draw comes from (randomUniformCtx; NULL stops the program)
 *
 * @returns Float value
 */
float xavierUniform(float gain, size_t fanIn, size_t fanOut, rngStream_t *rs);

#endif // ODT_DISTRIBUTIONS_H
