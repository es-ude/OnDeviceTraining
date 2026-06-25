#ifndef LAYER_COMMON_H
#define LAYER_COMMON_H

#include <assert.h>
#include <stddef.h>

#include "Tensor.h"

/*! Bias presence tri-state for layer init structs.
 *  BIAS_DEFAULT lands at C99 zero-init; factories resolve it to the PyTorch
 *  default for that layer type (true for Conv / Linear).  */
typedef enum {
    BIAS_DEFAULT = 0,
    BIAS_TRUE = 1,
    BIAS_FALSE = 2,
} bias_t;

_Static_assert(BIAS_DEFAULT == 0,
               "BIAS_DEFAULT must be enum value 0 so .bias zero-init defaults to PyTorch default");

/*! Weight initialization scheme for layer init structs.
 *  INIT_DEFAULT lands at C99 zero-init; factories resolve it to PyTorch's
 *  default weight init for that layer type — kaiming_uniform_(a=sqrt(5)),
 *  i.e. uniform(+/- 1/sqrt(fan_in)). The bias is ALWAYS uniform(+/- 1/sqrt(fan_in))
 *  regardless of the weight scheme (PyTorch convention). */
typedef enum initScheme {
    INIT_DEFAULT = 0,     /*!< PyTorch parity: weight kaiming a=sqrt(5) (bound 1/sqrt(fan_in)) */
    INIT_KAIMING_UNIFORM, /*!< He; .gain (0 -> sqrt(2)) */
    INIT_XAVIER_UNIFORM,  /*!< Glorot; .gain (0 -> 1) */
} initScheme_t;

_Static_assert(INIT_DEFAULT == 0,
               "INIT_DEFAULT must be enum value 0 so .weightInit zero-init defaults to PyTorch");

/*! Weight init recipe carried on the layer init structs. Zero-init
 *  (scheme INIT_DEFAULT, gain 0) reproduces PyTorch's default. */
typedef struct weightInit {
    initScheme_t scheme;
    float gain; /*!< 0 selects the scheme's default gain. Ignored for INIT_DEFAULT. */
} weightInit_t;

/*! Initialize a FLOAT32 weight tensor in place according to `cfg`.
 *  Resolves scheme -> distribution and calls initDistribution.
 *
 *  - INIT_DEFAULT: kaimingUniform(gain = sqrt(1/3), fanIn) = uniform(+/- 1/sqrt(fanIn)),
 *                  matching PyTorch kaiming_uniform_(a=sqrt(5)). gain ignored.
 *  - INIT_KAIMING_UNIFORM: kaimingUniform(gain ? gain : sqrt(2), fanIn).
 *  - INIT_XAVIER_UNIFORM: xavierUniform(gain ? gain : 1, fanIn, fanOut).
 *
 *  Aborts (PRINT_ERROR + exit) if the tensor is not FLOAT32. */
void initWeightTensor(tensor_t *weight, weightInit_t cfg, size_t fanIn, size_t fanOut);

/*! Initialize a FLOAT32 bias tensor in place to PyTorch's default
 *  uniform(+/- 1/sqrt(fanIn)). Aborts if the tensor is not FLOAT32. */
void initBiasTensor(tensor_t *bias, size_t fanIn);

#endif /* LAYER_COMMON_H */
