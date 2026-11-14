#ifndef LAYER_COMMON_H
#define LAYER_COMMON_H

#include <stdbool.h>
#include <stddef.h>

#include "Tensor.h"

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngKey rngKey_t;

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
    /*! The layer's init key (trainingRngInitKey(rng, l), l = the layer's position
     *  in the model): the weight draws from its (PARAM, 0) stream, the bias from
     *  (PARAM, 1). Required: a factory without one stops at its first draw. Read
     *  during the factory call only. */
    const rngKey_t *key;
} weightInit_t;

/*! Trainability tri-state for layer init structs.
 *  TRAINABLE_DEFAULT lands at C99 zero-init; factories resolve it to
 *  trainable (PyTorch requires_grad=True parity). TRAINABLE_FALSE freezes
 *  the layer at create time: no grad buffers are allocated, the optimizer
 *  skips its parameters, and backward skips its weight/bias grads. */
typedef enum {
    TRAINABLE_DEFAULT = 0,
    TRAINABLE_TRUE = 1,
    TRAINABLE_FALSE = 2,
} trainable_t;

_Static_assert(
    TRAINABLE_DEFAULT == 0,
    "TRAINABLE_DEFAULT must be enum value 0 so .trainable zero-init defaults to trainable");

/*! Resolve the tri-state; aborts on out-of-range values. */
bool resolveTrainable(trainable_t t, const char *factoryName);

/*! Validates a random-init recipe at a factory entry, before anything is allocated: a known
 *  scheme and FLOAT32 param storage (random init draws floats). Pass biasStorage NULL when
 *  the layer has no bias. SYM-stored params are reached by FLOAT32-init followed by
 *  requantizeTensorInPlace. initWeightTensor/initBiasTensor trust this check. */
void validateRandomInit(weightInit_t init, const quantization_t *weightStorage,
                        const quantization_t *biasStorage, const char *factoryName);

/*! Fills a FLOAT32 weight tensor validated by validateRandomInit, drawing from the
 *  (PARAM, 0) stream of cfg.key; stops at the first draw if cfg.key is NULL.
 *  - INIT_DEFAULT: kaimingUniform(sqrt(1/3), fanIn) = uniform(+/-1/sqrt(fanIn)); gain ignored.
 *  - INIT_KAIMING_UNIFORM: kaimingUniform(gain ? gain : sqrt(2), fanIn).
 *  - INIT_XAVIER_UNIFORM: xavierUniform(gain ? gain : 1, fanIn, fanOut). */
void initWeightTensor(tensor_t *weight, weightInit_t cfg, size_t fanIn, size_t fanOut);

/*! Fills a FLOAT32 bias tensor with PyTorch's default uniform(+/-1/sqrt(fanIn)) whatever the
 *  weight scheme, drawing from the (PARAM, 1) stream of initKey, a stream separate from the
 *  weight's; stops at the first draw if initKey is NULL. */
void initBiasTensor(tensor_t *bias, size_t fanIn, const rngKey_t *initKey);

#endif /* LAYER_COMMON_H */
