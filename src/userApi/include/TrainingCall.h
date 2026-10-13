#ifndef TRAINING_CALL_H
#define TRAINING_CALL_H

/* Header-only and below both InferenceApi.h and TrainingLoopApi.h: the eval
 * and the training levels both take it, and InferenceApi cannot include
 * TrainingLoopApi.h without a library cycle. RematScheduler.h repeats the
 * forward typedef (C11 allows the redefinition), so a caller that passes NULL
 * needs no scheduler header. */
typedef struct rematScheduler rematScheduler_t;
/* Pointer-only use: the identical typedefs live in RNG.h and TrainingLoopApi.h
 * (C11 allows the repetition), so this header needs neither include path. */
typedef struct rngKey rngKey_t;
typedef struct trainingRng trainingRng_t;

/*! The trailing argument of the five training/eval levels. A NULL call means
 *  the same as a zero-initialised one (the trainingRunOptions_t idiom). Each
 *  level hands its callee a copy of the call it received with only .key
 *  replaced by its own (trainingCallForIndex; the epoch level puts the step
 *  key it derives from .rng), so a field reaches every level and a caller's
 *  key never reaches a level below its own. */
typedef struct trainingCall {
    /* NULLable; NULL = an ephemeral scheduler per grads call on the
     * training call's default plan (remat D15, D30).
     * Otherwise borrowed: initialised and deinitialised by the caller, keyed
     * to the model and to the input shape the grads level receives. Read by
     * the grads and the eval level; the batch and epoch levels pass it down
     * (remat D19). */
    rematScheduler_t *remat;
    /* NULLable, borrowed for the call: the key of this level. The grads level
     * (calculateGradsFn_t) and the eval level (inferenceWithLossFn_t) read it
     * as the call key, from which each layer's key is folded; the batch level
     * reads it as the step key; the epoch level ignores it. NULL = no keys
     * below this level. */
    const rngKey_t *key;
    /* NULLable, borrowed: read only by the epoch level, which derives the
     * step key of each update from it and counts the updates; every other
     * level ignores it. */
    trainingRng_t *rng;
} trainingCall_t;

#endif // TRAINING_CALL_H
