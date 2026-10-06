#ifndef TRAINING_CALL_H
#define TRAINING_CALL_H

/* Header-only and below both InferenceApi.h and TrainingLoopApi.h: the eval
 * and the training levels both take it, and InferenceApi cannot include
 * TrainingLoopApi.h without a library cycle. RematScheduler.h repeats the
 * forward typedef (C11 allows the redefinition), so a caller that passes NULL
 * needs no scheduler header. */
typedef struct rematScheduler rematScheduler_t;

/*! The trailing argument of the five training/eval levels. A NULL call means
 *  the same as a zero-initialised one (the trainingRunOptions_t idiom). */
typedef struct trainingCall {
    /* NULLable; NULL = an ephemeral scheduler per grads call (remat D30).
     * Otherwise borrowed: initialised and deinitialised by the caller, keyed
     * to the model and to the input shape the grads level receives. Read by
     * the grads and the eval level; the batch and epoch levels pass it down
     * (remat D19). */
    rematScheduler_t *remat;
} trainingCall_t;

#endif // TRAINING_CALL_H
