#ifndef INFERENCE_H
#define INFERENCE_H

#include "DataLoader.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Tensor.h"
#include "TrainingCall.h"

typedef struct inferenceStats {
    tensor_t *output;
    float loss;
} inferenceStats_t;

void freeInferenceStats(inferenceStats_t *inferenceStats);

tensor_t *inference(layer_t **model, size_t numberOfLayers, tensor_t *input);

/*! Wraps each sample with batchViewOf before calling inference() -- see
 *  docs/conventions/data-shape.md, "Who adds the batch axis". */
tensor_t **inferenceBatched(layer_t **model, size_t numberOfLayers, batch_t *batch);

/*! call is NULLable: a NULL call or a NULL call->remat runs the forward on
 *  per-call buffers, as before #4. A non-NULL call->remat exits until #4 PR3
 *  runs evaluation on a caller's scheduler (remat D19). */
inferenceStats_t *inferenceWithLoss(layer_t **model, size_t numberOfLayers, tensor_t *input,
                                    tensor_t *label, lossFuncType_t funcType,
                                    reduction_t forwardReduction, const trainingCall_t *call);

#endif // INFERENCE_H
