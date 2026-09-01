#define SOURCE_FILE "CALCULATE_GRADS_SEQUENTIAL"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "LossFunction.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Softmax.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingLoopApiInternal.h"

static void setDropoutLayersTraining(layer_t **model, size_t modelSize, bool training) {
    for (size_t i = 0; i < modelSize; i++) {
        if (model[i]->type == DROPOUT) {
            model[i]->config->dropout->training = training;
        }
    }
}

/* Defined below (needed here); forward-declared because layerParameters lives
 * further down in this file. */
static bool layerParameters(layer_t *layer, parameter_t **weightOut, parameter_t **biasOut);

/* Deepest (closest-to-input) layer whose parameters still train (#380 PR2).
 * Below it no dx is consumed, so backward truncates there; modelSize = none. */
static size_t deepestTrainableIndex(layer_t **model, size_t modelSize) {
    for (size_t i = 0; i < modelSize; i++) {
        parameter_t *w = NULL;
        parameter_t *b = NULL;
        if (layerParameters(model[i], &w, &b) && !layerIsFrozen(model[i])) {
            return i;
        }
    }
    return modelSize;
}

static trainingStats_t *calculateGradsImpl(layer_t **model, size_t modelSize,
                                           lossConfig_t lossConfig, reduction_t forwardReduction,
                                           tensor_t *input, tensor_t *label, traceSink_t sink,
                                           void *sinkCtx) {

    tensor_t *layerOutputs[modelSize + 1];
    layerOutputs[0] = input;
    setDropoutLayersTraining(model, modelSize, true);
    initLayerOutputs(layerOutputs, model, modelSize);

    // Forward pass
    for (size_t i = 0; i < modelSize; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        forwardFn_t forward = layerFunctions[currentLayerType].forward;
        forward(currentLayer, layerOutputs[i], layerOutputs[i + 1]);
        if (sink != NULL) {
            sink(sinkCtx, i, currentLayerType, "fwd", layerOutputs[i + 1]);
        }
    }

    trainingStats_t *trainingStats = initTrainingStats(layerOutputs[modelSize]);
    copyTensor(trainingStats->output, layerOutputs[modelSize]);

    // LOSS

    lossFunctions_t lossFns = lossFunctions[lossConfig.funcType];
    float loss = lossFns.forward(layerOutputs[modelSize], label, forwardReduction);
    trainingStats->loss = loss;

    // Backward pass
    size_t backwardIndex = modelSize - 1;
    if (lossConfig.funcType == CROSS_ENTROPY) {
        backwardIndex -= 1;
    }

    /* #380 PR2: backward truncates at the deepest trainable layer -- below it
     * no dx is consumed, so the loss-grad seed and the whole loop are skipped
     * entirely when no layer trains (deepest == modelSize sentinel). */
    size_t deepest = deepestTrainableIndex(model, modelSize);
    if (deepest < modelSize) {
        tensor_t gradNext;
        initGradTensor(&gradNext, layerOutputs[modelSize], NULL);
        lossFns.backward(layerOutputs[modelSize], label, &gradNext);
        if (sink != NULL) {
            sink(sinkCtx, modelSize, model[modelSize - 1]->type, "lossgrad", &gradNext);
        }

        for (int i = (int)backwardIndex; i >= (int)deepest; i--) {
            layerType_t layerType = model[i]->type;
            /* agrad@i = gradient w.r.t. layer i's OUTPUT (the wire grad entering layer i's
             * backward), matching the PyTorch forward-hook activation.grad. */
            if (sink != NULL) {
                sink(sinkCtx, (size_t)i, layerType, "agrad", &gradNext);
            }
            backwardFn_t backward = layerFunctions[layerType].backward;
            if ((size_t)i == deepest) {
                /* deepest trainable layer: grads only -- nothing below consumes dx */
                backward(model[i], layerOutputs[i], &gradNext, NULL);
            } else {
                tensor_t gradCurr;
                initGradTensor(&gradCurr, layerOutputs[i], backwardWireQ(model[i]));
                backward(model[i], layerOutputs[i], &gradNext, &gradCurr);
                deInitGradTensor(&gradNext);
                gradNext = gradCurr;
            }
        }
        deInitGradTensor(&gradNext);
    }

    deInitLayerOutputs(layerOutputs, modelSize);

    setDropoutLayersTraining(model, modelSize, false);
    return trainingStats;
}

trainingStats_t *calculateGradsSequential(layer_t **model, size_t modelSize,
                                          lossConfig_t lossConfig, reduction_t forwardReduction,
                                          tensor_t *input, tensor_t *label) {
    return calculateGradsImpl(model, modelSize, lossConfig, forwardReduction, input, label, NULL,
                              NULL);
}

trainingStats_t *tracedGrads(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                             reduction_t forwardReduction, tensor_t *input, tensor_t *label,
                             traceSink_t sink, void *ctx) {
    return calculateGradsImpl(model, modelSize, lossConfig, forwardReduction, input, label, sink,
                              ctx);
}

/* Return the two parameter_t* of a trainable layer (bias may be NULL).
 * Non-trainable layers return false. */
static bool layerParameters(layer_t *layer, parameter_t **weightOut, parameter_t **biasOut) {
    switch (layer->type) {
    case LINEAR:
        *weightOut = layer->config->linear->weights;
        *biasOut = layer->config->linear->bias;
        return true;
    case CONV1D:
        *weightOut = layer->config->conv1d->weights;
        *biasOut = layer->config->conv1d->bias; /* may be NULL */
        return true;
    case CONV1D_TRANSPOSED:
        *weightOut = layer->config->conv1dTransposed->weights;
        *biasOut = layer->config->conv1dTransposed->bias;
        return true;
    case LAYERNORM:
        *weightOut = layer->config->layerNorm->gamma;
        *biasOut = layer->config->layerNorm->beta;
        return true;
    case GROUPNORM:
        *weightOut = layer->config->groupNorm->gamma;
        *biasOut = layer->config->groupNorm->beta;
        return true;
    default:
        return false;
    }
}

static void traceModelParams(layer_t **model, size_t modelSize, const char *tag, bool wantGrad,
                             traceSink_t sink, void *ctx) {
    char phase[64];
    for (size_t i = 0; i < modelSize; i++) {
        parameter_t *w = NULL, *b = NULL;
        if (!layerParameters(model[i], &w, &b)) {
            continue;
        }
        /* Frozen layers (#380): getGradFromParameter returns NULL (Task 1 elides
         * the grad tensor). The TraceApi contract promises sinks a borrowed VALID
         * tensor -- skip the call entirely rather than handing them NULL. */
        tensor_t *wt = wantGrad ? getGradFromParameter(w) : getParamFromParameter(w);
        if (wt != NULL) {
            snprintf(phase, sizeof(phase), "%s.weight", tag);
            sink(ctx, i, model[i]->type, phase, wt);
        }
        if (b != NULL) {
            tensor_t *bt = wantGrad ? getGradFromParameter(b) : getParamFromParameter(b);
            if (bt != NULL) {
                snprintf(phase, sizeof(phase), "%s.bias", tag);
                sink(ctx, i, model[i]->type, phase, bt);
            }
        }
    }
}

void traceModelWeights(layer_t **model, size_t modelSize, const char *tag, traceSink_t sink,
                       void *ctx) {
    traceModelParams(model, modelSize, tag, /*wantGrad=*/false, sink, ctx);
}

void traceModelGrads(layer_t **model, size_t modelSize, const char *tag, traceSink_t sink,
                     void *ctx) {
    traceModelParams(model, modelSize, tag, /*wantGrad=*/true, sink, ctx);
}

static void initLayerOutputs(tensor_t **layerOutputs, layer_t **model, size_t sizeNetwork) {
    for (size_t i = 0; i < sizeNetwork; i++) {
        layer_t *currentLayer = model[i];
        quantization_t *currentQ = layerOutputQ(currentLayer);
        if (currentQ == NULL) {
            // Flatten has no per-layer quantization; output dtype equals input dtype.
            currentQ = layerOutputs[i]->quantization;
        }

        calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayer->type].calcOutputShape;
        size_t numberOfDims = layerOutputs[i]->shape->numberOfDimensions;
        if (currentLayer->type == FLATTEN) {
            numberOfDims = 2;
        }

        size_t *dims = reserveMemory(numberOfDims * sizeof(size_t));
        size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
        shape_t *outShape = reserveMemory(sizeof(shape_t));

        outShape->dimensions = dims;
        outShape->numberOfDimensions = numberOfDims;
        outShape->orderOfDimensions = order;

        calcOutputShape(currentLayer, layerOutputs[i]->shape, outShape);

        size_t numberOfValues = calcNumberOfElementsByShape(outShape);
        size_t sizeData = calcNumberOfBytesForData(currentQ, numberOfValues);
        uint8_t *data = reserveMemory(sizeData);

        quantization_t *q = reserveMemory(sizeof(quantization_t));
        switch (currentQ->type) {
        case FLOAT32:
            initFloat32Quantization(q);
            break;
        case SYM_INT32:
            q->type = SYM_INT32;
            symInt32QConfig_t *currentQC = currentQ->qConfig;
            symInt32QConfig_t *qC = reserveMemory(sizeof(symInt32QConfig_t));
            initSymInt32QConfigWithQMaxBits(currentQC->roundingMode, qC, currentQC->qMaxBits);
            initSymInt32Quantization(qC, q);
            break;
        case BFP: {
            /* BFP epic PR2 (plan Decision 5): the template supplies widths,
             * rounding and groupSize; numGroups is DERIVED from THIS wire's
             * element count. A layerQuant_t profile is shape-agnostic — the
             * same template is routinely shared across layers whose wires
             * differ in size — so a template numGroups can only ever be a
             * guess, and honoring it would size exponents[] against a buffer
             * it does not describe (the packer's group index is unbounded by
             * numGroups: a heap overflow, not a wrong number). Exponents start
             * at the zero state; the forward's OUT_WRITE epilogue derives the
             * real grid. */
            bfpQConfig_t *currentBfpQC = currentQ->qConfig;
            bfpQConfig_t *bfpQC = reserveMemory(sizeof(bfpQConfig_t));
            /* groupSize == wire elements derives numGroups == 1: one group
             * spanning the tensor IS per-tensor blocking, whose canonical
             * spelling is {1,0} — the derived {1,N} would violate the config
             * grammar even though the divisibility guard passed. */
            if (currentBfpQC->groupSize == 0 || currentBfpQC->groupSize == numberOfValues) {
                initBfpQConfig(currentBfpQC->mantissaBits, currentBfpQC->exponentBits,
                               currentBfpQC->roundingMode, bfpQC);
            } else {
                if (numberOfValues % currentBfpQC->groupSize != 0) {
                    PRINT_ERROR("initLayerOutputs: BFP wire groupSize %zu does not divide the "
                                "wire's %zu elements -- pick a divisor or a per-tensor {1,0} "
                                "template",
                                currentBfpQC->groupSize, numberOfValues);
                    exit(1);
                }
                initBfpQConfigGrouped(currentBfpQC->mantissaBits, currentBfpQC->exponentBits,
                                      currentBfpQC->roundingMode,
                                      numberOfValues / currentBfpQC->groupSize,
                                      currentBfpQC->groupSize, bfpQC);
            }
            initBfpQuantization(bfpQC, q);
            break;
        }
        default:
            PRINT_ERROR("Unknown QType!");
            exit(1);
        }

        tensor_t *tensor = reserveMemory(sizeof(tensor_t));
        tensor->data = data;
        tensor->quantization = q;
        tensor->shape = outShape;

        tensor->sparsity = NULL;
        if (layerOutputs[i]->sparsity != NULL) {
            sparsity_t *sparsity = reserveMemory(sizeof(sparsity_t));
            tensor->sparsity = sparsity;
        }

        layerOutputs[i + 1] = tensor;
    }
}

static void deInitLayerOutputs(tensor_t **layerOutputs, size_t modelSize) {
    for (size_t i = 1; i <= modelSize; i++) {
        freeTensor(layerOutputs[i]);
    }
}

static void initGradTensor(tensor_t *grad, tensor_t *layerOutput, quantization_t *wireQ) {
    shape_t *currentShape = layerOutput->shape;
    quantization_t *currentQ = (wireQ != NULL) ? wireQ : layerOutput->quantization;

    size_t *dims = reserveMemory(currentShape->numberOfDimensions * sizeof(size_t));
    size_t *order = reserveMemory(currentShape->numberOfDimensions * sizeof(size_t));
    shape_t *inShape = reserveMemory(sizeof(shape_t));

    inShape->dimensions = dims;
    inShape->numberOfDimensions = currentShape->numberOfDimensions;
    inShape->orderOfDimensions = order;

    memcpy(inShape->dimensions, currentShape->dimensions,
           currentShape->numberOfDimensions * sizeof(size_t));
    memcpy(inShape->orderOfDimensions, currentShape->orderOfDimensions,
           currentShape->numberOfDimensions * sizeof(size_t));

    setOrderOfDimsForNewTensor(inShape->numberOfDimensions, inShape->orderOfDimensions);

    size_t numberOfValues = calcNumberOfElementsByShape(currentShape);
    size_t sizeData = calcNumberOfBytesForData(currentQ, numberOfValues);
    uint8_t *data = reserveMemory(sizeData);

    quantization_t *q = reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        initFloat32Quantization(q);
        break;
    case SYM_INT32: {
        symInt32QConfig_t *currentQC = currentQ->qConfig;
        symInt32QConfig_t *qC = reserveMemory(sizeof(symInt32QConfig_t));
        initSymInt32QConfigWithQMaxBits(currentQC->roundingMode, qC, currentQC->qMaxBits);
        initSymInt32Quantization(qC, q);
        break;
    }
    case BFP: {
        /* Same derive-from-element-count rule as initLayerOutputs (Decision 5).
         * `currentQ` is the layer's propLossQ template — except on the loss-grad
         * seed path, where wireQ is NULL and the template is the model OUTPUT's
         * own (already wire-sized) config; the rule is identical either way,
         * since it only ever reads groupSize/widths and re-derives numGroups. */
        bfpQConfig_t *currentBfpQC = currentQ->qConfig;
        bfpQConfig_t *bfpQC = reserveMemory(sizeof(bfpQConfig_t));
        /* groupSize == wire elements -> per-tensor {1,0}, see initLayerOutputs. */
        if (currentBfpQC->groupSize == 0 || currentBfpQC->groupSize == numberOfValues) {
            initBfpQConfig(currentBfpQC->mantissaBits, currentBfpQC->exponentBits,
                           currentBfpQC->roundingMode, bfpQC);
        } else {
            if (numberOfValues % currentBfpQC->groupSize != 0) {
                PRINT_ERROR("initGradTensor: BFP dx-wire groupSize %zu does not divide the wire's "
                            "%zu elements -- pick a divisor or a per-tensor {1,0} template",
                            currentBfpQC->groupSize, numberOfValues);
                exit(1);
            }
            initBfpQConfigGrouped(
                currentBfpQC->mantissaBits, currentBfpQC->exponentBits, currentBfpQC->roundingMode,
                numberOfValues / currentBfpQC->groupSize, currentBfpQC->groupSize, bfpQC);
        }
        initBfpQuantization(bfpQC, q);
        break;
    }
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }

    grad->data = data;
    grad->quantization = q;
    grad->shape = inShape;

    grad->sparsity = NULL;
    if (layerOutput->sparsity != NULL) {
        sparsity_t *sparsity = reserveMemory(sizeof(sparsity_t));
        grad->sparsity = sparsity;
    }
}

static void deInitGradTensor(tensor_t *tensor) {
    freeData(tensor);
    freeShape(tensor->shape);
    freeQuantization(tensor->quantization);
}

static trainingStats_t *initTrainingStats(tensor_t *output) {
    trainingStats_t *trainingStats = reserveMemory(sizeof(trainingStats_t));

    tensor_t *o = getTensorLike(output);
    trainingStats->output = o;

    return trainingStats;
}
