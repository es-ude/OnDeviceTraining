#define SOURCE_FILE "INFERENCE_Api"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Softmax.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"

// Initializes buffer to match output
static void initBufferOutput(tensor_t *buffer, layer_t *currentLayer, shape_t *inputShape,
                             sparsity_t *inputSparsity, quantization_t *inputQ) {
    layerType_t currentLayerType = currentLayer->type;
    quantization_t *currentQ = layerOutputQ(currentLayer);
    if (currentQ == NULL) {
        // Flatten has no per-layer quantization; output dtype equals input dtype.
        currentQ = inputQ;
    }

    size_t sizeDims = inputShape->numberOfDimensions;

    shape_t *outShape = reserveMemory(sizeof(shape_t));
    size_t *outDims = reserveMemory(sizeDims * sizeof(size_t));
    size_t *outOrder = reserveMemory(sizeDims * sizeof(size_t));

    outShape->dimensions = outDims;
    outShape->numberOfDimensions = sizeDims;
    outShape->orderOfDimensions = outOrder;

    calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayerType].calcOutputShape;
    calcOutputShape(currentLayer, inputShape, outShape);

    size_t numValues = calcNumberOfElementsByShape(outShape);
    size_t sizeData = calcNumberOfBytesForData(currentQ, numValues);
    uint8_t *data = reserveMemory(sizeData);

    quantization_t *q = reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        initFloat32Quantization(q);
        break;
    case SYM_INT32: {
        symInt32QConfig_t *currentQC = currentQ->qConfig;
        symInt32QConfig_t *symInt32QC = reserveMemory(sizeof(symInt32QConfig_t));

        initSymInt32QConfigWithQMaxBits(currentQC->roundingMode, symInt32QC, currentQC->qMaxBits);
        initSymInt32Quantization(symInt32QC, q);
        break;
    }
    case BFP: {
        /* BFP epic PR2 (plan Decision 5), the inference-path twin of
         * initLayerOutputs: widths/rounding/groupSize come from the layer's
         * declared template, numGroups is DERIVED from this buffer's own
         * element count. Exponents start at the zero state — the forward's
         * OUT_WRITE epilogue derives the grid. */
        bfpQConfig_t *currentBfpQC = currentQ->qConfig;
        bfpQConfig_t *bfpQC = reserveMemory(sizeof(bfpQConfig_t));
        if (currentBfpQC->groupSize == 0) {
            initBfpQConfig(currentBfpQC->mantissaBits, currentBfpQC->exponentBits,
                           currentBfpQC->roundingMode, bfpQC);
        } else {
            if (numValues % currentBfpQC->groupSize != 0) {
                PRINT_ERROR("initBufferOutput: BFP wire groupSize %zu does not divide the wire's "
                            "%zu elements -- pick a divisor or a per-tensor {1,0} template",
                            currentBfpQC->groupSize, numValues);
                exit(1);
            }
            initBfpQConfigGrouped(currentBfpQC->mantissaBits, currentBfpQC->exponentBits,
                                  currentBfpQC->roundingMode, numValues / currentBfpQC->groupSize,
                                  currentBfpQC->groupSize, bfpQC);
        }
        initBfpQuantization(bfpQC, q);
        break;
    }
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }

    setTensorValues(buffer, data, outShape, q, inputSparsity);
}

// Initializes buffer to match given input
static void initBufferInput(tensor_t *input, tensor_t *buffer) {
    quantization_t *currentQ = input->quantization;

    size_t sizeDims = input->shape->numberOfDimensions;

    shape_t *outShape = reserveMemory(sizeof(shape_t));
    size_t *outDims = reserveMemory(sizeDims * sizeof(size_t));
    size_t *outOrder = reserveMemory(sizeDims * sizeof(size_t));

    outShape->dimensions = outDims;
    outShape->numberOfDimensions = sizeDims;
    outShape->orderOfDimensions = outOrder;

    size_t numValues = calcNumberOfElementsByTensor(input);
    size_t sizeData = calcNumberOfBytesForData(currentQ, numValues);
    uint8_t *data = reserveMemory(sizeData);

    quantization_t *q = reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        q->type = FLOAT32;
        q->qConfig = NULL;
        break;
    case SYM_INT32:
        q->type = SYM_INT32;
        symInt32QConfig_t *currentQC = currentQ->qConfig;
        symInt32QConfig_t *symInt32QC = reserveMemory(sizeof(symInt32QConfig_t));
        symInt32QC->roundingMode = currentQC->roundingMode;
        symInt32QC->scale = currentQC->scale;
        symInt32QC->qMaxBits = currentQC->qMaxBits;
        q->qConfig = symInt32QC;
        break;
    case BFP: {
        /* BFP epic PR2: the entry buffer mirrors the INPUT's own config, so
         * validate the attach-time identity numGroups*groupSize == elements
         * first — a hand-built input tensor can bypass initTensor's gate, and a
         * malformed config would otherwise be re-derived into a DIFFERENT
         * geometry, which the copyTensor below rejects with a confusing
         * config-mismatch message. Geometry then comes from the same
         * derive-from-element-count rule as the other three allocators, so all
         * four agree by construction. The exponent VALUES are copied because
         * they ARE the input's grid (mirrors the SYM_INT32 scale copy above);
         * the trailing copyTensor would carry them too, but the buffer must be
         * a valid BFP tensor on its own the moment it exists. */
        bfpQConfig_t *currentBfpQC = currentQ->qConfig;
        validateBfpQConfigShape(currentBfpQC, numValues);
        bfpQConfig_t *bfpQC = reserveMemory(sizeof(bfpQConfig_t));
        if (currentBfpQC->groupSize == 0) {
            initBfpQConfig(currentBfpQC->mantissaBits, currentBfpQC->exponentBits,
                           currentBfpQC->roundingMode, bfpQC);
        } else {
            initBfpQConfigGrouped(currentBfpQC->mantissaBits, currentBfpQC->exponentBits,
                                  currentBfpQC->roundingMode, numValues / currentBfpQC->groupSize,
                                  currentBfpQC->groupSize, bfpQC);
        }
        memcpy(bfpQC->exponents, currentBfpQC->exponents, bfpQC->numGroups * sizeof(uint8_t));
        initBfpQuantization(bfpQC, q);
        break;
    }
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }

    setTensorValues(buffer, data, outShape, q, input->sparsity);

    copyTensor(buffer, input);
}

static void deInitBuffer(tensor_t *buffer) {
    freeData(buffer);
    freeShape(buffer->shape);
    freeQuantization(buffer->quantization);
}

tensor_t *inference(layer_t **model, size_t numberOfLayers, tensor_t *input) {
    tensor_t outputNext;

    initBufferInput(input, &outputNext);

    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        forwardFn_t forward = layerFunctions[currentLayerType].forward;

        tensor_t outputCurr;
        initBufferOutput(&outputCurr, currentLayer, outputNext.shape, outputNext.sparsity,
                         outputNext.quantization);
        forward(currentLayer, &outputNext, &outputCurr);

        deInitBuffer(&outputNext);
        outputNext = outputCurr;
    }

    tensor_t *output = getTensorLike(&outputNext);
    convertTensor(&outputNext, output);
    deInitBuffer(&outputNext);
    return output;
}

tensor_t **inferenceBatched(layer_t **model, size_t numberOfLayers, batch_t *batch) {
    tensor_t **tensorArr = reserveMemory(batch->size * sizeof(tensor_t));

    for (size_t i = 0; i < batch->size; i++) {
        tensorArr[i] = inference(model, numberOfLayers, batch->samples[i]->item);
    }

    return tensorArr;
}

/* Sized from the PRODUCED output, not the label: a classifier's label is
 * rank-1 [C] while the model emits [1, C] — sizing from the label made the
 * later shape copy overflow the dimensions array (ASan-verified). Mirrors
 * initTrainingStats (CalculateGradsSequential.c). */
static inferenceStats_t *reserveInferenceStats(tensor_t *producedOutput) {
    inferenceStats_t *inferenceStats = reserveMemory(sizeof(inferenceStats_t));
    inferenceStats->output = getTensorLike(producedOutput);
    return inferenceStats;
}

void freeInferenceStats(inferenceStats_t *inferenceStats) {
    freeTensor(inferenceStats->output);
    freeReservedMemory(inferenceStats);
}

inferenceStats_t *inferenceWithLoss(layer_t **model, size_t numberOfLayers, tensor_t *input,
                                    tensor_t *label, lossFuncType_t funcType,
                                    reduction_t forwardReduction) {
    tensor_t outputNext;
    initBufferInput(input, &outputNext);

    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        forwardFn_t forward = layerFunctions[currentLayerType].forward;

        tensor_t outputCurr;
        initBufferOutput(&outputCurr, currentLayer, outputNext.shape, outputNext.sparsity,
                         outputNext.quantization);
        forward(currentLayer, &outputNext, &outputCurr);
        deInitBuffer(&outputNext);
        outputNext = outputCurr;
    }

    inferenceStats_t *inferenceStats = reserveInferenceStats(&outputNext);
    convertTensor(&outputNext, inferenceStats->output);

    lossFunctions_t lossFns = lossFunctions[funcType];
    float loss = lossFns.forward(&outputNext, label, forwardReduction);
    inferenceStats->loss = loss;

    deInitBuffer(&outputNext);
    return inferenceStats;
}
