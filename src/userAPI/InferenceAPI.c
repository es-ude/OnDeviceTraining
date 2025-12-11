#include <stdio.h>

#include "Tensor.h"
#include "Layer.h"
#include "InferenceAPI.h"
#include "TensorAPI.h"

tensor_t *inference(layer_t **model, size_t numberOfLayers, tensor_t *input) {
    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        quantization_t *currentQ = currentLayer->outputQ;

        size_t outputNumberOfDims = input->shape->numberOfDimensions;
        size_t sizeDims = outputNumberOfDims;
        size_t outDims[sizeDims];
        size_t outOrder[sizeDims];

        shape_t outShape;
        outShape.dimensions = outDims;
        outShape.numberOfDimensions = outputNumberOfDims;
        outShape.orderOfDimensions = outOrder;

        calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayerType].calcOutputShape;
        calcOutputShape(currentLayer, input->shape, &outShape);

        size_t numValues =
            calcNumberOfElementsByShape(&outShape);

        size_t sizeData = calcBytesOutputData(currentLayer->outputQ, numValues);
        uint8_t data[sizeData];

        void *maybeSparsityBitmask = NULL;
        uint8_t sparsityBitmask[sizeData];
        if(input->sparsityBitmask != NULL) {
            maybeSparsityBitmask = sparsityBitmask;
        }

        quantization_t q;
        asymQConfig_t asymQConfig;
        switch (currentQ->type) {
        case FLOAT32:
            q.type = FLOAT32;
            q.qConfig = NULL;
            break;
        case ASYM:
            q.type = ASYM;
            asymQConfig_t *currentQC = currentQ->qConfig;
            asymQConfig.scale = currentQC->scale;
            asymQConfig.qBits = currentQC->qBits;
            asymQConfig.roundingMode = currentQC->roundingMode;
            asymQConfig.zeroPoint = currentQC->zeroPoint;
            q.qConfig = &asymQConfig;
            break;
        default:
            break;
        }

        tensor_t intermediateOutput;
        setTensorValues(&intermediateOutput, data, &outShape, &q, maybeSparsityBitmask);

        forwardFn_t forward = layerFunctions[currentLayerType].forward;
        forward(currentLayer, input, &intermediateOutput);

        if (i == numberOfLayers - 1) {
            tensor_t *output = getTensorLike(&intermediateOutput);
            copyTensor(output, &intermediateOutput);
            printTensor(output);

            return output;
        }

        copyTensor(input, &intermediateOutput);
    }
}
