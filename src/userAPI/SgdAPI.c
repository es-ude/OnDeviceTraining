#include <stdlib.h>

#include "Tensor.h"
#include "Layer.h"
#include "SgdAPI.h"

momentumBuffer_t **initMomentumBuffersFloat(layer_t *model, size_t sizeModel) {

    size_t sizeMomentumBuffers = calcTotalNumberOfMomentumBuffers(model, sizeModel);
    momentumBuffer_t **momentumBuffers = calloc(sizeMomentumBuffers, sizeof(momentumBuffer_t *));

    for (size_t i = 0; i < sizeModel; i++) {
        layer_t currentLayer = model[i];
        layerConfig_t *layerConfig = currentLayer.config;

        switch (currentLayer.type) {
        case LINEAR:
            linearConfig_t *linearConfig = layerConfig->linear;

            momentumBuffer_t *weightMomentumBuffer = calloc(1, sizeof(momentumBuffer_t));
            parameter_t *weights = linearConfig->weights;
            tensor_t *weightMomentums = calloc(1, sizeof(tensor_t));
            size_t sizeWeights = calcNumberOfElementsByTensor(linearConfig->weights->param);
            shape_t *weightsShape = weights->param->shape;
            uint8_t *weightsSparsityBitmask = weights->param->sparsityBitmask;
            uint8_t *weightMomentumData = calloc(sizeWeights, sizeof(float));
            quantization_t *weightMomentumsQ = calloc(1, sizeof(quantization_t));
            initFloat32Quantization(weightMomentumsQ);
            setTensorValues(weightMomentums, weightMomentumData, weightsShape, weightMomentumsQ,
                            weightsSparsityBitmask);
            initMomentumBuffer(weightMomentumBuffer, weights, weightMomentums);



            momentumBuffer_t *biasMomentumBuffer = calloc(1, sizeof(momentumBuffer_t));
            parameter_t *biass = linearConfig->bias;
            tensor_t *biasMomentums = calloc(1, sizeof(tensor_t));
            size_t sizebiass = calcNumberOfElementsByTensor(linearConfig->bias->param);
            shape_t *biassShape = biass->param->shape;
            uint8_t *biassSparsityBitmask = biass->param->sparsityBitmask;
            uint8_t *biasMomentumData = calloc(sizebiass, sizeof(float));
            quantization_t *biasMomentumsQ = calloc(1, sizeof(quantization_t));
            initFloat32Quantization(biasMomentumsQ);
            setTensorValues(biasMomentums, biasMomentumData, biassShape, biasMomentumsQ,
                            biassSparsityBitmask);
            initMomentumBuffer(biasMomentumBuffer, biass, biasMomentums);


            momentumBuffers[i] = weightMomentumBuffer;
            momentumBuffers[i+1] = biasMomentumBuffer;
            break;
        default:
            break;
        }
    }
    return momentumBuffers;
}

sgd_t *sgdInit(layer_t *model, size_t sizeModel, float learningRate, float momentumFactor,
               float weightDecay) {

    sgd_t *sgd = calloc(1, sizeof(sgd_t));

    sgd->learningRate = learningRate;
    sgd->weightDecay = weightDecay;
    sgd->momentumFactor = momentumFactor;
    sgd->sizeMomentumBuffers = calcTotalNumberOfMomentumBuffers(model, sizeModel);
    sgd->momentumBuffers = initMomentumBuffersFloat(model, sizeModel);

    return sgd;
}
