#define SOURCE_FILE "ODT_CONV1D"

#include "Conv1d.h"

#include "Add.h"
#include "Common.h"
#include "Conv1dKernel.h"
#include "ConvTranspose1dKernel.h"
#include "Layer.h"
#include "Mul.h"
#include "Quantization.h"
#include "Rounding.h"
#include "SlidingWindow1d.h"
#include "StorageApi.h"
#include "Tensor.h"

void initConv1dConfigWithWeightsAndBias(conv1dConfig_t *conv1dConfig, kernel_t *kernel,
                                        parameter_t *weights, parameter_t *bias, size_t groups,
                                        quantization_t *forwardQ, quantization_t *weightGradQ,
                                        quantization_t *biasGradQ, quantization_t *propLossQ) {
    if (groups == 0) {
        PRINT_ERROR("Conv1d: groups must be >= 1");
        exit(1);
    }
    if (kernel->size != weights->param->shape->dimensions[2]) {
        PRINT_ERROR("Conv1d: kernel->size (%zu) must equal weight kernelSize (%zu)", kernel->size,
                    weights->param->shape->dimensions[2]);
        exit(1);
    }
    conv1dConfig->kernel = kernel;
    conv1dConfig->weights = weights;
    conv1dConfig->bias = bias;
    conv1dConfig->groups = groups;
    conv1dConfig->forwardQ = forwardQ;
    conv1dConfig->weightGradQ = weightGradQ;
    conv1dConfig->biasGradQ = biasGradQ;
    conv1dConfig->propLossQ = propLossQ;
}

void conv1dForwardFloat(layer_t *conv1dLayer, tensor_t *input, tensor_t *output) {
    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    conv1dKernelFloat32(input, weightTensor, biasTensor, cfg->kernel, cfg->groups, output);
}

void conv1dForwardSymInt32(layer_t *conv1dLayer, tensor_t *input, tensor_t *output) {
    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    conv1dKernelSymInt32(input, weightTensor, biasTensor, cfg->kernel, cfg->groups, output);
}

void conv1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    conv1dConfig_t *cfg = layer->config->conv1d;
    switch (cfg->forwardQ->type) {
    case FLOAT32:
        conv1dForwardFloat(layer, input, output);
        break;
    case SYM_INT32:
        conv1dForwardSymInt32(layer, input, output);
        break;
    default:
        PRINT_ERROR("Conv1d forward: quantization type not implemented");
        exit(1);
    }
}

static void conv1dCalcWeightGradsFloat32(conv1dConfig_t *cfg, tensor_t *forwardInput,
                                         tensor_t *lossGrad) {
    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[0];

    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1d backward (weightGrad): lossGrad batch (%zu) does not match "
                    "forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1d backward (weightGrad): lossGrad outChannels (%zu) does not match "
                    "weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("Conv1d backward: lossGrad outputLength (%zu) does not match "
                    "geometry derived from forwardInput (%zu)",
                    outputLength, geom.outputLength);
        exit(1);
    }

    float const *xArr = (float const *)forwardInput->data;
    float const *gyArr = (float const *)lossGrad->data;
    float *gwArr = (float *)cfg->weights->grad->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    float gy = gyArr[(b * outChannels + oc) * outputLength + outPos];

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;

                            float xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            gwArr[(oc * inChPerGroup + icOffset) * kernelSize + kernelIdx] +=
                                xv * gy;
                        }
                    }
                }
            }
        }
    }
}

static void conv1dCalcBiasGradsFloat32(conv1dConfig_t *cfg, tensor_t *lossGrad) {
    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];

    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1d backward (biasGrad): lossGrad outChannels (%zu) does not match "
                    "bias Cout (%zu)",
                    outChannels, biasOutChannels);
        exit(1);
    }

    float const *gyArr = (float const *)lossGrad->data;
    float *gbArr = (float *)cfg->bias->grad->data;

    for (size_t oc = 0; oc < outChannels; oc++) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                sum += gyArr[(b * outChannels + oc) * outputLength + outPos];
            }
        }
        gbArr[oc] += sum;
    }
}

void conv1dCalcWeightGradsSymInt32(conv1dConfig_t *cfg, tensor_t *forwardInput,
                                   tensor_t *lossGrad) {
    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[0];

    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1d backward (weightGrad): lossGrad batch (%zu) does not match "
                    "forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1d backward (weightGrad): lossGrad outChannels (%zu) does not match "
                    "weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("Conv1d backward (SYM weightGrad): lossGrad outputLength (%zu) does not "
                    "match geometry derived from forwardInput (%zu)",
                    outputLength, geom.outputLength);
        exit(1);
    }

    float inScale = ((symInt32QConfig_t *)forwardInput->quantization->qConfig)->scale;
    float lossScale = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale;

    tensor_t *weightGrad = cfg->weights->grad;
    size_t numberOfWeights = calcNumberOfElementsByTensor(weightGrad);

    /* Fresh int32 intermediate at scale s_in*s_loss, allocated via reserveMemory
     * (allocation-locality rule — NOT a stack VLA). reserveMemory zero-inits. */
    int32_t *interData = reserveMemory(numberOfWeights * sizeof(int32_t));

    symInt32QConfig_t interQC;
    initSymInt32QConfig(((symInt32QConfig_t *)weightGrad->quantization->qConfig)->roundingMode,
                        &interQC);
    interQC.scale = inScale * lossScale;
    quantization_t interQ;
    initSymInt32Quantization(&interQC, &interQ);
    tensor_t intermediate;
    /* intermediate borrows weightGrad->shape read-only (addSymInt32TensorsInplace
     * only reads shapes); mirrors Linear.c backwardSymInt32. */
    setTensorValues(&intermediate, (uint8_t *)interData, weightGrad->shape, &interQ, NULL);

    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *gyArr = (int32_t const *)lossGrad->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    int32_t gy = gyArr[(b * outChannels + oc) * outputLength + outPos];

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;

                            int32_t xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            interData[(oc * inChPerGroup + icOffset) * kernelSize + kernelIdx] +=
                                mulInt32s(xv, gy);
                        }
                    }
                }
            }
        }
    }

    addSymInt32TensorsInplace(weightGrad, &intermediate);
    freeReservedMemory(interData);
}

void conv1dCalcBiasGradsSymInt32(conv1dConfig_t *cfg, tensor_t *lossGrad) {
    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];

    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1d backward (biasGrad): lossGrad outChannels (%zu) does not match "
                    "bias Cout (%zu)",
                    outChannels, biasOutChannels);
        exit(1);
    }

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    tensor_t *biasGrad = cfg->bias->grad;
    int32_t *gbArr = (int32_t *)biasGrad->data;

    float lossScale = ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale;
    symInt32QConfig_t *bgQC = (symInt32QConfig_t *)biasGrad->quantization->qConfig;
    float bgScale = bgQC->scale;

    for (size_t oc = 0; oc < outChannels; oc++) {
        /* int32 accumulator (NO int64): loss mantissas are int12-range per the
         * qMaxBits=12 operand contract, so the batch*outputLength sum stays
         * well within int32 (even more headroom than the old int16 path). */
        int32_t sum = 0;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                sum += gyArr[(b * outChannels + oc) * outputLength + outPos];
            }
        }
        gbArr[oc] += rescaleIntoAccumulatorScale(sum, lossScale, bgScale, bgQC->roundingMode);
    }
}

void conv1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                         tensor_t *propLoss) {
    conv1dConfig_t *cfg = layer->config->conv1d;

    conv1dCalcWeightGradsFloat32(cfg, forwardInput, lossGrad);

    if (cfg->bias) {
        conv1dCalcBiasGradsFloat32(cfg, lossGrad);
    }

    convTranspose1dKernelFloat32(lossGrad, cfg->weights->param, NULL, cfg->kernel, cfg->groups, 0u,
                                 propLoss);
}

void conv1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                    tensor_t *propLoss) {
    conv1dConfig_t *cfg = layer->config->conv1d;

    switch (cfg->weightGradQ->type) {
    case FLOAT32:
        conv1dCalcWeightGradsFloat32(cfg, forwardInput, lossGrad);
        break;
    case SYM_INT32:
        conv1dCalcWeightGradsSymInt32(cfg, forwardInput, lossGrad);
        break;
    default:
        PRINT_ERROR("Conv1d backward (weightGrad): quantization type not implemented");
        exit(1);
    }

    switch (cfg->biasGradQ->type) {
    case FLOAT32:
        if (cfg->bias) {
            conv1dCalcBiasGradsFloat32(cfg, lossGrad);
        }
        break;
    case SYM_INT32:
        if (cfg->bias) {
            conv1dCalcBiasGradsSymInt32(cfg, lossGrad);
        }
        break;
    default:
        PRINT_ERROR("Conv1d backward (biasGrad): quantization type not implemented");
        exit(1);
    }

    switch (cfg->propLossQ->type) {
    case FLOAT32:
        convTranspose1dKernelFloat32(lossGrad, cfg->weights->param, NULL, cfg->kernel, cfg->groups,
                                     0u, propLoss);
        break;
    case SYM_INT32:
        if (propLoss->quantization->type != SYM_INT32) {
            PRINT_ERROR("Conv1d backward: propLossQ is SYM_INT32 but the propLoss tensor is "
                        "not (#187)");
            exit(1);
        }
        convTranspose1dKernelSymInt32(lossGrad, cfg->weights->param, NULL, cfg->kernel, cfg->groups,
                                      0u, propLoss);
        break;
    default:
        PRINT_ERROR("Conv1d backward (propLoss): quantization type not implemented");
        exit(1);
    }
}

void conv1dCalcOutputShape(layer_t *conv1dLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;
    size_t batchSize = inputShape->dimensions[0];
    size_t inputLength = inputShape->dimensions[2];
    size_t outChannels = cfg->weights->param->shape->dimensions[0];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outChannels;
    outputShape->dimensions[2] = geom.outputLength;
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
