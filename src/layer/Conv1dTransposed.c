#define SOURCE_FILE "ODT_CONV1D_TRANSPOSED"

#include "Conv1dTransposed.h"

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

void initConv1dTransposedConfigWithWeightsAndBias(
    conv1dTransposedConfig_t *cfg, kernel_t *kernel, parameter_t *weights, parameter_t *bias,
    size_t groups, size_t outputPadding, quantization_t *forwardQ, quantization_t *weightGradQ,
    quantization_t *biasGradQ, quantization_t *propLossQ) {
    if (groups == 0) {
        PRINT_ERROR("Conv1dTransposed: groups must be >= 1");
        exit(1);
    }
    if (kernel->paddingType != VALID) {
        PRINT_ERROR("Conv1dTransposed: only VALID paddingType supported in Phase 1");
        exit(1);
    }
    if (outputPadding != 0 &&
        outputPadding >=
            ((kernel->stride > kernel->dilation) ? kernel->stride : kernel->dilation)) {
        PRINT_ERROR("Conv1dTransposed: outputPadding (%zu) must be < max(stride=%zu, "
                    "dilation=%zu)",
                    outputPadding, kernel->stride, kernel->dilation);
        exit(1);
    }
    if (kernel->size != weights->param->shape->dimensions[2]) {
        PRINT_ERROR("Conv1dTransposed: kernel->size (%zu) must equal weight kernelSize (%zu)",
                    kernel->size, weights->param->shape->dimensions[2]);
        exit(1);
    }
    cfg->kernel = kernel;
    cfg->weights = weights;
    cfg->bias = bias;
    cfg->groups = groups;
    cfg->outputPadding = outputPadding;
    cfg->forwardQ = forwardQ;
    cfg->weightGradQ = weightGradQ;
    cfg->biasGradQ = biasGradQ;
    cfg->propLossQ = propLossQ;
}

void conv1dTransposedForwardFloat(layer_t *layer, tensor_t *input, tensor_t *output) {
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    convTranspose1dKernelFloat32(input, weightTensor, biasTensor, cfg->kernel, cfg->groups,
                                 cfg->outputPadding, output);
}

void conv1dTransposedForwardSymInt32(layer_t *layer, tensor_t *input, tensor_t *output) {
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    convTranspose1dKernelSymInt32(input, weightTensor, biasTensor, cfg->kernel, cfg->groups,
                                  cfg->outputPadding, output);
}

void conv1dTransposedForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    switch (cfg->forwardQ->type) {
    case FLOAT32:
        conv1dTransposedForwardFloat(layer, input, output);
        break;
    case SYM_INT32:
        conv1dTransposedForwardSymInt32(layer, input, output);
        break;
    default:
        PRINT_ERROR("Conv1dTransposed forward: quantization type not implemented");
        exit(1);
    }
}

static void conv1dTransposedCalcWeightGradsFloat32(conv1dTransposedConfig_t *cfg,
                                                   tensor_t *forwardInput, tensor_t *lossGrad) {
    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];

    size_t expectedOutLen =
        convTranspose1dOutputLength(inputLength, cfg->kernel, cfg->outputPadding);
    if (expectedOutLen != outputLength) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outputLength (%zu) does "
                    "not match the transpose geometry from forwardInput (expected %zu)",
                    outputLength, expectedOutLen);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    // Conv1dTransposed weight shape [Cin, Cout/groups, K] -> Cout = dim[1] * groups.
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[1] * groups;
    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad batch (%zu) does not "
                    "match forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outChannels (%zu) does "
                    "not match weight Cout (%zu)",
                    outChannels, weightOutChannels);
        exit(1);
    }

    long long outputLengthSigned = (long long)outputLength;
    long long dilation = (long long)cfg->kernel->dilation;
    size_t stride = cfg->kernel->stride;

    float const *xArr = (float const *)forwardInput->data;
    float const *gyArr = (float const *)lossGrad->data;
    float *gwArr = (float *)cfg->weights->grad->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    float xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    long long outBase = (long long)(inPos * stride); // VALID -> padLeft=0

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        for (size_t k = 0; k < kernelSize; k++) {
                            long long outIdx = outBase + (long long)k * dilation;
                            if (outIdx < 0 || outIdx >= outputLengthSigned) {
                                continue;
                            }
                            float gy =
                                gyArr[(b * outChannels + oc) * outputLength + (size_t)outIdx];
                            gwArr[(ic * outChPerGroup + ocOffset) * kernelSize + k] += xv * gy;
                        }
                    }
                }
            }
        }
    }
}

static void conv1dTransposedCalcBiasGradsFloat32(conv1dTransposedConfig_t *cfg,
                                                 tensor_t *lossGrad) {
    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];

    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];
    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (biasGrad): lossGrad outChannels (%zu) does not "
                    "match bias Cout (%zu)",
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

void conv1dTransposedCalcWeightGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *forwardInput,
                                             tensor_t *lossGrad) {
    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];

    size_t expectedOutLen =
        convTranspose1dOutputLength(inputLength, cfg->kernel, cfg->outputPadding);
    if (expectedOutLen != outputLength) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outputLength (%zu) does "
                    "not match the transpose geometry from forwardInput (expected %zu)",
                    outputLength, expectedOutLen);
        exit(1);
    }

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    // Conv1dTransposed weight shape [Cin, Cout/groups, K] -> Cout = dim[1] * groups.
    size_t weightOutChannels = cfg->weights->param->shape->dimensions[1] * groups;
    if (batch != lossGrad->shape->dimensions[0]) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad batch (%zu) does not "
                    "match forwardInput batch (%zu)",
                    lossGrad->shape->dimensions[0], batch);
        exit(1);
    }
    if (outChannels != weightOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): lossGrad outChannels (%zu) does "
                    "not match weight Cout (%zu)",
                    outChannels, weightOutChannels);
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
    setTensorValues(&intermediate, (uint8_t *)interData, weightGrad->shape, &interQ, NULL);

    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *gyArr = (int32_t const *)lossGrad->data;

    long long outputLengthSigned = (long long)outputLength;
    long long dilation = (long long)cfg->kernel->dilation;
    size_t stride = cfg->kernel->stride;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    int32_t xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    long long outBase = (long long)(inPos * stride); // VALID -> padLeft=0

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        for (size_t k = 0; k < kernelSize; k++) {
                            long long outIdx = outBase + (long long)k * dilation;
                            if (outIdx < 0 || outIdx >= outputLengthSigned) {
                                continue;
                            }
                            int32_t gy =
                                gyArr[(b * outChannels + oc) * outputLength + (size_t)outIdx];
                            interData[(ic * outChPerGroup + ocOffset) * kernelSize + k] +=
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

void conv1dTransposedCalcBiasGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad) {
    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];

    size_t biasOutChannels = cfg->bias->param->shape->dimensions[0];
    if (outChannels != biasOutChannels) {
        PRINT_ERROR("Conv1dTransposed backward (biasGrad): lossGrad outChannels (%zu) does not "
                    "match bias Cout (%zu)",
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
         * well within int32. */
        int32_t sum = 0;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                sum += gyArr[(b * outChannels + oc) * outputLength + outPos];
            }
        }
        gbArr[oc] += rescaleIntoAccumulatorScale(sum, lossScale, bgScale, bgQC->roundingMode);
    }
}

void conv1dTransposedBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                                   tensor_t *propLoss) {
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;

    conv1dTransposedCalcWeightGradsFloat32(cfg, forwardInput, lossGrad);

    if (cfg->bias) {
        conv1dTransposedCalcBiasGradsFloat32(cfg, lossGrad);
    }

    // dL/dx via the adjoint: conv1d-correlation of lossGrad with weight.
    // The kernel here uses VALID (Phase-1 contract). conv1dKernelFloat32
    // accepts the same weight tensor (no flip needed, per spec §5.2).
    conv1dKernelFloat32(lossGrad, cfg->weights->param, NULL, cfg->kernel, cfg->groups, propLoss);
}

void conv1dTransposedBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                              tensor_t *propLoss) {
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;

    switch (cfg->weightGradQ->type) {
    case FLOAT32:
        conv1dTransposedCalcWeightGradsFloat32(cfg, forwardInput, lossGrad);
        break;
    case SYM_INT32:
        conv1dTransposedCalcWeightGradsSymInt32(cfg, forwardInput, lossGrad);
        break;
    default:
        PRINT_ERROR("Conv1dTransposed backward (weightGrad): quantization type not implemented");
        exit(1);
    }

    switch (cfg->biasGradQ->type) {
    case FLOAT32:
        if (cfg->bias) {
            conv1dTransposedCalcBiasGradsFloat32(cfg, lossGrad);
        }
        break;
    case SYM_INT32:
        if (cfg->bias) {
            conv1dTransposedCalcBiasGradsSymInt32(cfg, lossGrad);
        }
        break;
    default:
        PRINT_ERROR("Conv1dTransposed backward (biasGrad): quantization type not implemented");
        exit(1);
    }

    switch (cfg->propLossQ->type) {
    case FLOAT32:
        // dL/dx via the adjoint: conv1d-correlation of lossGrad with weight (VALID, Phase-1).
        conv1dKernelFloat32(lossGrad, cfg->weights->param, NULL, cfg->kernel, cfg->groups,
                            propLoss);
        break;
    case SYM_INT32:
        if (propLoss->quantization->type != SYM_INT32) {
            PRINT_ERROR("Conv1dTransposed backward: propLossQ is SYM_INT32 but the propLoss "
                        "tensor is not (#187)");
            exit(1);
        }
        conv1dKernelSymInt32(lossGrad, cfg->weights->param, NULL, cfg->kernel, cfg->groups,
                             propLoss);
        break;
    default:
        PRINT_ERROR("Conv1dTransposed backward (propLoss): quantization type not implemented");
        exit(1);
    }
}

void conv1dTransposedCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1dTransposed expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    size_t batchSize = inputShape->dimensions[0];
    size_t inputLength = inputShape->dimensions[2];
    // Conv1dTransposed weight shape: [Cin, Cout/groups, K]
    // Cout = (Cout/groups) * groups
    size_t outChannelsPerGroup = cfg->weights->param->shape->dimensions[1];
    size_t outChannels = outChannelsPerGroup * cfg->groups;

    size_t outputLength = convTranspose1dOutputLength(inputLength, cfg->kernel, cfg->outputPadding);

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outChannels;
    outputShape->dimensions[2] = outputLength;
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
