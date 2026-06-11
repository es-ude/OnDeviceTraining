#define SOURCE_FILE "CROSS_ENTROPY"

#include <stdlib.h>

#include "Common.h"
#include "CrossEntropy.h"
#include "Log.h"
#include "TensorConversion.h"

#include <math.h>

float crossEntropyForwardFloat(tensor_t *softmaxOutput, tensor_t *distribution,
                               reduction_t reduction) {
    size_t n = calcNumberOfElementsByTensor(softmaxOutput);
    float *p = (float *)softmaxOutput->data;
    float *y = (float *)distribution->data;

    float loss = 0.f;
    for (size_t i = 0; i < n; i++) {
        float pi = p[i];

        if (!isfinite(pi)) {
            printf("NaN/Inf softmax at %zu\n", i);
            printf("%f\n", pi);
            abort();
        }

        pi = fmaxf(pi, 1e-7f);
        loss += y[i] * -logf(pi);
    }

    if (reduction == REDUCTION_MEAN && softmaxOutput->shape->numberOfDimensions >= 2) {
        size_t microbatch = softmaxOutput->shape->dimensions[0];
        return loss / (float)microbatch;
    }
    return loss;
}

float crossEntropyForward(tensor_t *softmaxOutput, tensor_t *distribution, reduction_t reduction) {
    switch (softmaxOutput->quantization->type) {
    case FLOAT32:
        return crossEntropyForwardFloat(softmaxOutput, distribution, reduction);
    default:
        PRINT_ERROR("CrossEntropy forward only implemented for FLOAT32!");
        exit(1);
    }
}

/*float crossEntropyForwardFloat(tensor_t *softmaxOutput, tensor_t *distribution) {
    size_t numberOfValues = calcNumberOfElementsByTensor(softmaxOutput);

    float *softmaxOutputFloat = (float *)softmaxOutput->data;
    float *distributionFloat = (float *)distribution->data;


    float loss = 0.f;
    for (size_t i = 0; i < numberOfValues; i++) {
        if(softmaxOutputFloat[i] == 0) {
            // Question aks Leo if == 1 or small value
            softmaxOutputFloat[i] = 0.00000000001f;
        }

        printf("%f\n", softmaxOutputFloat[i]);

        loss += distributionFloat[i] * -logFloat(softmaxOutputFloat[i]);
    }

    return loss;
}*/

static void crossEntropySoftmaxBackwardFloat(tensor_t *softmaxOutput, tensor_t *distribution,
                                             tensor_t *loss) {
    size_t totalInputSize = calcNumberOfElementsByTensor(softmaxOutput);

    float *softmaxOutputFloat = (float *)softmaxOutput->data;
    float *distributionFloat = (float *)distribution->data;
    float *lossFloat = (float *)loss->data;

    for (size_t i = 0; i < totalInputSize; i++) {
        lossFloat[i] = softmaxOutputFloat[i] - distributionFloat[i];
    }
}

static void crossEntropySoftmaxBackwardAsym(tensor_t *softmaxOutput, tensor_t *distribution,
                                            tensor_t *loss) {
    size_t inputSize = calcNumberOfElementsByTensor(softmaxOutput);

    tensor_t softmaxOutputFloat;
    quantization_t softmaxOutputFloatQ;
    initFloat32Quantization(&softmaxOutputFloatQ);
    uint8_t softmaxOutputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(softmaxOutputFloatData, &softmaxOutputFloatQ, softmaxOutput,
                                 &softmaxOutputFloat);
    convertTensor(softmaxOutput, &softmaxOutputFloat);

    tensor_t distributionFloat;
    quantization_t distributionFloatQ;
    initFloat32Quantization(&distributionFloatQ);
    uint8_t distributionFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(distributionFloatData, &distributionFloatQ, distribution,
                                 &distributionFloat);
    convertTensor(distribution, &distributionFloat);

    tensor_t lossFloat;
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);
    uint8_t lossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(lossFloatData, &lossFloatQ, loss, &lossFloat);
    convertTensor(loss, &lossFloat);

    float *softmaxOutputFloatArr = (float *)softmaxOutputFloat.data;
    float *distributionFloatArr = (float *)distributionFloat.data;
    float *lossFloatArr = (float *)lossFloat.data;

    for (size_t i = 0; i < inputSize; i++) {
        lossFloatArr[i] = softmaxOutputFloatArr[i] - distributionFloatArr[i];
    }

    convertTensor(&lossFloat, loss);
}

// IMPORTANT: This implementation already takes the softmax backward into account
void crossEntropySoftmaxBackward(tensor_t *softmaxOutput, tensor_t *distribution, tensor_t *loss) {
    switch (softmaxOutput->quantization->type) {
    case FLOAT32:
        crossEntropySoftmaxBackwardFloat(softmaxOutput, distribution, loss);
        break;
    case ASYM:
        crossEntropySoftmaxBackwardAsym(softmaxOutput, distribution, loss);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

float computeMeanScaleCE(size_t totalSamples, tensor_t *modelOutput) {
    (void)modelOutput;
    return 1.0f / (float)totalSamples;
}
