#define SOURCE_FILE "LINEAR"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "Add.h"
#include "Common.h"
#include "Layer.h"
#include "Linear.h"
#include "Matmul.h"
#include "Rounding.h"
#include "TensorConversion.h"

void linearInitConfig(linearConfig_t *linearConfig, parameter_t *weights, parameter_t *bias,
                      quantization_t *forwardQ, quantization_t *weightGradQ,
                      quantization_t *biasGradQ, quantization_t *propLossQ) {
    linearConfig->weights = weights;
    linearConfig->bias = bias;
    linearConfig->forwardQ = forwardQ;
    linearConfig->weightGradQ = weightGradQ;
    linearConfig->biasGradQ = biasGradQ;
    linearConfig->propLossQ = propLossQ;
}

void linearForwardFloat(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulFloat32TensorsWithBias(input, w, output, b);
    transposeTensor(w, 0, 1);
}

void linearForwardSymInt32(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulSymInt32TensorsWithBias(input, w, output, b);
    transposeTensor(w, 0, 1);
}

void linearForward(layer_t *linearLayer, tensor_t *input, tensor_t *output) {
    linearConfig_t *linearConfig = linearLayer->config->linear;

    tensor_t *weights = getParamFromParameter(linearConfig->weights);
    tensor_t *bias = getParamFromParameter(linearConfig->bias);

    switch (linearConfig->forwardQ->type) {
    case FLOAT32:
        linearForwardFloat(weights, bias, input, output);
        break;
    case SYM_INT32:
        linearForwardSymInt32(weights, bias, input, output);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

void linearCalcWeightGradsFloat32(tensor_t *forwardInput, tensor_t *loss, tensor_t *weightGrads) {
    size_t numberOfWeights = calcNumberOfElementsByTensor(weightGrads);

    tensor_t intermediateWGrad;
    float intermediateWGradData[numberOfWeights];
    quantization_t intermediateWGradQ;
    initFloat32Quantization(&intermediateWGradQ);
    setTensorValues(&intermediateWGrad, (uint8_t *)intermediateWGradData, weightGrads->shape,
                    &intermediateWGradQ, weightGrads->sparsity);

    transposeTensor(loss, 0, 1);
    matmulFloat32Tensors(loss, forwardInput, &intermediateWGrad);
    transposeTensor(loss, 0, 1);

    addFloat32TensorsInplace(weightGrads, &intermediateWGrad);
}

void linearCalcWeightGradsFloatWithConversion(linearConfig_t *linearConfig, tensor_t *forwardInput,
                                              tensor_t *loss) {
    tensor_t *paramWG = getGradFromParameter(linearConfig->weights);
    tensor_t *wG = paramWG;
    tensor_t *fwdIn = forwardInput;
    tensor_t *l = loss;

    tensor_t forwardInputFloat;
    size_t sizeForwardInputFloat = calcNumberOfElementsByTensor(forwardInput);
    float forwardInputFloatData[sizeForwardInputFloat];
    quantization_t forwardInputFloatQ;
    initFloat32Quantization(&forwardInputFloatQ);

    if (fwdIn->quantization->type != FLOAT32) {
        setTensorValuesForConversion((uint8_t *)forwardInputFloatData, &forwardInputFloatQ, fwdIn,
                                     &forwardInputFloat);
        convertTensor(forwardInput, &forwardInputFloat);
        fwdIn = &forwardInputFloat;
    }

    tensor_t lossFloat;
    size_t sizeLoss = calcNumberOfElementsByTensor(loss);
    float lossFloatData[sizeLoss];
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);

    if (l->quantization->type != FLOAT32) {
        setTensorValuesForConversion((uint8_t *)lossFloatData, &lossFloatQ, l, &lossFloat);
        convertTensor(loss, &lossFloat);
        l = &lossFloat;
    }

    tensor_t weightGradFloat;
    size_t sizeWeightGrad = calcNumberOfElementsByTensor(wG);
    float weightGradFloatData[sizeWeightGrad];
    quantization_t weightGradFloatQ;
    initFloat32Quantization(&weightGradFloatQ);

    if (wG->quantization->type != FLOAT32) {
        setTensorValuesForConversion((uint8_t *)weightGradFloatData, &weightGradFloatQ, wG,
                                     &weightGradFloat);
        convertTensor(wG, &weightGradFloat);
        wG = &weightGradFloat;
    }

    linearCalcWeightGradsFloat32(fwdIn, l, wG);
    convertTensor(wG, paramWG);
}

void linearCalcBiasGradsFloat32(tensor_t *loss, tensor_t *biasGrad) {
    /* Reduce the loss over the batch (leading) axis into the rank-1 bias grad:
     * biasGrad[f] += sum_n loss[n,f]. Mirrors conv1dCalcBiasGradsFloat32. */
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrad);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    size_t batch = (numFeatures == 0) ? 0 : numLoss / numFeatures;
    float *bg = (float *)biasGrad->data;
    float *l = (float *)loss->data;
    for (size_t f = 0; f < numFeatures; f++) {
        float sum = 0.0f;
        for (size_t n = 0; n < batch; n++) {
            sum += l[n * numFeatures + f];
        }
        bg[f] += sum;
    }
}

void linearCalcBiasGradsFloatWithConversion(linearConfig_t *linearConfig, tensor_t *loss) {
    tensor_t *paramBG = getGradFromParameter(linearConfig->bias);
    tensor_t *bG = paramBG;
    tensor_t *l = loss;

    tensor_t lossFloat;
    size_t sizeLoss = calcNumberOfElementsByTensor(loss);
    float lossFloatData[sizeLoss];
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);

    if (l->quantization->type != FLOAT32) {
        setTensorValuesForConversion((uint8_t *)lossFloatData, &lossFloatQ, l, &lossFloat);
        convertTensor(loss, &lossFloat);
        l = &lossFloat;
    }

    tensor_t biasGradFloat;
    size_t sizeBiasGrad = calcNumberOfElementsByTensor(bG);
    float biasGradFloatData[sizeBiasGrad];
    quantization_t biasGradFloatQ;
    initFloat32Quantization(&biasGradFloatQ);

    if (bG->quantization->type != FLOAT32) {
        setTensorValuesForConversion((uint8_t *)biasGradFloatData, &biasGradFloatQ, bG,
                                     &biasGradFloat);
        convertTensor(bG, &biasGradFloat);
        bG = &biasGradFloat;
    }

    linearCalcBiasGradsFloat32(l, bG);
    convertTensor(bG, paramBG);
}

void linearCalcPropLossFloat32(tensor_t *loss, tensor_t *weights, tensor_t *propLoss) {
    matmulFloat32Tensors(loss, weights, propLoss);
}

void linearCalcPropLossFloatWithConversion(linearConfig_t *linearConfig, tensor_t *loss,
                                           tensor_t *propLoss) {
    tensor_t *w = getParamFromParameter(linearConfig->weights);
    tensor_t *l = loss;
    tensor_t *pL = propLoss;

    /* #187: the inputs are converted below, but the result is written RAW into
     * the caller's propLoss. A non-FLOAT32 propLoss would receive float bits
     * into an int32 buffer — silent garbage. Fail fast (Matmul convention). */
    if (propLoss->quantization->type != FLOAT32) {
        PRINT_ERROR("Linear backward: propLossQ is FLOAT32 but the propLoss tensor is not (#187)");
        exit(1);
    }

    tensor_t weightsFloat;
    size_t sizeWeights = calcNumberOfElementsByTensor(w);
    float weightsFloatData[sizeWeights];
    quantization_t weightsFloatQ;
    initFloat32Quantization(&weightsFloatQ);

    if (w->quantization->type != FLOAT32) {
        setTensorValuesForConversion((uint8_t *)weightsFloatData, &weightsFloatQ, w, &weightsFloat);
        convertTensor(w, &weightsFloat);
        w = &weightsFloat;
    }

    tensor_t lossFloat;
    size_t sizeLoss = calcNumberOfElementsByTensor(l);
    float lossFloatData[sizeLoss];
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);

    if (l->quantization->type != FLOAT32) {
        setTensorValuesForConversion((uint8_t *)lossFloatData, &lossFloatQ, l, &lossFloat);
        convertTensor(loss, &lossFloat);
        l = &lossFloat;
    }

    linearCalcPropLossFloat32(l, w, pL);
}

void backwardFloat(linearConfig_t *linearConfig, tensor_t *forwardInput, tensor_t *loss,
                   tensor_t *propLossTensor) {
    size_t numberOfWeights = calcNumberOfElementsByShape(linearConfig->weights->param->shape);

    tensor_t *weightGrad = getGradFromParameter(linearConfig->weights);

    tensor_t *biasGrad = getGradFromParameter(linearConfig->bias);

    tensor_t intermediateWGrad;
    uint8_t intermediateWGradData[numberOfWeights * sizeof(float)];
    setTensorValues(&intermediateWGrad, intermediateWGradData, weightGrad->shape,
                    weightGrad->quantization, weightGrad->sparsity);

    linearCalcWeightGradsFloat32(forwardInput, loss, &intermediateWGrad);
    addFloat32TensorsInplace(weightGrad, &intermediateWGrad);

    linearCalcBiasGradsFloat32(loss, biasGrad);

    tensor_t *weightData = getParamFromParameter(linearConfig->weights);

    linearCalcPropLossFloat32(loss, weightData, propLossTensor);
}

void linearCalcWeightGradsSymInt32(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads) {
    size_t numberOfWeights = calcNumberOfElementsByTensor(weightGrads);

    symInt32QConfig_t *wgQC = weightGrads->quantization->qConfig;

    tensor_t intermediateWGrad;
    int32_t intermediateWGradData[numberOfWeights];
    symInt32QConfig_t intermediateWGradQC;
    initSymInt32QConfig(wgQC->roundingMode, &intermediateWGradQC);
    quantization_t intermediateWGradQ;
    initSymInt32Quantization(&intermediateWGradQC, &intermediateWGradQ);
    setTensorValues(&intermediateWGrad, (uint8_t *)intermediateWGradData, weightGrads->shape,
                    &intermediateWGradQ, weightGrads->sparsity);

    transposeTensor(loss, 1, 0);
    matmulSymInt32Tensors(loss, forwardInput, &intermediateWGrad);
    transposeTensor(loss, 1, 0);

    addSymInt32TensorsInplace(weightGrads, &intermediateWGrad);
}

void linearCalcWeightGradsSymInt32WithConversion(linearConfig_t *linearConfig, tensor_t *loss,
                                                 tensor_t *forwardInput) {
    symInt32QConfig_t *symInt32QC = linearConfig->weightGradQ->qConfig;
    roundingMode_t roundingMode = symInt32QC->roundingMode;

    tensor_t *paramWG = getGradFromParameter(linearConfig->weights);
    tensor_t *wG = paramWG;
    tensor_t *fwdIn = forwardInput;
    tensor_t *l = loss;

    tensor_t fwdInSymInt32;
    size_t sizeFwdInput = calcNumberOfElementsByTensor(fwdIn);
    int32_t fwdInSymInt32Data[sizeFwdInput];
    quantization_t fwdInSymInt32Q;
    symInt32QConfig_t fwdInSymInt32QC;
    initSymInt32QConfig(roundingMode, &fwdInSymInt32QC);
    initSymInt32Quantization(&fwdInSymInt32QC, &fwdInSymInt32Q);

    if (fwdIn->quantization->type != SYM_INT32) {
        setTensorValuesForConversion((uint8_t *)fwdInSymInt32Data, &fwdInSymInt32Q, fwdIn,
                                     &fwdInSymInt32);
        convertTensor(fwdIn, &fwdInSymInt32);
        fwdIn = &fwdInSymInt32;
    }

    tensor_t lossSymInt32;
    size_t sizeLoss = calcNumberOfElementsByTensor(loss);
    int32_t lossSymInt32Data[sizeLoss];
    quantization_t lossSymInt32Q;
    symInt32QConfig_t lossSymInt32QC;
    initSymInt32QConfig(roundingMode, &lossSymInt32QC);
    initSymInt32Quantization(&lossSymInt32QC, &lossSymInt32Q);

    if (loss->quantization->type != SYM_INT32) {
        setTensorValuesForConversion((uint8_t *)lossSymInt32Data, &lossSymInt32Q, loss,
                                     &lossSymInt32);
        convertTensor(loss, &lossSymInt32);
        l = &lossSymInt32;
    }

    tensor_t wGSymInt32;
    size_t sizeWeightGrads = calcNumberOfElementsByTensor(wG);
    int32_t wGSymInt32Data[sizeWeightGrads];
    quantization_t wGSymInt32Q;
    symInt32QConfig_t wGSymInt32QC;
    initSymInt32QConfig(roundingMode, &wGSymInt32QC);
    initSymInt32Quantization(&wGSymInt32QC, &wGSymInt32Q);

    if (wG->quantization->type != SYM_INT32) {
        setTensorValuesForConversion((uint8_t *)wGSymInt32Data, &wGSymInt32Q, wG, &wGSymInt32);
        convertTensor(wG, &wGSymInt32);
        wG = &wGSymInt32;
    }

    linearCalcWeightGradsSymInt32(l, fwdIn, wG);
    convertTensor(wG, paramWG);
}

void linearCalcBiasGradsSymInt32(tensor_t *biasGrads, tensor_t *loss) {
    /* Reduce loss over the batch axis into the rank-1 bias grad, rescaling from
     * the loss scale into the bias-grad scale (fixed-point), mirroring the
     * forward seed convention. */
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrads);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    size_t batch = (numFeatures == 0) ? 0 : numLoss / numFeatures;
    int32_t *bg = (int32_t *)biasGrads->data;
    int32_t *l = (int32_t *)loss->data;
    float lossScale = ((symInt32QConfig_t *)loss->quantization->qConfig)->scale;
    float bgScale = ((symInt32QConfig_t *)biasGrads->quantization->qConfig)->scale;
    for (size_t f = 0; f < numFeatures; f++) {
        /* int32 accumulator (NO int64 in SYM paths): loss mantissas are
         * int16-range per the qMaxBits<=16 contract, so the batch sum stays
         * within int32 for any batch <= 65536 — far beyond any real batch. */
        int32_t sum = 0;
        for (size_t n = 0; n < batch; n++) {
            sum += l[n * numFeatures + f];
        }
        bg[f] += (int32_t)roundf((float)sum * lossScale / bgScale);
    }
}

void linearCalcBiasGradsSymInt32WithConversion(linearConfig_t *linearConfig, tensor_t *loss) {
    symInt32QConfig_t *symInt32QC = linearConfig->weightGradQ->qConfig;
    roundingMode_t roundingMode = symInt32QC->roundingMode;

    tensor_t *paramBG = getGradFromParameter(linearConfig->bias);
    tensor_t *bG = paramBG;
    tensor_t *l = loss;

    tensor_t lossSymInt32;
    size_t sizeLoss = calcNumberOfElementsByTensor(l);
    int32_t lossSymInt32Data[sizeLoss];
    quantization_t lossSymInt32Q;
    symInt32QConfig_t lossSymInt32QC;
    initSymInt32QConfig(roundingMode, &lossSymInt32QC);
    initSymInt32Quantization(&lossSymInt32QC, &lossSymInt32Q);

    if (l->quantization->type != SYM_INT32) {
        setTensorValuesForConversion((uint8_t *)lossSymInt32Data, &lossSymInt32Q, l, &lossSymInt32);
        convertTensor(l, &lossSymInt32);
        l = &lossSymInt32;
    }

    tensor_t bGSymInt32;
    size_t sizeBias = calcNumberOfElementsByTensor(bG);
    int32_t bGSymInt32Data[sizeBias];
    quantization_t bGSymInt32Q;
    symInt32QConfig_t bGSymInt32QC;
    initSymInt32QConfig(roundingMode, &bGSymInt32QC);
    initSymInt32Quantization(&bGSymInt32QC, &bGSymInt32Q);

    if (bG->quantization->type != SYM_INT32) {
        setTensorValuesForConversion((uint8_t *)bGSymInt32Data, &bGSymInt32Q, bG, &bGSymInt32);
        convertTensor(bG, &bGSymInt32);
        bG = &bGSymInt32;
    }

    linearCalcBiasGradsSymInt32(bG, l);
    convertTensor(bG, paramBG);
}

void linearCalcPropLossSymInt32(tensor_t *weights, tensor_t *loss, tensor_t *propLoss) {
    matmulSymInt32Tensors(loss, weights, propLoss);
}

void linearCalcPropLossSymInt32WithConversion(linearConfig_t *linearConfig, tensor_t *loss,
                                              tensor_t *propLoss) {
    symInt32QConfig_t *symInt32QC = linearConfig->weightGradQ->qConfig;
    roundingMode_t roundingMode = symInt32QC->roundingMode;

    tensor_t *w = getParamFromParameter(linearConfig->weights);
    tensor_t *l = loss;

    /* #187: matmulSymInt32Tensors writes the result scale through propLoss's
     * qConfig; a FLOAT32 propLoss has qConfig == NULL -> segfault. Fail fast
     * (Matmul convention). */
    if (propLoss->quantization->type != SYM_INT32) {
        PRINT_ERROR(
            "Linear backward: propLossQ is SYM_INT32 but the propLoss tensor is not (#187)");
        exit(1);
    }

    tensor_t wSymInt32;
    size_t sizeWeights = calcNumberOfElementsByTensor(w);
    int32_t wSymInt32Data[sizeWeights];
    quantization_t wSymInt32Q;
    symInt32QConfig_t wSymInt32QC;
    initSymInt32QConfig(roundingMode, &wSymInt32QC);
    initSymInt32Quantization(&wSymInt32QC, &wSymInt32Q);

    if (w->quantization->type != SYM_INT32) {
        setTensorValuesForConversion((uint8_t *)wSymInt32Data, &wSymInt32Q, w, &wSymInt32);
        convertTensor(w, &wSymInt32);
        w = &wSymInt32;
    }

    tensor_t lSymInt32;
    size_t sizeL = calcNumberOfElementsByTensor(l);
    int32_t lSymInt32Data[sizeL];
    quantization_t lSymInt32Q;
    symInt32QConfig_t lSymInt32QC;
    initSymInt32QConfig(roundingMode, &lSymInt32QC);
    initSymInt32Quantization(&lSymInt32QC, &lSymInt32Q);

    if (l->quantization->type != SYM_INT32) {
        setTensorValuesForConversion((uint8_t *)lSymInt32Data, &lSymInt32Q, l, &lSymInt32);
        convertTensor(l, &lSymInt32);
        l = &lSymInt32;
    }

    linearCalcPropLossSymInt32(w, l, propLoss);
}

void backwardSymInt32(linearConfig_t *linearConfig, tensor_t *forwardInput, tensor_t *loss,
                      tensor_t *propLoss) {
    size_t numberOfWeights = calcNumberOfElementsByShape(linearConfig->weights->param->shape);

    tensor_t *weights = getParamFromParameter(linearConfig->weights);
    tensor_t *weightGrads = getGradFromParameter(linearConfig->weights);
    tensor_t *biasGrads = getGradFromParameter(linearConfig->bias);

    symInt32QConfig_t *weightGradsSymInt32QC = weightGrads->quantization->qConfig;

    tensor_t intermediateWeightGradsSymInt32;
    symInt32QConfig_t intermediateWeightGradsQC;
    initSymInt32QConfig(weightGradsSymInt32QC->roundingMode, &intermediateWeightGradsQC);
    quantization_t intermediateWeightGradsQ;
    initSymInt32Quantization(&intermediateWeightGradsQC, &intermediateWeightGradsQ);
    int32_t intermediateWeightGradsData[numberOfWeights];
    setTensorValues(&intermediateWeightGradsSymInt32, (uint8_t *)intermediateWeightGradsData,
                    weightGrads->shape, &intermediateWeightGradsQ, NULL);

    linearCalcWeightGradsSymInt32(loss, forwardInput, &intermediateWeightGradsSymInt32);
    addSymInt32TensorsInplace(weightGrads, &intermediateWeightGradsSymInt32);

    linearCalcBiasGradsSymInt32(biasGrads, loss);

    linearCalcPropLossSymInt32(weights, loss, propLoss);
}

void linearBackward(layer_t *linearLayer, tensor_t *forwardInput, tensor_t *loss,
                    tensor_t *propLoss) {
    linearConfig_t *linearConfig = linearLayer->config->linear;

    switch (linearConfig->weightGradQ->type) {
    case FLOAT32:
        linearCalcWeightGradsFloatWithConversion(linearConfig, forwardInput, loss);
        break;
    case SYM_INT32:
        linearCalcWeightGradsSymInt32WithConversion(linearConfig, loss, forwardInput);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }

    switch (linearConfig->biasGradQ->type) {
    case FLOAT32:
        linearCalcBiasGradsFloatWithConversion(linearConfig, loss);
        break;
    case SYM_INT32:
        linearCalcBiasGradsSymInt32WithConversion(linearConfig, loss);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }

    switch (linearConfig->propLossQ->type) {
    case FLOAT32:
        linearCalcPropLossFloatWithConversion(linearConfig, loss, propLoss);
        break;
    case SYM_INT32:
        linearCalcPropLossSymInt32WithConversion(linearConfig, loss, propLoss);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

void linearCalcOutputShape(layer_t *linearLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 2) {
        PRINT_ERROR("Linear layer expects 2D input, got %luD\n", inputShape->numberOfDimensions);
    }

    size_t batchSize = inputShape->dimensions[0];

    linearConfig_t *cfg = linearLayer->config->linear;
    shape_t *weightShape = cfg->weights->param->shape;
    size_t outFeatures = weightShape->dimensions[0];

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outFeatures;

    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
