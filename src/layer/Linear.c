#include <stdio.h>

#include "Add.h"
#include "Layer.h"
#include "Matmul.h"
#include "Rounding.h"
#include "TensorConversion.h"
#include "Linear.h"


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
    matmulFloat32Tensors(input, w, output);
    transposeTensor(w, 0, 1);
    addFloat32TensorsInplace(output, b);
}

void linearForwardSymInt32(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulSymInt32Tensors(input, w, output);
    transposeTensor(w, 0, 1);

    addSymInt32TensorsInplace(output, b);
}

void linearForward(layer_t *linearLayer, tensor_t *input, tensor_t *output) {
    linearConfig_t *linearConfig = linearLayer->config->linear;

    tensor_t *weights = getTensorFromParameter(linearConfig->weights);
    tensor_t *bias = getTensorFromParameter(linearConfig->bias);

    switch (linearConfig->forwardQ->type) {
    case FLOAT32:
        linearForwardFloat(weights, bias, input, output);
        break;
    case SYM_INT32:
        linearForwardSymInt32(weights, bias, input, output);
        break;
    default:
        break;
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

    matmulFloat32Tensors(loss, forwardInput, &intermediateWGrad);
    addFloat32TensorsInplace(weightGrads, &intermediateWGrad);
}

void linearCalcWeightGradsFloatWithConversion(linearConfig_t *linearConfig, tensor_t *forwardInput,
                                              tensor_t *loss) {
    tensor_t *paramWG = getGradTensorFromParameter(linearConfig->weights);
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
    addFloat32TensorsInplace(biasGrad, loss);
}

void linearCalcBiasGradsFloatWithConversion(linearConfig_t *linearConfig, tensor_t *loss) {
    tensor_t *paramBG = getGradTensorFromParameter(linearConfig->bias);
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
    transposeTensor(loss, 0, 1);
    matmulFloat32Tensors(loss, weights, propLoss);
    transposeTensor(loss, 0, 1);
}

void linearCalcPropLossFloatWithConversion(linearConfig_t *linearConfig, tensor_t *loss,
                                           tensor_t *propLoss) {
    tensor_t *w = getTensorFromParameter(linearConfig->weights);
    tensor_t *l = loss;
    tensor_t *pL = propLoss;

    tensor_t weightsFloat;
    size_t sizeWeights = calcNumberOfElementsByTensor(w);
    float weightsFloatData[sizeWeights];
    quantization_t weightsFloatQ;
    initFloat32Quantization(&weightsFloatQ);

    if (w->quantization->type != FLOAT32) {
        setTensorValuesForConversion((uint8_t *)weightsFloatData, &weightsFloatQ, w,
                                     &weightsFloat);
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

    linearCalcPropLossFloat32(w, l, pL);
}


void backwardFloat(linearConfig_t *linearConfig, tensor_t *forwardInput, tensor_t *loss,
                   tensor_t *propLossTensor) {
    size_t numberOfWeights =
        calcNumberOfElementsByShape(linearConfig->weights->param->shape);

    tensor_t *weightGrad = getGradTensorFromParameter(linearConfig->weights);

    tensor_t *biasGrad = getGradTensorFromParameter(linearConfig->bias);

    tensor_t intermediateWGrad;
    uint8_t intermediateWGradData[numberOfWeights * sizeof(float)];
    setTensorValues(&intermediateWGrad, intermediateWGradData, weightGrad->shape,
                    weightGrad->quantization, weightGrad->sparsity);

    linearCalcWeightGradsFloat32(forwardInput, loss, &intermediateWGrad);
    addFloat32TensorsInplace(weightGrad, &intermediateWGrad);

    linearCalcBiasGradsFloat32(loss, biasGrad);

    tensor_t *weightData = getTensorFromParameter(linearConfig->weights);

    linearCalcPropLossFloat32(loss, weightData, propLossTensor);
}


void linearCalcWeightGradsSymInt32(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads) {
    matmulSymInt32Tensors(loss, forwardInput, weightGrads);
}

void linearCalcWeightGradsSymInt32WithConversion(linearConfig_t *linearConfig, tensor_t *loss,
                                                 tensor_t *forwardInput) {
    symInt32QConfig_t *symInt32QC = linearConfig->weightGradQ->qConfig;
    roundingMode_t roundingMode = symInt32QC->roundingMode;

    tensor_t *paramWG = getGradTensorFromParameter(linearConfig->weights);
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
    uint32_t lossSymInt32Data[sizeLoss];
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

    tensor_t intermediateWG;
    int32_t intermediateWGData[sizeWeightGrads];
    quantization_t intermediateWGQ;
    symInt32QConfig_t intermediateWGQC;
    initSymInt32QConfig(roundingMode, &intermediateWGQC);
    initSymInt32Quantization(&intermediateWGQC, &intermediateWGQ);
    setTensorValues(&intermediateWG, (uint8_t *)intermediateWGData, wG->shape, &intermediateWGQ, wG->sparsity);

    linearCalcWeightGradsSymInt32(l, fwdIn, &intermediateWG);
    addSymInt32TensorsInplace(wG, &intermediateWG);
    convertTensor(wG, paramWG);
}


void linearCalcBiasGradsSymInt32(tensor_t *biasGrads, tensor_t *loss) {
    addSymInt32TensorsInplace(biasGrads, loss);
}

void linearCalcBiasGradsSymInt32WithConversion(linearConfig_t *linearConfig, tensor_t *loss) {
    symInt32QConfig_t *symInt32QC = linearConfig->weightGradQ->qConfig;
    roundingMode_t roundingMode = symInt32QC->roundingMode;

    tensor_t *paramBG = getGradTensorFromParameter(linearConfig->bias);
    tensor_t *bG = paramBG;
    tensor_t *l = loss;

    tensor_t lossSymInt32;
    size_t sizeLoss = calcNumberOfElementsByTensor(l);
    uint32_t lossSymInt32Data[sizeLoss];
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
    uint32_t bGSymInt32Data[sizeBias];
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
    transposeTensor(loss, 0, 1);
    matmulSymInt32Tensors(loss, weights, propLoss);
    transposeTensor(loss, 0, 1);
}

void linearCalcPropLossSymInt32WithConversion(linearConfig_t *linearConfig, tensor_t *loss,
                                              tensor_t *propLoss) {
    symInt32QConfig_t *symInt32QC = linearConfig->weightGradQ->qConfig;
    roundingMode_t roundingMode = symInt32QC->roundingMode;

    tensor_t *w = getTensorFromParameter(linearConfig->weights);
    tensor_t *l = loss;

    tensor_t wSymInt32;
    size_t sizeWeights = calcNumberOfElementsByTensor(w);
    uint32_t wSymInt32Data[sizeWeights];
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
    uint32_t lSymInt32Data[sizeL];
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
    size_t numberOfWeights =
        calcNumberOfElementsByShape(linearConfig->weights->param->shape);

    tensor_t *weights = getTensorFromParameter(linearConfig->weights);
    tensor_t *weightGrads = getGradTensorFromParameter(linearConfig->weights);
    tensor_t *biasGrads = getGradTensorFromParameter(linearConfig->bias);

    symInt32QConfig_t *weightGradsSymInt32QC = weightGrads->quantization->qConfig;

    tensor_t intermediateWeightGradsSymInt32;
    symInt32QConfig_t intermediateWeightGradsQC;
    initSymInt32QConfig(weightGradsSymInt32QC->roundingMode, &intermediateWeightGradsQC);
    quantization_t intermediateWeightGradsQ;
    initSymInt32Quantization(&intermediateWeightGradsQC, &intermediateWeightGradsQ);
    uint32_t intermediateWeightGradsData[numberOfWeights];
    setTensorValues(&intermediateWeightGradsSymInt32, (uint8_t *)intermediateWeightGradsData,
                    weightGrads->shape, &intermediateWeightGradsQ, NULL);

    linearCalcWeightGradsSymInt32(loss, forwardInput,
                                  &intermediateWeightGradsSymInt32);
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
        break;
    }

    switch (linearConfig->biasGradQ->type) {
    case FLOAT32:
        linearCalcBiasGradsFloatWithConversion(linearConfig, loss);
        break;
    case SYM_INT32:
        linearCalcBiasGradsSymInt32WithConversion(linearConfig, loss);
        break;
    default:
        break;
    }

    switch (linearConfig->propLossQ->type) {
    case FLOAT32:
        linearCalcPropLossFloatWithConversion(linearConfig, loss, propLoss);
        break;
    case SYM_INT32:
        linearCalcPropLossSymInt32WithConversion(linearConfig, loss, propLoss);
        break;
    default:
        break;
    }
}

void linearCalcOutputShape(layer_t *linearLayer, shape_t *inputShape, shape_t *outputShape) {
    size_t batchSize = inputShape->dimensions[0];

    linearConfig_t *cfg = linearLayer->config->linear;
    shape_t *weightShape = cfg->weights->param->shape;
    size_t outFeatures = weightShape->dimensions[0];

    outputShape->dimensions[0] = outFeatures;
    outputShape->dimensions[1] = batchSize;

    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
