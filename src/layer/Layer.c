#define SOURCE_FILE "LAYER"

#include "Layer.h"
#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "Flatten.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Softmax.h"

layerFunctions_t layerFunctions[] = {
    [LINEAR] = {linearForward, linearBackward, linearCalcOutputShape},
    [RELU] = {reluForward, reluBackward, reluCalcOutputShape},
    [CONV1D] = {conv1dForward, conv1dBackward, conv1dCalcOutputShape},
    [CONV1D_TRANSPOSED] = {conv1dTransposedForward, conv1dTransposedBackward,
                           conv1dTransposedCalcOutputShape},
    [MAXPOOL1D] = {maxPool1dForward, maxPool1dBackward, maxPool1dCalcOutputShape},
    [AVGPOOL1D] = {avgPool1dForward, avgPool1dBackward, avgPool1dCalcOutputShape},
    [SOFTMAX] = {softmaxForward, softmaxBackward, softmaxCalcOutputShape},
    [FLATTEN] = {flattenForward, flattenBackward, flattenCalcOutputShape},
    [QUANTIZATION] = {quantizationForward, quantizationBackward, quantizationCalcOutputShape},
    [ADAPTIVE_AVGPOOL1D] = {adaptiveAvgPool1dForward, adaptiveAvgPool1dBackward,
                            adaptiveAvgPool1dCalcOutputShape},
    [DROPOUT] = {dropoutForward, dropoutBackward, dropoutCalcOutputShape},
    [LAYERNORM] = {layerNormForward, layerNormBackward, layerNormCalcOutputShape}};

void initLayer(layer_t *layer, layerType_t type, layerConfig_t *config) {
    layer->type = type;
    layer->config = config;
}
