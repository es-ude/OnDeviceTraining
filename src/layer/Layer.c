#define SOURCE_FILE "LAYER"

#include "Layer.h"
#include "Conv1d.h"
#include "Flatten.h"
#include "Linear.h"
#include "Relu.h"
#include "Softmax.h"

layerFunctions_t layerFunctions[] = {
    [LINEAR] = {linearForward, linearBackward, linearCalcOutputShape},
    [RELU] = {reluForward, reluBackward, reluCalcOutputShape},
    [CONV1D] = {conv1dForward, conv1dBackward, conv1dCalcOutputShape},
    [SOFTMAX] = {softmaxForward, softmaxBackward, softmaxCalcOutputShape},
    [FLATTEN] = {flattenForward, flattenBackward, flattenCalcOutputShape}};

void initLayer(layer_t *layer, layerType_t type, layerConfig_t *config) {
    layer->type = type;
    layer->config = config;
}
