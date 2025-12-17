#include "Layer.h"
#include "Linear.h"
#include "Relu.h"
#include "Softmax.h"

layerFunctions_t layerFunctions[] = {
    [LINEAR] = {linearForward, linearBackward, linearCalcOutputShape},
    [RELU] = {reluForward, reluBackward, reluCalcOutputShape},
    [CONV1D] = {NULL, NULL, NULL},
    [SOFTMAX] = {softmaxForward, softmaxBackward, softmaxCalcOutputShape}
};

void initLayer(layer_t *layer, layerType_t type, layerConfig_t* config) {
    layer->type = type;
    layer->config = config;
}