#include "TensorConversion.h"
#include "QuantizationLayer.h"

void quantization(tensor_t *input, tensor_t *output) {
    qtype_t inputQType = input->quantization->type;
    qtype_t outputQType = output->quantization->type;
    conversionFunction_t conversion = conversionMatrix[inputQType][outputQType];
    conversion(input, output);
}