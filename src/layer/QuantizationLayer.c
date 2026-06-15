#define SOURCE_FILE "QUANTIZATION_LAYER"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "QuantizationLayer.h"
#include "TensorConversion.h"

/* Dispatch over (input dtype, output dtype) like convertTensor, with ONE
 * deliberate difference: convertTensor's same-type branch is a memmove +
 * scale copy (convertTensorsWithSameType, TensorConversion.c), which would
 * pass accumulator-range mantissas through UNCHANGED. A SYM_INT32->SYM_INT32
 * Quantization layer must REQUANTIZE instead, so the same-dtype SYM_INT32
 * case goes through the conversionMatrix diagonal (requantSymInt32Tensor,
 * wired in PR C). Any other same-dtype pair is a configuration error: a
 * Quantization layer that neither changes dtype nor requantizes is a
 * misconfigured no-op. Cross-dtype pairs route through convertTensor, which
 * carries the NULL-entry guard for unsupported pairs (PR B). */
static void dispatchQuantization(tensor_t *input, tensor_t *output) {
    qtype_t inputDType = input->quantization->type;
    qtype_t outputDType = output->quantization->type;

    if (inputDType == outputDType) {
        if (inputDType != SYM_INT32) {
            PRINT_ERROR("Quantization layer: same input/output dtype %d is only supported "
                        "for SYM_INT32 (requant); other same-dtype pairs are a config error",
                        (int)inputDType);
            exit(1);
        }
        conversionMatrix[SYM_INT32][SYM_INT32](input, output);
        return;
    }
    convertTensor(input, output);
}

void quantizationForward(layer_t *layer, tensor_t *inputTensor, tensor_t *outputTensor) {
    (void)layer; /* dtype dispatch reads the tensors; config->quantization->forwardQ
                  * already determined outputTensor's dtype at allocation time
                  * (initLayerOutputs / initBufferOutput). */
    dispatchQuantization(inputTensor, outputTensor);
}

void quantizationBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss,
                          tensor_t *propLoss) {
    (void)layer;
    (void)forwardInput; /* straight-through: dy does not depend on x. Both forward
                         * and backward requant map their own absmax exactly onto
                         * qMax, so neither pass saturates and no STE clipping mask
                         * is needed (spec D3). */
    dispatchQuantization(loss, propLoss);
}

void quantizationCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    (void)layer;
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
