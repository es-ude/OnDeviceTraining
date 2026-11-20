#define SOURCE_FILE "OP_TABLE"

#include "OpTable.h"
#include "AddLayer.h"

/* The one forward rank rule: every layer keeps its input rank except Flatten,
 * which yields rank 2. Dispatches through layerFunctions[] at call time, never
 * a copied pointer, because tests patch that table and the patch must reach
 * every driver. */
static void opUnaryShape(layer_t *layer, const shape_t *const *in, uint8_t k, shape_t *out) {
    (void)k;
    out->numberOfDimensions = (layer->type == FLATTEN) ? 2u : in[0]->numberOfDimensions;
    /* Safe cast: calcOutputShapeFn_t predates const and no implementation
     * writes its input shape. */
    layerFunctions[layer->type].calcOutputShape(layer, (shape_t *)in[0], out);
}

/* Like opUnaryShape, index layerFunctions[] per call so a patched slot is reached. */
static void opUnaryForward(layer_t *layer, tensor_t *const *in, uint8_t k, tensor_t *out,
                           const layerExec_t *exec) {
    (void)k;
    layerFunctions[layer->type].forward(layer, in[0], out, exec);
}

/* k and dxMode are ignored: arity is validated at the entry and a unary row is never asked
 * for ACC. dx[0] == NULL passes through as propLoss NULL (a grads-only call). */
static void opUnaryBackward(layer_t *layer, tensor_t *const *in, uint8_t k, tensor_t *dy,
                            tensor_t *const *dx, const uint8_t *dxMode, const layerExec_t *exec) {
    (void)k;
    (void)dxMode;
    layerFunctions[layer->type].backward(layer, in[0], dy, dx[0], exec);
}

#define UNARY(reads)                                                                               \
    {.forward = opUnaryForward,                                                                    \
     .backward = opUnaryBackward,                                                                  \
     .shape = opUnaryShape,                                                                        \
     .arity = 1u,                                                                                  \
     .readsInputs = (reads),                                                                       \
     .readsOutput = false,                                                                         \
     .accumulatesDx = false}

const opFunctions_t opFunctions[] = {
    /* x feeds only the weight grad (dW from dy and x); a frozen layer skips it. */
    [LINEAR] = UNARY(OP_READS_IF_TRAINABLE),
    [CONV1D] = UNARY(OP_READS_IF_TRAINABLE),
    [CONV1D_TRANSPOSED] = UNARY(OP_READS_IF_TRAINABLE),
    /* Norms (LayerNorm/GroupNorm/BatchNorm1d) recompute x-hat from x, frozen or not; ReLU
     * masks dy by the sign of x; Softmax recomputes softmax(x) for its Jacobian. */
    [LAYERNORM] = UNARY(OP_READS_ALWAYS),
    [GROUPNORM] = UNARY(OP_READS_ALWAYS),
    [BATCHNORM1D] = UNARY(OP_READS_ALWAYS),
    [RELU] = UNARY(OP_READS_ALWAYS),
    [SOFTMAX] = UNARY(OP_READS_ALWAYS),
    /* Pools route dy by argmax indices or window geometry; Flatten reshapes; Quantization is
     * straight-through; Dropout uses its stored mask and p. */
    [MAXPOOL1D] = UNARY(OP_READS_NEVER),
    [AVGPOOL1D] = UNARY(OP_READS_NEVER),
    [ADAPTIVE_AVGPOOL1D] = UNARY(OP_READS_NEVER),
    [FLATTEN] = UNARY(OP_READS_NEVER),
    [QUANTIZATION] = UNARY(OP_READS_NEVER),
    [DROPOUT] = UNARY(OP_READS_NEVER),
    /* dx = dy on both slots: no input and no output is read. */
    [ADD] = {.forward = addForward,
             .backward = addBackward,
             .shape = addShape,
             .arity = 2u,
             .readsInputs = OP_READS_NEVER,
             .readsOutput = false,
             .accumulatesDx = true},
};

_Static_assert(sizeof opFunctions / sizeof opFunctions[0] == OP_TABLE_ROWS,
               "one op row per layerType_t member");
