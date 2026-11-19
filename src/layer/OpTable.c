#define SOURCE_FILE "OP_TABLE"

#include "OpTable.h"

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

#define UNARY {.shape = opUnaryShape, .arity = 1u}

const opFunctions_t opFunctions[] = {
    [LINEAR] = UNARY,       [RELU] = UNARY,
    [CONV1D] = UNARY,       [CONV1D_TRANSPOSED] = UNARY,
    [MAXPOOL1D] = UNARY,    [AVGPOOL1D] = UNARY,
    [SOFTMAX] = UNARY,      [FLATTEN] = UNARY,
    [QUANTIZATION] = UNARY, [ADAPTIVE_AVGPOOL1D] = UNARY,
    [DROPOUT] = UNARY,      [LAYERNORM] = UNARY,
    [GROUPNORM] = UNARY,    [BATCHNORM1D] = UNARY,
};

_Static_assert(sizeof opFunctions / sizeof opFunctions[0] == OP_TABLE_ROWS,
               "one op row per layerType_t member");
