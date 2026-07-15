#define SOURCE_FILE "ADAM-W"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "AdamW.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Mul.h"
#include "PointwiseFused.h"
#include "Tensor.h"

void adamWInit(adamW_t *adamW, float learningRate, double beta1, double beta2, double eps,
               double weightDecay, arithmetic_t updateMath) {
    /* Only ARITH_FLOAT32 update arithmetic is implemented; fail at
     * construction so a training run cannot start on an unimplemented
     * arm (#310 pattern, Sgd.c precedent). */
    if (updateMath.type != ARITH_FLOAT32) {
        PRINT_ERROR("AdamW updateMath: only ARITH_FLOAT32 is implemented "
                    "(integer-arithmetic update numerics not yet designed, #310 pattern)");
        exit(1);
    }
    /* Negated comparisons so NaN betas/eps fail too. */
    if (!(beta1 >= 0.0 && beta1 < 1.0) || !(beta2 >= 0.0 && beta2 < 1.0)) {
        PRINT_ERROR("AdamW: betas must lie in [0, 1) (got beta1=%f, beta2=%f)", beta1, beta2);
        exit(1);
    }
    if (!(eps > 0.0)) {
        PRINT_ERROR("AdamW: eps must be > 0 (got %f)", eps);
        exit(1);
    }
    adamW->learningRate = learningRate;
    adamW->beta1 = beta1;
    adamW->beta2 = beta2;
    adamW->eps = eps;
    adamW->weightDecay = weightDecay;
    adamW->stepCount = 0;
    adamW->updateMath = updateMath;
}

float adamWGetLr(optimizer_t *optimizer) {
    return optimizer->impl->adamW->learningRate;
}

void adamWSetLr(optimizer_t *optimizer, float learningRate) {
    optimizer->impl->adamW->learningRate = learningRate;
}

typedef struct {
    float w1; /* (float)(1 - beta1) */
} adamWMomentCtx_t;
typedef struct {
    float beta2, s2; /* (float)beta2, (float)(1 - beta2) */
} adamWVarianceCtx_t;
typedef struct {
    float decay, bc2sqrt, eps, stepScale;
} adamWParamCtx_t;

/* All three kernels use rawOut as the accumulator: op[0] is copied into
 * rawOut unless the funnel already aliased them (FLOAT32 fast path with
 * writesInPlaceSafe), then the inplace helpers run ON rawOut. This keeps
 * the kernels correct on the staged path too (quantized moments/params:
 * there op[0], rawOut and the target are three distinct buffers and the
 * epilogue reads only rawOut). The kernels allocate nothing. */
static void seedRawOutFromFirstOperand(tensor_t **op, tensor_t *rawOut) {
    if (rawOut->data != op[0]->data) {
        size_t numberOfValues = calcNumberOfElementsByTensor(rawOut);
        memcpy(rawOut->data, op[0]->data, numberOfValues * sizeof(float));
    }
}

/* operands {m, g} -> rawOut = lerp(m, g, w1) = m + w1*(g - m), fmaf-fused
 * exactly like ATen's small-weight branch (torch exp_avg.lerp_). */
static void adamWMomentKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                              const void *ctxv) {
    (void)n;
    (void)aux;
    const adamWMomentCtx_t *ctx = ctxv;
    seedRawOutFromFirstOperand(op, rawOut);
    lerpFloat32TensorsInplace(rawOut, op[1], ctx->w1);
}

/* operands {v, g} -> rawOut = beta2*v + s2*g*g (torch's own two passes:
 * exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)). */
static void adamWVarianceKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                                const void *ctxv) {
    (void)n;
    (void)aux;
    const adamWVarianceCtx_t *ctx = ctxv;
    seedRawOutFromFirstOperand(op, rawOut);
    mulFloat32ElementWithFloat32TensorInplace(rawOut, ctx->beta2);
    addcmulFloat32TensorsInplace(rawOut, op[1], op[1], ctx->s2);
}

/* operands {param, m, v} -> rawOut = decay*param
 *   + stepScale*m / (sqrt(v)/bc2sqrt + eps)
 * (torch: param.mul_(1-lr*wd) ... param.addcdiv_(exp_avg, denom,
 * value=-lr/bc1) -- same six per-element roundings, denom never
 * materialized). */
static void adamWParamKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                             const void *ctxv) {
    (void)n;
    (void)aux;
    const adamWParamCtx_t *ctx = ctxv;
    seedRawOutFromFirstOperand(op, rawOut);
    mulFloat32ElementWithFloat32TensorInplace(rawOut, ctx->decay);
    addcdivDenomFloat32TensorsInplace(rawOut, op[1], op[2], ctx->bc2sqrt, ctx->eps, ctx->stepScale);
}

void adamWStep(optimizer_t *optim) {
    adamW_t *adamW = optim->impl->adamW;

    /* Re-checked here (not only in adamWInit): hand-assembled optimizers
     * must hit the same wall (#310 pattern, Sgd.c precedent). */
    if (adamW->updateMath.type != ARITH_FLOAT32) {
        PRINT_ERROR("AdamW updateMath: only ARITH_FLOAT32 is implemented "
                    "(integer-arithmetic update numerics not yet designed, #310 pattern)");
        exit(1);
    }

    adamW->stepCount++;

    /* Per-step scalars in DOUBLE (PyTorch's python-double prelude), cast to
     * float ONCE at the ctx boundary below -- float-composed betas lose
     * 1-beta unrecoverably (AdamW.h numerics note). */
    double lr = (double)adamW->learningRate;
    double t = (double)adamW->stepCount;
    double w1 = 1.0 - adamW->beta1;
    double s2 = 1.0 - adamW->beta2;
    double bc1 = 1.0 - pow(adamW->beta1, t);
    double bc2sqrt = sqrt(1.0 - pow(adamW->beta2, t));
    double stepScale = -(lr / bc1);
    double decay = 1.0 - lr * adamW->weightDecay;

    adamWMomentCtx_t mc = {.w1 = (float)w1};
    adamWVarianceCtx_t vc = {.beta2 = (float)adamW->beta2, .s2 = (float)s2};
    adamWParamCtx_t pc = {.decay = (float)decay,
                          .bc2sqrt = (float)bc2sqrt,
                          .eps = (float)adamW->eps,
                          .stepScale = (float)stepScale};

    for (size_t i = 0; i < optim->sizeStates; i++) {
        parameter_t *p = optim->parameter[i];
        tensor_t *m = optim->states[i]->stateBuffers[0];
        tensor_t *v = optim->states[i]->stateBuffers[1];

        executeOp(&(opSpec_t){.kernel = adamWMomentKernel,
                              .ctx = &mc,
                              .inputs = (tensor_t *[]){m, p->grad},
                              .nInputs = 2,
                              .arithmetic = adamW->updateMath,
                              .mode = OUT_WRITE,
                              .auxOut = NULL,
                              .writesInPlaceSafe = true},
                  m);
        executeOp(&(opSpec_t){.kernel = adamWVarianceKernel,
                              .ctx = &vc,
                              .inputs = (tensor_t *[]){v, p->grad},
                              .nInputs = 2,
                              .arithmetic = adamW->updateMath,
                              .mode = OUT_WRITE,
                              .auxOut = NULL,
                              .writesInPlaceSafe = true},
                  v);
        executeOp(&(opSpec_t){.kernel = adamWParamKernel,
                              .ctx = &pc,
                              .inputs = (tensor_t *[]){p->param, m, v},
                              .nInputs = 3,
                              .arithmetic = adamW->updateMath,
                              .mode = OUT_WRITE,
                              .auxOut = NULL,
                              .writesInPlaceSafe = true},
                  p->param);
    }
}
