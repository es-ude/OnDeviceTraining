#ifndef ENV5_RUNTIME_EXECUTE_OP_H
#define ENV5_RUNTIME_EXECUTE_OP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Tensor.h"

/* The one conversion funnel (design spec 2026-07-03 PR1b.2, D1). Every op runs:
 *   prologue   — operands whose dtype != arithmetic are converted into
 *                transient stack scratch (sources are never mutated)
 *   kernel     — pure computation: operands (in arithmetic representation)
 *                -> raw intermediate (SYM kernels emit raw int32 mantissas);
 *                also given auxOut (kernel-written verbatim, e.g. MaxPool's
 *                argmax indices) and ctx (kernel geometry/config), neither of
 *                which the funnel touches
 *   epilogue   — the intermediate is written/accumulated into the target,
 *                in the TARGET's own dtype. Persistent buffers are never
 *                pulled through the arithmetic. auxOut is NEVER funnel-
 *                converted — the kernel writes it in its own storage format.
 *                Rounding ownership (#282): the OUT_WRITE requant rounds by
 *                the OP's arithmetic.roundingMode; the ACC epilogues round by
 *                the accumulator's own storage qConfig (grid discipline, D4).
 * Escape hatch policy: the prologue/epilogue helpers stay static in
 * ExecuteOp.c. Opening one for an op that does not fit the n-inputs/1-output
 * shape requires a documented exception here. */

typedef void (*opKernelFn_t)(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                             tensor_t *auxOut, const void *ctx);

typedef enum {
    OUT_WRITE,               /* overwrite target (wire / forward output); SYM->SYM
                              * routes through the conversionMatrix diagonal
                              * (requant), never the same-type memmove */
    OUT_ACC_DYNAMIC_RESCALE, /* Strategy A: dequant both -> float-add -> fresh
                              * absmax scale (SYM targets); exact add (FLOAT32) */
    OUT_ACC_FIXED_SCALE      /* bias scheme: rescale increment into the target's
                              * EXISTING scale via rescaleIntoAccumulatorScale
                              * (honors the TARGET's roundingMode), integer
                              * add, no clamp (SYM targets); exact add (FLOAT32) */
} outputMode_t;

/*! @brief Descriptor for a funnel op invocation (design spec D1).
 * ctx: kernel geometry/config, opaque to the funnel, passed straight through
 *      to the kernel; NULL for config-free kernels.
 * auxOut: kernel-written verbatim, in ITS OWN storage format — never funnel-
 *         converted; NULL for single-output ops (e.g. MaxPool1d's argmax
 *         indices live here). */
typedef struct opSpec {
    opKernelFn_t kernel; // has nothing to do with convolution
    const void *ctx;
    tensor_t **inputs;
    size_t nInputs;
    arithmetic_t arithmetic;
    outputMode_t mode;
    tensor_t *auxOut;
    /* #296 Stage 1: opt-in for OUT_WRITE aliasing when the target is also an
     * input. true == the kernel reads element i of every input before writing
     * element i of the output (elementwise), so writing the target's storage
     * directly is safe. Zero-init (false) = conservative staging. Aliasing
     * additionally requires FLOAT32 arithmetic AND a FLOAT32 target — for any
     * other combination the epilogue's conversion/width-restore is load-
     * bearing and the funnel always stages. Alias detection is exact-base-
     * pointer equality; overlapping sub-views (raw-pointer tensor wiring) are
     * outside the contract. */
    bool writesInPlaceSafe;
    /* Group-quant PR2 (Task 3; final-review Fix 2/3) + PR4 (Task 3):
     * per-OPERAND opt-in for a grouped input — SYM OR ASYM (symQConfig_t /
     * asymQConfig_t numGroups > 1; the two grouped carrier dtypes share the
     * shape grammar, D6) — under EITHER arithmetic type. 0 = no grouped
     * operand allowed anywhere (zero-init safe — every existing opSpec
     * compound literal that never heard of grouped operands still denies
     * them); i+1 = inputs[i] (and ONLY inputs[i]) may be grouped. A grouped
     * input reaching the prologue at any OTHER position, or at all when this
     * is 0, fail-fasts. Declaring ops as of group-quant PR3/PR4: the
     * GEMM-family forward AND dx weights (Linear/Conv1d/ConvT1d, both
     * directions) and the optimizer param-update ops (SGD
     * stateless/mState/mParam, AdamW param). Every other op is a non-carrier
     * (spec §3: grads, bias, gamma/beta, wires, momentum stay per-tensor) —
     * this funnel-wide seam enforces that, on BOTH the ARITH_SYM_INT32 and
     * ARITH_FLOAT32 prologue arms.
     *
     * ARITH_SYM_INT32: the prologue unpacks the declared operand's mantissas
     * into scratch — grouped SYM via unpackSignExtend (sign-extended raw
     * int32), grouped ASYM via a zero-extend + per-element `code - zp[g]`
     * shift (PR4: after the shift both dtypes present the IDENTICAL
     * signed-mantissa image, D5) — and POISONS the scratch
     * symInt32QConfig_t's scale to 1.0f and qMaxBits to the source's qBits: a
     * grouped operand has no single scalar scale, so any kernel reading
     * scratch->quantization->scale here is a bug — group-aware kernels MUST
     * take per-group scales from their own ctx (e.g. Matmul's weightGroups;
     * for ASYM weights the layer passes a symQConfig-shaped VIEW of the asym
     * config), never this field. Set together with a ctx that actually
     * carries the group shape (Linear.c: both or neither).
     *
     * ARITH_FLOAT32: no poisoning/unpack mechanics — the declared operand
     * proceeds through the EXISTING group-aware convertTensor dequant
     * (convertSymTensorToFloat32Tensor / convertAsymTensorToFloatTensor,
     * group-aware since PR2/PR4 Task 2); this field is purely a gate on that
     * path, not a different mechanism. */
    size_t groupedSymOperandPos;
} opSpec_t;

void executeOp(const opSpec_t *spec, tensor_t *target);

/* Fail-fast guard for the OUT_WRITE==0 hazard (PR3 spec D1, per-layer
 * accumulate-mode knob): weightGradAccMode/biasGradAccMode are by-value
 * layerQuant_t/config fields with no "unset" sentinel of their own, and
 * OUT_WRITE happens to be the zero-init value -- a hand-wired config that
 * forgets to set them would otherwise silently pass OUT_WRITE into a grad
 * accumulate call. Call at each grad executeOp call site, right before
 * reading the mode; `context` names the layer and field for the message
 * (e.g. "Linear weightGradAccMode"). Shared here (not duplicated per layer
 * file) since every grad-accumulating layer already depends on ExecuteOp for
 * the funnel call itself -- no new library dependency. */
static inline void executeOpValidateAccMode(outputMode_t mode, const char *context) {
    if (mode != OUT_ACC_DYNAMIC_RESCALE && mode != OUT_ACC_FIXED_SCALE) {
        PRINT_ERROR("%s: not a valid grad accumulate mode (got %d) -- config field never set? "
                    "(PR3 spec, #261)",
                    context, (int)mode);
        exit(1);
    }
}

/* Copies operand 0 into rawOut (data + SYM scale if applicable). For ops whose
 * increment is produced inline (LayerNorm dgamma/dbeta only — the Quantization
 * layer is a pure conversion node routed through executeConvert instead).
 * Ignores auxOut/ctx. */
void executeOpIdentityKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                             tensor_t *auxOut, const void *ctx);

/* Kernel-less funnel form: storage-to-storage conversion (1 input,
 * OUT_WRITE semantics). SYM->SYM routes through the conversionMatrix
 * diagonal (requant), never the same-type memmove. Supports every
 * populated conversionMatrix cell. Unlike executeOp's OUT_WRITE epilogue,
 * a bare conversion IS a storage encode: it rounds by the target's own
 * qConfig roundingMode (#282). */
void executeConvert(tensor_t *input, tensor_t *target);

#endif // ENV5_RUNTIME_EXECUTE_OP_H
