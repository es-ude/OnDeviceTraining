#ifndef ENV5_RUNTIME_EXECUTE_OP_H
#define ENV5_RUNTIME_EXECUTE_OP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Tensor.h"

typedef struct rngStream rngStream_t;
typedef struct rngKey rngKey_t;

/* Which ops emit the integer raw (BFP_RAW: int32 mantissas + one UNBIASED
 * int16 exponent per TARGET group, bfpRawQConfig_t; the kernel MUST write
 * every group's exponent) instead of the FLOAT32 raw. Each value is an
 * explicit per-op opt-in, so an op never switches raw format by accident
 * (each layer's backward sets its own value). Every value is ARITH_BFP
 * only. Raw geometry: a BFP target's {numGroups, groupSize}; a FLOAT32
 * target has no groups, so the raw is per-tensor ({1,0}, one exponent over
 * the whole output).
 *   BFP_RAW_FORWARD      the integer forward. Also the forward marker: into a
 *                        BFP target, a FLOAT32-stored operand is a wiring
 *                        error (the forward is float-free after the input
 *                        quantization layer), not a staging request.
 *   BFP_RAW_BACKWARD_DX  a backward op writing a dx wire (Conv1d, Linear,
 *                        MaxPool1d, AvgPool1d, Softmax dx, the Quantization
 *                        layer's BFP -> BFP dx).
 *                        FLOAT32-stored operands keep staging, as in every
 *                        backward op.
 *   Both: OUT_WRITE only. The epilogue is the integer re-block
 *   (conversionMatrix[BFP_RAW][BFP], under the op-rounding swap, so the
 *   wire rounds by the OP's mode) or, for a FLOAT32 target, the dequant
 *   ([BFP_RAW][FLOAT32]); any other target dtype fails fast.
 *   BFP_RAW_BACKWARD_ACC a grad op accumulating into BFP grad STORAGE
 *                        (Conv1d, Linear weightGrad / biasGrad). OUT_ACC_
 *                        DYNAMIC_RESCALE into a BFP target only: the epilogue
 *                        is the integer engine accumulateBfpRawIntoBfpRescale
 *                        (rounds by the grad's own storage mode -- the ACC
 *                        rule below -- one draw per element in element order
 *                        from the op's ACC(0) stream). A FLOAT32 grad and the
 *                        fixed grid (OUT_ACC_FIXED_SCALE) are not this
 *                        contract: the layer keeps NONE (FLOAT32 raw) there,
 *                        and either combination here fails fast. FLOAT32-
 *                        stored operands keep staging. */
typedef enum bfpRawContract {
    BFP_RAW_NONE = 0,
    BFP_RAW_FORWARD,
    BFP_RAW_BACKWARD_DX,
    BFP_RAW_BACKWARD_ACC,
} bfpRawContract_t;

/* The one conversion funnel. Every op runs:
 *   prologue   — operands whose dtype != arithmetic are converted into
 *                transient stack scratch (sources are never mutated)
 *   kernel     — pure computation: operands (in arithmetic representation)
 *                -> raw intermediate (SYM kernels emit raw int32 mantissas,
 *                BFP kernels emit FLOAT32);
 *                also given auxOut (kernel-written verbatim, e.g. MaxPool's
 *                argmax indices) and ctx (kernel geometry/config), neither of
 *                which the funnel touches
 *   epilogue   — the intermediate is written/accumulated into the target,
 *                in the TARGET's own dtype. Persistent buffers are never
 *                pulled through the arithmetic. auxOut is NEVER funnel-
 *                converted — the kernel writes it in its own storage format.
 *                The OUT_WRITE requant rounds by the op's
 *                arithmetic.roundingMode; the ACC epilogues by the target's own.
 * Escape hatch policy: the prologue/epilogue helpers stay static in
 * ExecuteOp.c. Opening one for an op that does not fit the n-inputs/1-output
 * shape requires a documented exception here. */

/* rs: the op's KERNEL(0) stream (NULL without an op key). A kernel that rounds
 * hands it to its rounding calls; every other kernel ignores it. */
typedef void (*opKernelFn_t)(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                             tensor_t *auxOut, const void *ctx, rngStream_t *rs);

/* Caller guarantees nInputs <= this. Bump deliberately; opSpec_t.bfpStage is
 * sized by it. */
#define EXECUTE_OP_MAX_INPUTS 3

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

/*! @brief Descriptor for a funnel op invocation.
 * ctx: kernel geometry/config, opaque to the funnel, passed straight through
 *      to the kernel; NULL for config-free kernels.
 * auxOut: kernel-written verbatim, in ITS OWN storage format — never funnel-
 *         converted; NULL for single-output ops (e.g. MaxPool1d's argmax
 *         indices live here). */
typedef struct opSpec {
    opKernelFn_t kernel;
    const void *ctx;
    tensor_t **inputs;
    size_t nInputs;
    arithmetic_t arithmetic;
    /* The op key. The funnel opens one stream per phase from it: STAGE(i) for
     * staging operand i, KERNEL(0) for the kernel, OUT(0) for the OUT_WRITE
     * epilogue, ACC(j) for the j-th rounding pass of an ACC epilogue. NULL (the
     * zero-init) hands every phase a NULL stream. Read during the call only. */
    const rngKey_t *key;
    outputMode_t mode;
    tensor_t *auxOut;
    /* Opt-in for OUT_WRITE aliasing when the target is also an
     * input. true == the kernel reads element i of every input before writing
     * element i of the output (elementwise), so writing the target's storage
     * directly is safe. Zero-init (false) = conservative staging. Aliasing
     * additionally requires FLOAT32 arithmetic AND a FLOAT32 target — for any
     * other combination the epilogue's conversion/width-restore is load-
     * bearing and the funnel always stages. Alias detection is exact-base-
     * pointer equality; overlapping sub-views (raw-pointer tensor wiring) are
     * outside the contract. */
    bool writesInPlaceSafe;
    /* 0 = no grouped SYM/ASYM operand allowed; i+1 = inputs[i] (and only it) may
     * be grouped. Any other grouped operand fail-fasts. Declared by the GEMM-family
     * forward and dx weights and the optimizer param updates; grads, bias,
     * gamma/beta, wires and momentum stay per-tensor.
     *
     * ARITH_SYM_INT32: the declared operand is unpacked to signed mantissas (ASYM:
     * code - zp[g], the same image as SYM) and the scratch scale is poisoned to
     * 1.0f. Kernels must read per-group scales and the operand range from their
     * ctx (symGroupedWeight_t), never the scratch config. ARITH_FLOAT32: the
     * operand goes through the group-aware dequant; this field only gates it.
     * BFP-stored operands never use this field. */
    size_t groupedSymOperandPos;
    /* ARITH_BFP only: staging geometry for FLOAT32-stored operands, required for
     * each of them (caller guarantees). Entries are templates ({1,0} per-tensor
     * or {numGroups>1, groupSize>0} with numGroups*groupSize == the operand's
     * element count, plus mantissa/exponent widths); the
     * template's exponents pointer and roundingMode are IGNORED — the funnel
     * owns transient scratch exponent backing and stages with the OP's
     * arithmetic.roundingMode. NULL entry (zero-init) = the operand must
     * already be BFP-stored (its exponents are then borrowed zero-copy).
     * Ignored under other arithmetic types.
     *
     * Kernel-side contract under ARITH_BFP: every operand reaches the kernel
     * as UNPACKED int32 mantissa codes under a BFP-typed quantization — a
     * packed-storage byte count (calcNumberOfBytesForData) on such an
     * operand is wrong-by-design. Borrowed exponent arrays (BFP-stored
     * operands) are READ-ONLY: writing through them corrupts the source
     * tensor. Borrowed configs carry the source's STORAGE roundingMode;
     * staged configs carry the OP's arithmetic.roundingMode. */
    const bfpQConfig_t *bfpStage[EXECUTE_OP_MAX_INPUTS];
    /* ARITH_BFP only: the raw-format contract of the op (bfpRawContract_t
     * above). Zero-init (BFP_RAW_NONE) keeps the FLOAT32 raw. */
    bfpRawContract_t bfpRawOut;
} opSpec_t;

void executeOp(const opSpec_t *spec, tensor_t *target);

/* weightGradAccMode/biasGradAccMode have no "unset" value and OUT_WRITE is the
 * zero-init, so a config that forgot them would silently overwrite a grad. Call
 * at each grad executeOp site; `context` names the layer and field. */
static inline void executeOpValidateAccMode(outputMode_t mode, const char *context) {
    if (mode != OUT_ACC_DYNAMIC_RESCALE && mode != OUT_ACC_FIXED_SCALE) {
        PRINT_ERROR("%s: not a valid grad accumulate mode (got %d) -- config field never set?",
                    context, (int)mode);
        exit(1);
    }
}

/* Copies operand 0 into rawOut (data + SYM scale if applicable), for an
 * already-computed increment. Ignores auxOut/ctx. Incompatible with the unpacked-BFP scratch form
 * (its byte-count-based copy reads packed-storage sizes) — never pair with ARITH_BFP. */
void executeOpIdentityKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                             tensor_t *auxOut, const void *ctx, rngStream_t *rs);

/* Kernel-less funnel form: storage-to-storage conversion (1 input,
 * OUT_WRITE semantics). SYM->SYM routes through the conversionMatrix
 * diagonal (requant), never the same-type memmove. Supports every
 * populated conversionMatrix cell. Unlike executeOp's OUT_WRITE epilogue,
 * a bare conversion IS a storage encode: it rounds by the target's own
 * qConfig roundingMode. rs is handed to the cell's rounding calls. */
void executeConvert(tensor_t *input, tensor_t *target, rngStream_t *rs);

#endif // ENV5_RUNTIME_EXECUTE_OP_H
