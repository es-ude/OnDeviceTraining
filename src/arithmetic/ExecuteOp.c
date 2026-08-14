#define SOURCE_FILE "EXECUTE-OP"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Add.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Quantization.h"
#include "Rounding.h"
#include "Tensor.h"
#include "TensorConversion.h"

void executeOpIdentityKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                             tensor_t *auxOut, const void *ctx) {
    (void)nOperands;
    (void)auxOut;
    (void)ctx;
    tensor_t *src = operands[0];
    size_t n = calcNumberOfElementsByTensor(src);
    size_t bytes = calcNumberOfBytesForData(src->quantization, n);
    memcpy(rawOut->data, src->data, bytes);
    if (src->quantization->type == SYM_INT32) {
        ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale =
            ((symInt32QConfig_t *)src->quantization->qConfig)->scale;
    }
}

/* Phase 4, OUT_WRITE: intermediate -> target in the target's dtype. Same-dtype
 * SYM->SYM must REQUANT via the conversionMatrix diagonal — convertTensor's
 * same-type branch is a memmove that would pass raw mantissas through
 * unrestored (the QuantizationLayer.c trap). */
static void writeOutConversion(tensor_t *intermediate, tensor_t *target) {
    if (intermediate->quantization->type == SYM_INT32 && target->quantization->type == SYM_INT32) {
        conversionMatrix[SYM_INT32][SYM_INT32](intermediate, target);
        return;
    }
    /* Same trap class for BFP: the same-type branch either dies on the width
     * guard or copies stale exponents verbatim; the diagonal re-derives the
     * target's per-group exponents fresh on the target's geometry. */
    if (intermediate->quantization->type == BFP && target->quantization->type == BFP) {
        conversionMatrix[BFP][BFP](intermediate, target);
        return;
    }
    convertTensor(intermediate, target);
}

/* #282 seam: where a quantized tensor keeps its storage-requant rounding
 * mode. NULL for dtypes without one (FLOAT32/INT32/BOOL). */
static roundingMode_t *storageRoundingSlot(tensor_t *tensor) {
    switch (tensor->quantization->type) {
    case SYM_INT32:
        return &((symInt32QConfig_t *)tensor->quantization->qConfig)->roundingMode;
    case SYM:
        return &((symQConfig_t *)tensor->quantization->qConfig)->roundingMode;
    case ASYM:
        return &((asymQConfig_t *)tensor->quantization->qConfig)->roundingMode;
    case BFP:
        return &((bfpQConfig_t *)tensor->quantization->qConfig)->roundingMode;
    default:
        return NULL;
    }
}

/* OUT_WRITE epilogue (#282): the requant into a quantized target rounds by
 * the OPERATION's mode, injected by transiently swapping the target's storage
 * rounding slot around the conversion — the conversionMatrix signatures stay
 * untouched and the swap is restored before returning (the qConfig is
 * serialized into checkpoints and stays authoritative for storage/inference
 * encodes). The ACC epilogues (accumulateOut) deliberately keep the TARGET's
 * mode: accumulate is a read-modify-write under the accumulator's own storage
 * grid, whose rounding is part of the grid discipline like scale (spec D4). */
static void writeOut(tensor_t *intermediate, tensor_t *target, roundingMode_t opRounding) {
    roundingMode_t *slot = storageRoundingSlot(target);
    if (slot == NULL) {
        writeOutConversion(intermediate, target);
        return;
    }
    roundingMode_t storageMode = *slot;
    *slot = opRounding;
    writeOutConversion(intermediate, target);
    *slot = storageMode;
}

void executeConvert(tensor_t *input, tensor_t *target) {
    /* Bare storage-to-storage conversion: a conversion node's rounding IS a
     * storage encode, so the target's own qConfig roundingMode applies here —
     * unlike the OUT_WRITE epilogue, which rounds by the operation (#282). */
    writeOutConversion(input, target);
}

/* Phase 4, ACC modes. The SYM_INT32->SYM_INT32 add is Strategy A via
 * accumulateSymInt32IntoSymInt32Rescale (bit-identical to Linear.c's
 * weight-grad accumulate); the FLOAT32-intermediate -> SYM_INT32 arm first
 * quantizes the increment to operand width with the TARGET's roundingMode
 * (reproduces the former LayerNorm helper layerNormAccumulateGradSymInt32,
 * deleted in PR1b; semantics live here now). Fixed-scale reproduces the
 * former linearCalcBiasGradsSymInt32 behavior: rescale into the target's
 * EXISTING scale via rescaleIntoAccumulatorScale (spec D4 — honors the
 * TARGET's roundingMode; Conv1d.c:288 precedent), no clamp, scale never
 * re-derived. The packed SYM/ASYM arms (spec §4.1-4.2) stream the increment
 * chunk-wise via the tensor-typed accumulate*Into* entry points (#296 Stage
 * 2) instead of staging a whole-tensor float view; a FLOAT32 intermediate is
 * passed as a direct pointer (no view/VLA at all). */
static void accumulateOut(tensor_t *intermediate, tensor_t *target, outputMode_t mode) {
    size_t n = calcNumberOfElementsByTensor(target);

    switch (target->quantization->type) {
    case FLOAT32:
        accumulateTensorIntoFloat32Inplace(target, intermediate);
        return;
    case SYM_INT32: {
        symInt32QConfig_t *targetQC = target->quantization->qConfig;
        if (targetQC->qMaxBits > ODT_SYM_GRAD_QMAXBITS) {
            PRINT_ERROR("executeOp: SYM grad target qMaxBits (%u) exceeds grad contract (%u)",
                        (unsigned)targetQC->qMaxBits, (unsigned)ODT_SYM_GRAD_QMAXBITS);
            exit(1);
        }
        if (mode == OUT_ACC_FIXED_SCALE) {
            if (intermediate->quantization->type != SYM_INT32) {
                PRINT_ERROR("executeOp: OUT_ACC_FIXED_SCALE needs a SYM intermediate "
                            "for a SYM target (got dtype %d)",
                            (int)intermediate->quantization->type);
                exit(1);
            }
            float intermScale = ((symInt32QConfig_t *)intermediate->quantization->qConfig)->scale;
            float targetScale = targetQC->scale;
            int32_t *tg = (int32_t *)target->data;
            int32_t *in = (int32_t *)intermediate->data;
            for (size_t i = 0; i < n; i++) {
                tg[i] += rescaleIntoAccumulatorScale(in[i], intermScale, targetScale,
                                                     targetQC->roundingMode);
            }
            return;
        }
        /* OUT_ACC_DYNAMIC_RESCALE */
        if (intermediate->quantization->type == SYM_INT32) {
            accumulateSymInt32IntoSymInt32Rescale(target, intermediate);
            return;
        }
        /* #296 residual (spec §5): this arm quantizes the whole increment
         * before the add — two sequential rounding blocks. An O(chunk)
         * version would have to re-draw or reorder the SR stream (bit-parity
         * break), so it keeps whole-tensor staging until an RNG-state
         * snapshot/replay exists. Reached only by SYM_INT32-STORED grads,
         * which #261 already discourages. */
        /* FLOAT32 intermediate: quantize to operand width first, roundingMode
         * from the TARGET's qConfig (LayerNorm.c:446-463 reproduction). */
        symInt32QConfig_t incQC;
        initSymInt32QConfig(targetQC->roundingMode, &incQC);
        quantization_t incQ;
        initSymInt32Quantization(&incQC, &incQ);
        uint8_t incSymData[(n > 0 ? n : 1) * sizeof(int32_t)];
        tensor_t incSym;
        setTensorValuesForConversion(incSymData, &incQ, intermediate, &incSym);
        convertTensor(intermediate, &incSym);
        addSymInt32TensorsInplace(target, &incSym);
        return;
    }
    case SYM: {
        symQConfig_t *targetQC = target->quantization->qConfig;
        if (targetQC->qBits > ODT_SYM_GRAD_QMAXBITS) {
            PRINT_ERROR("executeOp: SYM grad target qBits (%u) exceeds grad contract (%u)",
                        (unsigned)targetQC->qBits, (unsigned)ODT_SYM_GRAD_QMAXBITS);
            exit(1);
        }
        if (intermediate->quantization->type == FLOAT32) {
            if (mode == OUT_ACC_FIXED_SCALE) {
                accumulateFloatIntoSymTensorFixedGrid(target, (const float *)intermediate->data, n);
            } else {
                accumulateFloatIntoSymTensorRescale(target, (const float *)intermediate->data, n);
            }
        } else {
            if (mode == OUT_ACC_FIXED_SCALE) {
                accumulateTensorIntoSymFixedGrid(target, intermediate);
            } else {
                accumulateTensorIntoSymRescale(target, intermediate);
            }
        }
        return;
    }
    case ASYM: {
        asymQConfig_t *targetQC = target->quantization->qConfig;
        if (targetQC->qBits > ODT_SYM_GRAD_QMAXBITS) {
            PRINT_ERROR("executeOp: ASYM grad target qBits (%u) exceeds grad contract (%u)",
                        (unsigned)targetQC->qBits, (unsigned)ODT_SYM_GRAD_QMAXBITS);
            exit(1);
        }
        if (mode == OUT_ACC_FIXED_SCALE) {
            PRINT_ERROR("executeOp: no fit-preserving ASYM pack — ASYM grad targets "
                        "accumulate under OUT_ACC_DYNAMIC_RESCALE only (PR3 spec, #261)");
            exit(1);
        }
        if (intermediate->quantization->type == FLOAT32) {
            accumulateFloatIntoAsymTensorRescale(target, (const float *)intermediate->data, n);
        } else {
            accumulateTensorIntoAsymRescale(target, intermediate);
        }
        return;
    }
    default:
        PRINT_ERROR("executeOp: accumulate target dtype %d not supported "
                    "(accepted: FLOAT32, SYM_INT32, SYM, ASYM; INT32/BOOL "
                    "remain unsupported)",
                    (int)target->quantization->type);
        exit(1);
    }
}

void executeOp(const opSpec_t *spec, tensor_t *target) {
    tensor_t **inputs = spec->inputs;
    size_t nInputs = spec->nInputs;
    arithmetic_t arithmetic = spec->arithmetic;

    if (nInputs > EXECUTE_OP_MAX_INPUTS) {
        PRINT_ERROR("executeOp: %u inputs exceeds EXECUTE_OP_MAX_INPUTS (%u)", (unsigned)nInputs,
                    (unsigned)EXECUTE_OP_MAX_INPUTS);
        exit(1);
    }

    /* Phase 1 — prologue: convert mismatched operands into stack scratch,
     * sized per actually-converted operand (#296 Stage 1). All-matching ops
     * (the whole FLOAT32 training path) allocate nothing. */
    size_t rowBytes[EXECUTE_OP_MAX_INPUTS] = {0};
    size_t totalScratchBytes = 0;
    size_t totalStageExponents = 0;
    for (size_t i = 0; i < nInputs; i++) {
        bool matches;
        switch (arithmetic.type) {
        case ARITH_FLOAT32:
            matches = inputs[i]->quantization->type == FLOAT32;
            break;
        case ARITH_SYM_INT32:
            matches = inputs[i]->quantization->type == SYM_INT32;
            break;
        case ARITH_BFP:
            /* Packed BFP storage is never kernel-usable in place — every
             * operand is unpacked/staged into int32 scratch (Task 3-5 kernels
             * consume the unpacked-BFP scratch form only). */
            matches = false;
            if (inputs[i]->quantization->type == FLOAT32 && spec->bfpStage[i] != NULL) {
                const bfpQConfig_t *stage = spec->bfpStage[i];
                size_t n = calcNumberOfElementsByTensor(inputs[i]);
                bool stagePerTensor = stage->numGroups == 1 && stage->groupSize == 0;
                bool stageGrouped = stage->numGroups > 1 && stage->groupSize > 0 &&
                                    stage->numGroups * stage->groupSize == n;
                if (!stagePerTensor && !stageGrouped) {
                    PRINT_ERROR("executeOp: bfpStage[%zu] template shape {numGroups=%zu, "
                                "groupSize=%zu} is invalid for %zu elements — valid shapes "
                                "are {1,0} (per-tensor) or {>1,>0} with "
                                "numGroups*groupSize == n",
                                i, stage->numGroups, stage->groupSize, n);
                    exit(1);
                }
                totalStageExponents += stage->numGroups;
            }
            break;
        default:
            PRINT_ERROR("executeOp: arithmetic dtype %d not supported (FLOAT32/SYM_INT32/BFP)",
                        (int)arithmetic.type);
            exit(1);
        }
        if (!matches) {
            size_t n = calcNumberOfElementsByTensor(inputs[i]);
            rowBytes[i] = (n > 0 ? n : 1) * sizeof(int32_t);
            totalScratchBytes += rowBytes[i];
        }
    }
    uint8_t scratch[totalScratchBytes > 0 ? totalScratchBytes : 1];
    /* ARITH_BFP staging: funnel-owned exponent backing for FLOAT32-stored
     * operands, sliced per staged operand (numGroups entries each). BFP-stored
     * operands never touch this — their exponents are borrowed in place. */
    uint8_t stageExponents[totalStageExponents > 0 ? totalStageExponents : 1];
    size_t stageOffset = 0;
    tensor_t scratchTensors[EXECUTE_OP_MAX_INPUTS];
    quantization_t scratchQ[EXECUTE_OP_MAX_INPUTS];
    symInt32QConfig_t scratchQC[EXECUTE_OP_MAX_INPUTS];
    bfpQConfig_t scratchBfpQC[EXECUTE_OP_MAX_INPUTS];
    tensor_t *ops[EXECUTE_OP_MAX_INPUTS];

    size_t scratchOffset = 0;
    for (size_t i = 0; i < nInputs; i++) {
        if (rowBytes[i] == 0) {
            ops[i] = inputs[i];
            continue;
        }

        /* Group-quant PR2 (Task 3; final-review Fix 2/3) + PR4 (Task 3): a
         * grouped operand (numGroups > 1) -- SYM or ASYM, the two grouped
         * carrier dtypes share the {numGroups, groupSize} shape grammar (D6)
         * -- has no scalar compute image under EITHER arithmetic type: the
         * SYM->SYM_INT32 and ASYM->SYM_INT32 conversionMatrix cells
         * fail-fast on grouped sources (PR2 Task 2 / PR4 Task 1), and the
         * group-aware SYM->FLOAT32 / ASYM->FLOAT32 cells are only meant to
         * be reachable from declared carrier positions (GEMM-family
         * forward/dx weights, optimizer param updates), not e.g. an
         * arbitrary non-carrier operand slot. Gate BOTH arms here, before
         * either arm's convertTensor call, on the declared position alone --
         * a grouped operand at any other position (or when nothing is
         * declared, groupedSymOperandPos == 0) fail-fasts. */
        symQConfig_t *symQC = (inputs[i]->quantization->type == SYM)
                                  ? (symQConfig_t *)inputs[i]->quantization->qConfig
                                  : NULL;
        asymQConfig_t *asymQC = (inputs[i]->quantization->type == ASYM)
                                    ? (asymQConfig_t *)inputs[i]->quantization->qConfig
                                    : NULL;
        size_t operandNumGroups =
            symQC != NULL ? symQC->numGroups : (asymQC != NULL ? asymQC->numGroups : 1);
        bool grouped = operandNumGroups > 1;
        if (grouped && spec->groupedSymOperandPos != i + 1) {
            PRINT_ERROR(
                "executeOp: grouped %s operand (numGroups=%zu) at inputs[%zu] reached an op "
                "without a matching groupedSymOperandPos declaration — grouped tensors are "
                "legal only where an op declares them (GEMM-family forward/dx weights, "
                "optimizer param updates); everything else is a non-carrier (spec §3)",
                symQC != NULL ? "SYM" : "ASYM", operandNumGroups, i);
            exit(1);
        }

        switch (arithmetic.type) {
        case ARITH_FLOAT32:
            initFloat32Quantization(&scratchQ[i]);
            setTensorValuesForConversion(&scratch[scratchOffset], &scratchQ[i], inputs[i],
                                         &scratchTensors[i]);
            convertTensor(inputs[i], &scratchTensors[i]);
            break;
        case ARITH_SYM_INT32: {
            /* Decision 11 (BFP epic PR2): deny BEFORE the convertTensor route
             * — the [BFP][SYM_INT32] cell would silently collapse the
             * operand's group structure to a single scalar grid. */
            if (inputs[i]->quantization->type == BFP) {
                PRINT_ERROR("executeOp: BFP-stored operand %zu under ARITH_SYM_INT32 would "
                            "silently collapse its group structure to a scalar grid — use "
                            "ARITH_FLOAT32 (fake-quant) or ARITH_BFP (native), or convert "
                            "explicitly via a Quantization layer",
                            i);
                exit(1);
            }
            initSymInt32QConfig(arithmetic.roundingMode, &scratchQC[i]);
            initSymInt32Quantization(&scratchQC[i], &scratchQ[i]);
            setTensorValuesForConversion(&scratch[scratchOffset], &scratchQ[i], inputs[i],
                                         &scratchTensors[i]);

            if (grouped) {
                size_t n = calcNumberOfElementsByTensor(inputs[i]);
                if (symQC != NULL) {
                    unpackSignExtend(inputs[i]->data, symQC->qBits, 0,
                                     (int32_t *)scratchTensors[i].data, n);
                    scratchQC[i].qMaxBits = symQC->qBits;
                } else {
                    /* PR4 (Task 3), grouped ASYM: zero-extend the packed
                     * codes (byteConversion widen -- ASYM codes carry no
                     * sign bit), then shift each element into the
                     * signed-mantissa domain by ITS group's code-domain
                     * zeroPoint: mantissa = code - zp[g], g = i/groupSize
                     * (exact int32 subtract, both operands <= 2^16-1, D6).
                     * After the shift the scratch is the SAME mantissa image
                     * the SYM arm produces -- the group-aware kernel then
                     * applies per-group scales from its own ctx identically
                     * for both dtypes (D5: the grouped ASYM compute path IS
                     * the grouped SYM path on shifted mantissas). The zp is
                     * hoisted per run (one i/groupSize division per group,
                     * never per element); numGroups*groupSize == n by the
                     * attach-time shape validation. */
                    int32_t *mant = (int32_t *)scratchTensors[i].data;
                    byteConversion(inputs[i]->data, asymQC->qBits, (uint8_t *)mant, 32, n);
                    size_t idx = 0;
                    while (idx < n) {
                        size_t g = idx / asymQC->groupSize;
                        size_t runEnd = (g + 1) * asymQC->groupSize;
                        const int32_t zp = (int32_t)asymQC->zeroPoints[g];
                        for (; idx < runEnd; idx++) {
                            mant[idx] -= zp;
                        }
                    }
                    scratchQC[i].qMaxBits = asymQC->qBits;
                }
                /* Poison: a grouped operand has no single scalar scale — the
                 * group-aware kernel MUST read per-group scales via its own
                 * ctx (e.g. Matmul's weightGroups), never this field. */
                scratchQC[i].scale = 1.0f;
            } else {
                convertTensor(inputs[i], &scratchTensors[i]);
            }
            break;
        }
        case ARITH_BFP: {
            qtype_t stored = inputs[i]->quantization->type;
            size_t n = calcNumberOfElementsByTensor(inputs[i]);
            if (stored == BFP) {
                bfpQConfig_t *srcQC = inputs[i]->quantization->qConfig;
                /* Borrow: the struct copy ALIASES the source's exponents
                 * pointer (zero-copy); the prologue never writes borrowed
                 * exponents. Geometry/widths ride along for the kernel's
                 * shape/headroom checks. */
                scratchBfpQC[i] = *srcQC;
                initBfpQuantization(&scratchBfpQC[i], &scratchQ[i]);
                setTensorValuesForConversion(&scratch[scratchOffset], &scratchQ[i], inputs[i],
                                             &scratchTensors[i]);
                unpackSignExtend(inputs[i]->data, srcQC->mantissaBits, 0,
                                 (int32_t *)scratchTensors[i].data, n);
            } else if (stored == FLOAT32) {
                const bfpQConfig_t *stage = spec->bfpStage[i];
                if (stage == NULL) {
                    PRINT_ERROR("executeOp: FLOAT32-stored operand %zu under ARITH_BFP needs "
                                "a bfpStage geometry template (layer bug)",
                                i);
                    exit(1);
                }
                /* Geometry/widths from the template; exponent backing is the
                 * funnel's transient slice and the roundingMode is the OP's —
                 * staging rounds by the operation (#282 discipline), never by
                 * the template. */
                scratchBfpQC[i] = (bfpQConfig_t){.exponents = &stageExponents[stageOffset],
                                                 .numGroups = stage->numGroups,
                                                 .groupSize = stage->groupSize,
                                                 .roundingMode = arithmetic.roundingMode,
                                                 .mantissaBits = stage->mantissaBits,
                                                 .exponentBits = stage->exponentBits};
                stageOffset += stage->numGroups;
                initBfpQuantization(&scratchBfpQC[i], &scratchQ[i]);
                setTensorValuesForConversion(&scratch[scratchOffset], &scratchQ[i], inputs[i],
                                             &scratchTensors[i]);
                quantizeFloatBufferToBfpCodes((const float *)inputs[i]->data, n, &scratchBfpQC[i],
                                              (int32_t *)scratchTensors[i].data);
            } else {
                PRINT_ERROR("executeOp: operand %zu storage dtype %d unsupported under "
                            "ARITH_BFP (v1 accepts FLOAT32 or BFP; convert explicitly via a "
                            "Quantization layer)",
                            i, (int)stored);
                exit(1);
            }
            break;
        }
        }
        ops[i] = &scratchTensors[i];
        scratchOffset += rowBytes[i];
    }

    /* Phase 2 — intermediate in the arithmetic representation, target shape.
     * FLOAT32->FLOAT32 OUT_WRITE needs no epilogue conversion, so the kernel
     * may emit straight into the target ("aliasing") unless the target is
     * also a live operand and the kernel did not declare elementwise safety
     * (#296 Stage 1). */
    size_t outElems = calcNumberOfElementsByTensor(target);
    bool aliasOut = spec->mode == OUT_WRITE && arithmetic.type == ARITH_FLOAT32 &&
                    target->quantization->type == FLOAT32;
    if (aliasOut && !spec->writesInPlaceSafe) {
        for (size_t i = 0; i < nInputs; i++) {
            if (ops[i]->data == target->data) {
                aliasOut = false;
                break;
            }
        }
    }
    /* outElems == 0 with aliasing: raw.data may be NULL for N=0 tensors
     * (target->data straight from a calloc(1,0), #160) — safe because the
     * kernel below iterates 0 elements. */
    uint8_t rawData[(aliasOut || outElems == 0 ? 1 : outElems) * sizeof(int32_t)];
    tensor_t raw;
    quantization_t rawQ;
    symInt32QConfig_t rawQC;
    switch (arithmetic.type) {
    case ARITH_FLOAT32:
        initFloat32Quantization(&rawQ);
        break;
    case ARITH_SYM_INT32:
        initSymInt32QConfig(arithmetic.roundingMode, &rawQC);
        initSymInt32Quantization(&rawQC, &rawQ);
        break;
    case ARITH_BFP:
        /* Spec D7: BFP kernels fold same-exponent segments into a float
         * accumulator (ldexpf) and never round — the raw intermediate stays
         * FLOAT32; any width-restore/pack is the OUT_WRITE epilogue's job. */
        initFloat32Quantization(&rawQ);
        break;
    default:
        PRINT_ERROR("executeOp: arithmetic dtype %d not supported (FLOAT32/SYM_INT32/BFP)",
                    (int)arithmetic.type);
        exit(1);
    }
    setTensorValues(&raw, aliasOut ? target->data : rawData, target->shape, &rawQ,
                    target->sparsity);

    /* Phase 3 — kernel emits raw; auxOut/ctx pass through untouched by the
     * funnel (auxOut is NEVER funnel-converted). */
    spec->kernel(ops, nInputs, &raw, spec->auxOut, spec->ctx);

    /* Phase 4 — epilogue (target only; auxOut already final). */
    switch (spec->mode) {
    case OUT_WRITE:
        if (!aliasOut) {
            writeOut(&raw, target, arithmetic.roundingMode);
        }
        break;
    case OUT_ACC_DYNAMIC_RESCALE:
    case OUT_ACC_FIXED_SCALE:
        accumulateOut(&raw, target, spec->mode);
        break;
    }
}
