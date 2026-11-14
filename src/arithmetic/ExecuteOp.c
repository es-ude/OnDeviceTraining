#define SOURCE_FILE "EXECUTE-OP"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Add.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Quantization.h"
#include "RNG.h"
#include "Rounding.h"
#include "ScaleCanon.h"
#include "Tensor.h"
#include "TensorConversion.h"

void executeOpIdentityKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                             tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)nOperands;
    (void)auxOut;
    (void)ctx;
    tensor_t *src = operands[0];
    size_t n = calcNumberOfElementsByTensor(src);
    size_t bytes = calcNumberOfBytesForData(src->quantization, n);
    memcpy(rawOut->data, src->data, bytes);
    if (src->quantization->type == SYM_INT32) {
        storeCanonScale(&((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale,
                        loadCanonScale(&((symInt32QConfig_t *)src->quantization->qConfig)->scale));
    }
}

/* Raw INT32 codes are kept at scale 1 and declared 32 bits wide, so the
 * accumulator guard's proof covers the full int32 range. This scratch is an
 * operand or a requant source, never a requant target. */
static void stageInt32Raw(const tensor_t *src, tensor_t *scratch, symInt32QConfig_t *scratchQC) {
    scratchQC->qMaxBits = 32;
    storeCanonScale(&scratchQC->scale, 1.0f); /* raw-int: exactly scale one */
    memcpy(scratch->data, src->data,
           calcNumberOfElementsByTensor((tensor_t *)src) * sizeof(int32_t));
}

/* One phase's stream: NULL without an op key, otherwise the stream of
 * rngFold(*key, tag, index), opened into the caller's storage. */
static rngStream_t *openPhaseStream(const rngKey_t *key, uint32_t tag, uint32_t index,
                                    rngStream_t *storage) {
    if (key == NULL) {
        return NULL;
    }
    *storage = rngOpen(rngFold(*key, tag, index));
    return storage;
}

/* Phase 4, OUT_WRITE: intermediate -> target in the target's dtype. Same-dtype
 * SYM->SYM must REQUANT via the conversionMatrix diagonal — convertTensor's
 * same-type branch is a memmove that would pass raw mantissas through unrestored. */
static void writeOutConversion(tensor_t *intermediate, tensor_t *target, rngStream_t *rs) {
    if (intermediate->quantization->type == SYM_INT32 && target->quantization->type == SYM_INT32) {
        conversionMatrix[SYM_INT32][SYM_INT32](intermediate, target, rs);
        return;
    }
    /* Same trap class for BFP: the same-type branch either dies on the width
     * guard or copies stale exponents verbatim; the diagonal re-derives the
     * target's per-group exponents fresh on the target's geometry. */
    if (intermediate->quantization->type == BFP && target->quantization->type == BFP) {
        conversionMatrix[BFP][BFP](intermediate, target, rs);
        return;
    }
    convertTensor(intermediate, target, rs);
}

/* Where a quantized tensor keeps its storage-requant rounding
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

/* OUT_WRITE rounds by the operation's mode, swapped into the target's storage
 * rounding slot for the conversion and restored afterwards: the qConfig is
 * serialized into checkpoints and stays authoritative for storage encodes. The ACC
 * epilogues keep the target's own mode, as part of its grid like the scale. */
static void writeOut(tensor_t *intermediate, tensor_t *target, roundingMode_t opRounding,
                     rngStream_t *rs) {
    roundingMode_t *slot = storageRoundingSlot(target);
    if (slot == NULL) {
        writeOutConversion(intermediate, target, rs);
        return;
    }
    roundingMode_t storageMode = *slot;
    *slot = opRounding;
    writeOutConversion(intermediate, target, rs);
    *slot = storageMode;
}

void executeConvert(tensor_t *input, tensor_t *target, rngStream_t *rs) {
    /* A bare conversion is a storage encode: the target's own rounding applies. */
    writeOutConversion(input, target, rs);
}

/* Phase 4, ACC modes. Accumulation rounds on the target grid with the target's
 * mode. Dynamic rescale derives a fresh grid; fixed scale keeps the established
 * one (no clamp, scale never re-derived). A FLOAT32 increment into SYM_INT32 is
 * quantized to operand width before the add. Packed SYM/ASYM/BFP targets stream
 * the increment chunk-wise. */
static void accumulateOut(tensor_t *intermediate, tensor_t *target, outputMode_t mode,
                          const rngKey_t *key) {
    size_t n = calcNumberOfElementsByTensor(target);
    rngStream_t acc0Storage;
    rngStream_t *acc0 = openPhaseStream(key, RNG_TAG_ACC, 0u, &acc0Storage);

    switch (target->quantization->type) {
    case FLOAT32:
        accumulateTensorIntoFloat32Inplace(target, intermediate, acc0);
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
            float intermScale =
                loadCanonScale(&((symInt32QConfig_t *)intermediate->quantization->qConfig)->scale);
            float targetScale = loadCanonScale(&targetQC->scale);
            int32_t *tg = (int32_t *)target->data;
            int32_t *in = (int32_t *)intermediate->data;
            for (size_t i = 0; i < n; i++) {
                tg[i] += rescaleIntoAccumulatorScale(in[i], intermScale, targetScale,
                                                     targetQC->roundingMode, acc0);
            }
            return;
        }
        /* OUT_ACC_DYNAMIC_RESCALE */
        if (intermediate->quantization->type == SYM_INT32) {
            accumulateSymInt32IntoSymInt32Rescale(target, intermediate, acc0);
            return;
        }
        /* FLOAT32 intermediate: quantize with ACC 0, then add/requantize with ACC 1;
         * each pass draws its stream in element order. */
        symInt32QConfig_t incQC;
        initSymInt32QConfig(targetQC->roundingMode, &incQC);
        quantization_t incQ;
        initSymInt32Quantization(&incQC, &incQ);
        uint8_t incSymData[(n > 0 ? n : 1) * sizeof(int32_t)];
        tensor_t incSym;
        setTensorValuesForConversion(incSymData, &incQ, intermediate, &incSym);
        convertTensor(intermediate, &incSym, acc0);
        rngStream_t acc1Storage;
        addSymInt32TensorsInplace(target, &incSym,
                                  openPhaseStream(key, RNG_TAG_ACC, 1u, &acc1Storage));
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
                accumulateFloatIntoSymTensorFixedGrid(target, (const float *)intermediate->data, n,
                                                      acc0);
            } else {
                accumulateFloatIntoSymTensorRescale(target, (const float *)intermediate->data, n,
                                                    acc0);
            }
        } else {
            if (mode == OUT_ACC_FIXED_SCALE) {
                accumulateTensorIntoSymFixedGrid(target, intermediate, acc0);
            } else {
                accumulateTensorIntoSymRescale(target, intermediate, acc0);
            }
        }
        return;
    }
    case ASYM: {
        if (mode == OUT_ACC_FIXED_SCALE) {
            PRINT_ERROR("executeOp: no fit-preserving ASYM pack — ASYM grad targets "
                        "accumulate under OUT_ACC_DYNAMIC_RESCALE only");
            exit(1);
        }
        if (intermediate->quantization->type == FLOAT32) {
            accumulateFloatIntoAsymTensorRescale(target, (const float *)intermediate->data, n,
                                                 acc0);
        } else {
            accumulateTensorIntoAsymRescale(target, intermediate, acc0);
        }
        return;
    }
    case BFP: {
        /* FIXED_SCALE carries the per-group exponents and aborts on
         * mantissa overflow; DYNAMIC_RESCALE re-derives them per store. */
        if (intermediate->quantization->type == BFP_RAW) {
            /* BFP_RAW_BACKWARD_ACC (validated DYNAMIC_RESCALE before the
             * kernel): the integer engine, no float. */
            accumulateBfpRawIntoBfpRescale(target, intermediate, acc0);
            return;
        }
        if (intermediate->quantization->type == FLOAT32) {
            if (mode == OUT_ACC_FIXED_SCALE) {
                accumulateFloatIntoBfpTensorFixedGrid(target, (const float *)intermediate->data, n,
                                                      acc0);
            } else {
                accumulateFloatIntoBfpTensorRescale(target, (const float *)intermediate->data, n,
                                                    acc0);
            }
        } else {
            if (mode == OUT_ACC_FIXED_SCALE) {
                accumulateTensorIntoBfpFixedGrid(target, intermediate, acc0);
            } else {
                accumulateTensorIntoBfpRescale(target, intermediate, acc0);
            }
        }
        return;
    }
    default:
        PRINT_ERROR("executeOp: accumulate target dtype %d not supported "
                    "(accepted: FLOAT32, SYM_INT32, SYM, ASYM, BFP; INT32/BOOL "
                    "remain unsupported)",
                    (int)target->quantization->type);
        exit(1);
    }
}

void executeOp(const opSpec_t *spec, tensor_t *target) {
    tensor_t **inputs = spec->inputs;
    size_t nInputs = spec->nInputs;
    arithmetic_t arithmetic = spec->arithmetic;

    /* Phase 1 — prologue: convert mismatched operands into stack scratch,
     * sized per actually-converted operand. All-matching ops
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
            /* Packed BFP is never kernel-usable in place: every operand is unpacked
             * or staged into int32 scratch. */
            matches = false;
            /* The integer forward into a BFP wire is float-free after the
             * model's input quantization layer: a FLOAT32 operand here is a
             * wiring bug, not a staging request. Arm A (FLOAT32 wires) and
             * every backward op (BFP_RAW_BACKWARD_DX / _ACC included) keep staging. */
            if (inputs[i]->quantization->type == FLOAT32 && spec->bfpRawOut == BFP_RAW_FORWARD &&
                target->quantization->type == BFP) {
                PRINT_ERROR("executeOp: FLOAT32-stored operand %zu in an integer BFP forward "
                            "into a BFP wire -- the forward must see BFP operands only; "
                            "quantize the model input with a Quantization layer "
                            "(docs/conventions/arithmetic-bfp.md)",
                            i);
                exit(1);
            }
            if (inputs[i]->quantization->type == FLOAT32 && spec->bfpStage[i] != NULL) {
                totalStageExponents += spec->bfpStage[i]->numGroups;
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
        rngStream_t stageStreamStorage;
        rngStream_t *stageStream =
            openPhaseStream(spec->key, RNG_TAG_STAGE, (uint32_t)i, &stageStreamStorage);

        /* A grouped SYM/ASYM operand has no scalar compute image, so it is legal
         * only at the position the op declares (groupedSymOperandPos). */
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
                "optimizer param updates)",
                symQC != NULL ? "SYM" : "ASYM", operandNumGroups, i);
            exit(1);
        }

        switch (arithmetic.type) {
        case ARITH_FLOAT32:
            initFloat32Quantization(&scratchQ[i]);
            setTensorValuesForConversion(&scratch[scratchOffset], &scratchQ[i], inputs[i],
                                         &scratchTensors[i]);
            convertTensor(inputs[i], &scratchTensors[i], stageStream);
            break;
        case ARITH_SYM_INT32: {
            /* The [BFP][SYM_INT32] cell would collapse the group structure to one
             * scalar grid, so refuse before converting. */
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
                    /* Grouped ASYM: zero-extend the codes, then subtract each
                     * group's zero point, giving the same signed-mantissa image the
                     * SYM arm produces; the kernel applies per-group scales from its
                     * ctx. numGroups*groupSize == n by attach-time validation. */
                    int32_t *mant = (int32_t *)scratchTensors[i].data;
                    unpackZeroExtend(inputs[i]->data, asymQC->qBits, 0, mant, n);
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
                odtScaleStore(&scratchQC[i].scale, ODT_SCALE_ONE);
            } else if (inputs[i]->quantization->type == INT32) {
                stageInt32Raw(inputs[i], &scratchTensors[i], &scratchQC[i]);
            } else {
                convertTensor(inputs[i], &scratchTensors[i], stageStream);
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
                /* Geometry/widths from the caller's stage template (required for
                 * every FLOAT32 operand); exponent backing is the funnel's slice
                 * and staging rounds by the operation, never the template. */
                const bfpQConfig_t *stage = spec->bfpStage[i];
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
                                              (int32_t *)scratchTensors[i].data, stageStream);
            } else {
                PRINT_ERROR("executeOp: operand %zu storage dtype %d unsupported under "
                            "ARITH_BFP (accepts FLOAT32 or BFP; convert explicitly via a "
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
     * also a live operand and the kernel did not declare elementwise safety. */
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
    /* Integer raw (opSpec_t.bfpRawOut): geometry copied from the target --
     * the raw exponent is per TARGET group by contract -- and exponent
     * backing on the funnel's stack, like stageExponents. Validated before
     * the VLAs so a misuse dies with a message, not a bad size. */
    bfpRawQConfig_t rawBfpQC = {.exponents = NULL, .numGroups = 0, .groupSize = 0};
    if (spec->bfpRawOut != BFP_RAW_NONE) {
        if (spec->bfpRawOut != BFP_RAW_FORWARD && spec->bfpRawOut != BFP_RAW_BACKWARD_DX &&
            spec->bfpRawOut != BFP_RAW_BACKWARD_ACC) {
            PRINT_ERROR("executeOp: unknown bfpRawOut contract %d", (int)spec->bfpRawOut);
            exit(1);
        }
        if (arithmetic.type != ARITH_BFP) {
            PRINT_ERROR("executeOp: bfpRawOut is an ARITH_BFP contract (arithmetic "
                        "dtype %d) -- a fake-quant pin must not set it",
                        (int)arithmetic.type);
            exit(1);
        }
        if (spec->bfpRawOut == BFP_RAW_BACKWARD_ACC) {
            /* The grad contract: the integer engine is the DYNAMIC_RESCALE
             * arm into BFP storage; the fixed grid and FLOAT32 grads keep
             * the FLOAT32 raw (the layer selects NONE there). */
            if (spec->mode != OUT_ACC_DYNAMIC_RESCALE) {
                PRINT_ERROR("executeOp: BFP_RAW_BACKWARD_ACC needs OUT_ACC_DYNAMIC_RESCALE "
                            "(mode %d) -- the fixed grid keeps the FLOAT32 raw",
                            (int)spec->mode);
                exit(1);
            }
            if (target->quantization->type != BFP) {
                PRINT_ERROR("executeOp: BFP_RAW_BACKWARD_ACC accumulates into BFP storage "
                            "only (target dtype %d) -- a FLOAT32 grad keeps the FLOAT32 raw",
                            (int)target->quantization->type);
                exit(1);
            }
        } else if (spec->mode != OUT_WRITE) {
            PRINT_ERROR("executeOp: bfpRawOut needs the OUT_WRITE epilogue (mode %d) -- "
                        "accumulating ops use BFP_RAW_BACKWARD_ACC or the FLOAT32 raw",
                        (int)spec->mode);
            exit(1);
        }
        switch (target->quantization->type) {
        case BFP: {
            const bfpQConfig_t *targetQC = target->quantization->qConfig;
            rawBfpQC.numGroups = targetQC->numGroups;
            rawBfpQC.groupSize = targetQC->groupSize;
            break;
        }
        case FLOAT32:
            rawBfpQC.numGroups = 1;
            rawBfpQC.groupSize = 0;
            break;
        default:
            PRINT_ERROR("executeOp: bfpRawOut has no epilogue into target dtype %d (BFP re-block "
                        "or FLOAT32 dequant only)",
                        (int)target->quantization->type);
            exit(1);
        }
    }
    int16_t rawExponents[rawBfpQC.numGroups > 0 ? rawBfpQC.numGroups : 1];
    /* An empty aliased target may have NULL data; the kernel iterates 0 elements. */
    uint8_t rawData[(aliasOut || outElems == 0 ? 1 : outElems) * sizeof(int32_t)];
    tensor_t raw;
    quantization_t rawQ;
    symInt32QConfig_t rawQC;
    switch (arithmetic.type) {
    case ARITH_FLOAT32:
        initFloat32Quantization(&rawQ);
        break;
    case ARITH_SYM_INT32:
        /* The kernel overwrites rawQC's scale; the public raw producers also
         * its qMaxBits (static result width, up to 32), the file-static
         * Conv1d / ConvT grad kernels do not. The epilogue reads only the
         * target's width. */
        initSymInt32QConfig(arithmetic.roundingMode, &rawQC);
        initSymInt32Quantization(&rawQC, &rawQ);
        break;
    case ARITH_BFP:
        if (spec->bfpRawOut != BFP_RAW_NONE) {
            for (size_t g = 0; g < rawBfpQC.numGroups; g++) {
                rawExponents[g] = 0;
            }
            rawBfpQC.exponents = rawExponents;
            initBfpRawQuantization(&rawBfpQC, &rawQ);
            break;
        }
        /* Without the raw opt-in, BFP kernels accumulate in float and never round:
         * the raw intermediate is FLOAT32 and the epilogue packs. */
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
    rngStream_t kernelStorage;
    spec->kernel(ops, nInputs, &raw, spec->auxOut, spec->ctx,
                 openPhaseStream(spec->key, RNG_TAG_KERNEL, 0u, &kernelStorage));

    /* Phase 4 — epilogue (target only; auxOut already final). */
    switch (spec->mode) {
    case OUT_WRITE:
        if (!aliasOut) {
            rngStream_t outStorage;
            writeOut(&raw, target, arithmetic.roundingMode,
                     openPhaseStream(spec->key, RNG_TAG_OUT, 0u, &outStorage));
        }
        break;
    case OUT_ACC_DYNAMIC_RESCALE:
    case OUT_ACC_FIXED_SCALE:
        accumulateOut(&raw, target, spec->mode, spec->key);
        break;
    }
}
