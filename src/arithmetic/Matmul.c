#define SOURCE_FILE "MATMUL"

#ifdef TRACK_INSTRUCTIONS
#define MATMUL_FUNC_INT matmulIntTensorsWithInstructionCounter
#define MATMUL_FUNC_FLOAT matmulFloatTensorsWithInstructionCounter
#define MATMUL_FUNC_SYM_INT32 matmulSymIntTensorsWithInstructionCounter
#else
#define MATMUL_FUNC_INT matmulIntTensors
#define MATMUL_FUNC_FLOAT matmulFloatTensors
#define MATMUL_FUNC_SYM_INT32 matmulSymIntTensors
#endif

#include <stdio.h>
#include <stdlib.h>

#include "Arithmetic.h"
#include "Common.h"
#include "DTypes.h"
#include "Matmul.h"
#include "Mul.h"
#include "Rounding.h"
#include "Tensor.h"

size_t matmulInstructionCounter = 0;

static void matmulIntCore(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                          const int32_t *biasSeed) {
    if (aTensor->shape->numberOfDimensions > 2 || bTensor->shape->numberOfDimensions > 2) {
        PRINT_ERROR("Matmul only supports up to 2D Tensors");
        exit(1);
    }

    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;
    size_t bNumberOfDims = bTensor->shape->numberOfDimensions;
    size_t *bDims = bTensor->shape->dimensions;

    size_t aRows, aColumns;
    if (aNumberOfDims < 2) {
        aRows = 1;
        aColumns = getDimensionsByIndex(aTensor, 0);
    } else {
        aRows = getDimensionsByIndex(aTensor, 0);
        aColumns = getDimensionsByIndex(aTensor, 1);
    }

    size_t bRows = getDimensionsByIndex(bTensor, 0);
    size_t bColumns = (bNumberOfDims < 2) ? 1 : getDimensionsByIndex(bTensor, 1);

    size_t resultCounter = 0;

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %lu, bRows: %lu\n", aColumns, bRows);
        exit(1);
    }

    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            int32_t result = biasSeed ? biasSeed[columnIndex] : 0;
            for (size_t i = 0; i < aColumns; i++) {
                size_t aByteIndex = 0;
                if (aNumberOfDims == 1) {
                    aByteIndex = i * sizeof(int32_t);
                } else {
                    size_t aIndices[] = {rowIndex, i};
                    size_t aValueIndex = calcElementIndexByIndices(
                        aNumberOfDims, aDims, aIndices, aTensor->shape->orderOfDimensions);
                    aByteIndex = aValueIndex * sizeof(int32_t);
                }
                int32_t aValue = readBytesAsInt32(&aTensor->data[aByteIndex]);

                size_t bByteIndex = 0;
                if (bNumberOfDims == 1) {
                    bByteIndex = i * sizeof(int32_t);
                } else {
                    size_t bIndices[] = {i, columnIndex};
                    size_t bValueIndex = calcElementIndexByIndices(
                        bNumberOfDims, bDims, bIndices, bTensor->shape->orderOfDimensions);
                    bByteIndex = bValueIndex * sizeof(int32_t);
                }
                int32_t bValue = readBytesAsInt32(&bTensor->data[bByteIndex]);

                result += mulInt32s(aValue, bValue);
            }

            size_t outputByteIndex = resultCounter * sizeof(int32_t);
            writeInt32ToByteArray(result, &outputTensor->data[outputByteIndex]);
            resultCounter++;
        }
    }
}

void matmulIntTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulIntCore(aTensor, bTensor, outputTensor, NULL);
}

void matmulIntTensorsWithInstructionCounter(tensor_t *aTensor, tensor_t *bTensor,
                                            tensor_t *outputTensor) {
    matmulIntCore(aTensor, bTensor, outputTensor, NULL);
    ++matmulInstructionCounter;
}

void matmulInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    MATMUL_FUNC_INT(aTensor, bTensor, outputTensor);
}

static void matmulFloatCore(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                            const uint8_t *biasSeed) {
    if (aTensor->shape->numberOfDimensions > 2 || bTensor->shape->numberOfDimensions > 2) {
        PRINT_ERROR("Matmul only supports up to 2D Tensors");
        exit(1);
    }

    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;
    size_t bNumberOfDims = bTensor->shape->numberOfDimensions;
    size_t *bDims = bTensor->shape->dimensions;

    size_t aRows, aColumns = 0;
    if (aNumberOfDims < 2) {
        aRows = 1;
        aColumns = getDimensionsByIndex(aTensor, 0);
    } else {
        aRows = getDimensionsByIndex(aTensor, 0);
        aColumns = getDimensionsByIndex(aTensor, 1);
    }

    size_t bRows, bColumns = 0;
    if (bNumberOfDims < 2) {
        bRows = getDimensionsByIndex(bTensor, 0);
        bColumns = 1;
    } else {
        bRows = getDimensionsByIndex(bTensor, 0);
        bColumns = getDimensionsByIndex(bTensor, 1);
    }

    size_t resultCounter = 0;

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %lu, bRows: %lu\n", aColumns, bRows);
        exit(1);
    }

    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            float result = biasSeed
                               ? readBytesAsFloat((uint8_t *)&biasSeed[columnIndex * sizeof(float)])
                               : 0.0f;
            for (size_t i = 0; i < aColumns; i++) {
                size_t aByteIndex = 0;
                if (aNumberOfDims == 1) {
                    aByteIndex = i * sizeof(float);
                } else {
                    size_t aIndices[] = {rowIndex, i};
                    size_t aValueIndex = calcElementIndexByIndices(
                        aNumberOfDims, aDims, aIndices, aTensor->shape->orderOfDimensions);
                    aByteIndex = aValueIndex * sizeof(float);
                }
                float aValue = readBytesAsFloat(&aTensor->data[aByteIndex]);

                size_t bByteIndex = 0;
                if (bNumberOfDims == 1) {
                    bByteIndex = i * sizeof(float);
                } else {
                    size_t bIndices[] = {i, columnIndex};
                    size_t bValueIndex = calcElementIndexByIndices(
                        bNumberOfDims, bDims, bIndices, bTensor->shape->orderOfDimensions);
                    bByteIndex = bValueIndex * sizeof(float);
                }
                float bValue = readBytesAsFloat(&bTensor->data[bByteIndex]);
                result += mulFloat32s(aValue, bValue);
            }

            size_t outputByteIndex = resultCounter * sizeof(float);
            writeFloatToByteArray(result, &outputTensor->data[outputByteIndex]);
            resultCounter++;
        }
    }
}

void matmulFloatTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulFloatCore(aTensor, bTensor, outputTensor, NULL);
}

void matmulFloatTensorsWithInstructionCounter(tensor_t *aTensor, tensor_t *bTensor,
                                              tensor_t *outputTensor) {
    matmulFloatCore(aTensor, bTensor, outputTensor, NULL);
    ++matmulInstructionCounter;
}

void matmulFloat32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                  tensor_t *bias) {
    const uint8_t *seed = NULL;
    if (bias != NULL) {
        size_t bColumns =
            (bTensor->shape->numberOfDimensions < 2) ? 1 : getDimensionsByIndex(bTensor, 1);
        if (calcNumberOfElementsByTensor(bias) != bColumns) {
            PRINT_ERROR("matmulFloat32TensorsWithBias: bias element count != output columns");
            exit(1);
        }
        seed = bias->data;
    }
    matmulFloatCore(aTensor, bTensor, outputTensor, seed);
}

void matmulFloat32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    MATMUL_FUNC_FLOAT(aTensor, bTensor, outputTensor);
}

void matmulSymIntTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulInt32Tensors(aTensor, bTensor, outputTensor);

    symInt32QConfig_t *aSymInt32QC = aTensor->quantization->qConfig;
    symInt32QConfig_t *bSymInt32QC = bTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;

    outputSymInt32QC->scale = aSymInt32QC->scale * bSymInt32QC->scale;
}

void matmulSymIntTensorsWithInstructionCounter(tensor_t *aTensor, tensor_t *bTensor,
                                               tensor_t *outputTensor) {
    matmulInt32Tensors(aTensor, bTensor, outputTensor);

    symInt32QConfig_t *aSymInt32QC = aTensor->quantization->qConfig;
    symInt32QConfig_t *bSymInt32QC = bTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    outputSymInt32QC->scale = aSymInt32QC->scale * bSymInt32QC->scale;

    ++matmulInstructionCounter;
}

static void matmulValidateSymOperand(tensor_t *t, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("matmul SYM_INT32: %s must be SYM_INT32", what);
        exit(1);
    }
    symInt32QConfig_t *qc = t->quantization->qConfig;
    if (qc->qMaxBits > ODT_SYM_OPERAND_QMAXBITS) {
        PRINT_ERROR("matmul SYM_INT32: %s qMaxBits (%u) exceeds operand contract (%u) — int32 "
                    "product accumulation would overflow (#227)",
                    what, (unsigned)qc->qMaxBits, (unsigned)ODT_SYM_OPERAND_QMAXBITS);
        exit(1);
    }
}

void matmulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulValidateSymOperand(aTensor, "aTensor");
    matmulValidateSymOperand(bTensor, "bTensor");
    MATMUL_FUNC_SYM_INT32(aTensor, bTensor, outputTensor);
}

void matmulSymInt32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                   tensor_t *bias) {
    matmulValidateSymOperand(aTensor, "aTensor");
    matmulValidateSymOperand(bTensor, "bTensor");
    if (bias == NULL) {
        matmulIntCore(aTensor, bTensor, outputTensor, NULL);
    } else {
        /* Bias is a value-sum seed (not a product operand), so it is exempt from
         * the int12 operand bound but must still be SYM_INT32: the branch below
         * reads its data as int32 and its qConfig as symInt32QConfig_t (#247). */
        if (bias->quantization->type != SYM_INT32) {
            PRINT_ERROR("matmul SYM_INT32: bias must be SYM_INT32");
            exit(1);
        }
        size_t bColumns =
            (bTensor->shape->numberOfDimensions < 2) ? 1 : getDimensionsByIndex(bTensor, 1);
        if (calcNumberOfElementsByTensor(bias) != bColumns) {
            PRINT_ERROR("matmulSymInt32TensorsWithBias: bias element count != output columns");
            exit(1);
        }

        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        float aScale = ((symInt32QConfig_t *)aTensor->quantization->qConfig)->scale;
        float bScale = ((symInt32QConfig_t *)bTensor->quantization->qConfig)->scale;
        float biasScale = biasQC->scale;
        float outputScale = aScale * bScale;

        /* Rescale the bias into the accumulator's scale via the shared #189 helper
         * (guarded float->int32 cast): one fixed-point op per output column. */
        int32_t seed[bColumns];
        for (size_t c = 0; c < bColumns; c++) {
            int32_t biasIntC = readBytesAsInt32(&bias->data[c * sizeof(int32_t)]);
            seed[c] =
                rescaleIntoAccumulatorScale(biasIntC, biasScale, outputScale, biasQC->roundingMode);
        }
        matmulIntCore(aTensor, bTensor, outputTensor, seed);
    }

    symInt32QConfig_t *aQC = aTensor->quantization->qConfig;
    symInt32QConfig_t *bQC = bTensor->quantization->qConfig;
    symInt32QConfig_t *outputQC = outputTensor->quantization->qConfig;
    outputQC->scale = aQC->scale * bQC->scale;
}

/* Group-quant PR2 (Task 3): sibling of matmulIntCore, adding the running
 * group-partial rescale-combine. `b` must be 2D (a GEMM-family weight,
 * [outCols, reduceLen] storage order) with its reduction axis (logical dim
 * 0) storage-CONTIGUOUS -- true for every real weight wiring today (Linear.c
 * always exposes the physically-innermost axis as the reduction dim via
 * transposeTensor(w,0,1)); groups partition b's flat STORAGE array, so a
 * non-contiguous reduction would make "group of storage index" meaningless
 * and fail-fasts instead of silently misgrouping. Given that contiguity, the
 * per-(row,column) reduction loop below walks RUNS bounded by the next group
 * boundary (one division per RUN, not per element) — mirroring
 * packFloatBufferAsSym's grouped pack loop (Task 2) — rather than
 * re-deriving `b`'s storage index via calcElementIndexByIndices (itself
 * O(numberOfDims) divisions) on every reduction step. */
static void matmulIntCoreGrouped(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                 const int32_t *biasSeed, const symQConfig_t *weightGroups,
                                 float aScale, float sAcc, roundingMode_t roundingMode) {
    if (aTensor->shape->numberOfDimensions > 2 || bTensor->shape->numberOfDimensions != 2) {
        PRINT_ERROR("matmulIntCoreGrouped: grouped weight operand (b) must be 2D; a must be <=2D");
        exit(1);
    }

    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;
    size_t *bDims = bTensor->shape->dimensions;
    size_t *bOrder = bTensor->shape->orderOfDimensions;

    size_t aRows, aColumns;
    if (aNumberOfDims < 2) {
        aRows = 1;
        aColumns = getDimensionsByIndex(aTensor, 0);
    } else {
        aRows = getDimensionsByIndex(aTensor, 0);
        aColumns = getDimensionsByIndex(aTensor, 1);
    }
    size_t bRows = getDimensionsByIndex(bTensor, 0);
    size_t bColumns = getDimensionsByIndex(bTensor, 1);

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %lu, bRows: %lu\n", aColumns, bRows);
        exit(1);
    }

    /* Contiguity check (once, not per element): b's logical-dim-0 (the
     * reduction axis) must advance the physical storage index by exactly 1
     * per step. */
    size_t strideI = 1;
    if (aColumns > 1) {
        size_t idx0[] = {0, 0};
        size_t idx1[] = {1, 0};
        size_t v0 = calcElementIndexByIndices(2, bDims, idx0, bOrder);
        size_t v1 = calcElementIndexByIndices(2, bDims, idx1, bOrder);
        if (v1 != v0 + 1) {
            PRINT_ERROR("matmulIntCoreGrouped: grouped weight reduction axis is not storage-"
                        "contiguous (stride %zu) — groups bind to storage order, only a "
                        "contiguous reduction is supported",
                        (size_t)(v1 - v0));
            exit(1);
        }
    }

    size_t resultCounter = 0;
    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            size_t idxStart[] = {0, columnIndex};
            size_t wBase = calcElementIndexByIndices(2, bDims, idxStart, bOrder);

            int32_t acc = biasSeed ? biasSeed[columnIndex] : 0;
            int32_t partial = 0;
            size_t currentGroup = SIZE_MAX;
            size_t k = 0;
            while (k < aColumns) {
                size_t wStorageIdx = wBase + k;
                size_t g = wStorageIdx / weightGroups->groupSize;
                size_t groupEnd = (g + 1) * weightGroups->groupSize;
                size_t reduceEnd = wBase + aColumns;
                size_t runEnd = groupEnd < reduceEnd ? groupEnd : reduceEnd;
                size_t runLen = runEnd - wStorageIdx;

                if (g != currentGroup) {
                    if (currentGroup != SIZE_MAX) {
                        /* Group-boundary combine: fold the FINISHED group's raw
                         * int32 partial into the running accumulator scale
                         * (one rounding here, honoring the caller's rounding
                         * mode — never hardcoded). */
                        acc += rescaleIntoAccumulatorScale(
                            partial, aScale * weightGroups->scales[currentGroup], sAcc,
                            roundingMode);
                        partial = 0;
                    }
                    currentGroup = g;
                }

                for (size_t r = 0; r < runLen; r++) {
                    size_t i = k + r;
                    size_t aByteIndex;
                    if (aNumberOfDims == 1) {
                        aByteIndex = i * sizeof(int32_t);
                    } else {
                        size_t aIndices[] = {rowIndex, i};
                        size_t aValueIndex = calcElementIndexByIndices(
                            aNumberOfDims, aDims, aIndices, aTensor->shape->orderOfDimensions);
                        aByteIndex = aValueIndex * sizeof(int32_t);
                    }
                    int32_t aValue = readBytesAsInt32(&aTensor->data[aByteIndex]);
                    int32_t bValue =
                        readBytesAsInt32(&bTensor->data[(wStorageIdx + r) * sizeof(int32_t)]);
                    partial += mulInt32s(aValue, bValue);
                }
                k += runLen;
            }
            /* Tail combine: the LAST group never crosses a further boundary,
             * so its partial only ever gets folded in here. Per-channel
             * weights (groupSize == aColumns) never hit the mid-loop branch
             * at all -- this is their ONLY combine. */
            if (currentGroup != SIZE_MAX) {
                acc += rescaleIntoAccumulatorScale(
                    partial, aScale * weightGroups->scales[currentGroup], sAcc, roundingMode);
            }

            size_t outputByteIndex = resultCounter * sizeof(int32_t);
            writeInt32ToByteArray(acc, &outputTensor->data[outputByteIndex]);
            resultCounter++;
        }
    }
}

static void matmulValidateWeightGroups(const symQConfig_t *weightGroups) {
    if (weightGroups == NULL) {
        PRINT_ERROR("matmul SYM_INT32 grouped: weightGroups must not be NULL");
        exit(1);
    }
    if (weightGroups->qBits > ODT_SYM_OPERAND_QMAXBITS) {
        PRINT_ERROR("matmul SYM_INT32 grouped: weightGroups qBits (%u) exceeds operand contract "
                    "(%u) — int32 product accumulation would overflow (#227)",
                    (unsigned)weightGroups->qBits, (unsigned)ODT_SYM_OPERAND_QMAXBITS);
        exit(1);
    }
}

void matmulSymInt32TensorsGroupedWeight(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias,
                                        tensor_t *outputTensor, const symQConfig_t *weightGroups) {
    matmulValidateSymOperand(aTensor, "aTensor");
    matmulValidateWeightGroups(weightGroups);
    validateSymQConfigShape(weightGroups, calcNumberOfElementsByTensor(bTensor));

    /* s_acc = aScale * max_g(scales[g]) (GGUF pattern, #189): a single linear
     * pass over scales[], NEVER scales[0] alone — any other choice would let
     * some group's rescale factor scales[g]/s_wmax exceed 1, growing that
     * group's rescaled mantissa past its accumulator headroom. */
    float aScale = ((symInt32QConfig_t *)aTensor->quantization->qConfig)->scale;
    float maxScale = weightGroups->scales[0];
    for (size_t g = 1; g < weightGroups->numGroups; g++) {
        if (weightGroups->scales[g] > maxScale) {
            maxScale = weightGroups->scales[g];
        }
    }
    float sAcc = aScale * maxScale;

    /* b's roundingMode carries the OP's rounding mode here — the same
     * plumbing the executeOp prologue already uses for every SYM_INT32
     * scratch operand (initSymInt32QConfig(arithmetic.roundingMode, ...)),
     * so a caller invoking this entry directly (as the unit tests do) simply
     * sets it explicitly, exactly like the scalar entries' bias operand. */
    symInt32QConfig_t *bQC = bTensor->quantization->qConfig;
    size_t bColumns = getDimensionsByIndex(bTensor, 1);

    int32_t *seedPtr = NULL;
    int32_t seedBuf[bColumns > 0 ? bColumns : 1];
    if (bias != NULL) {
        if (bias->quantization->type != SYM_INT32) {
            PRINT_ERROR("matmul SYM_INT32 grouped: bias must be SYM_INT32");
            exit(1);
        }
        if (calcNumberOfElementsByTensor(bias) != bColumns) {
            PRINT_ERROR("matmulSymInt32TensorsGroupedWeight: bias element count != output "
                        "columns");
            exit(1);
        }
        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        for (size_t c = 0; c < bColumns; c++) {
            int32_t biasIntC = readBytesAsInt32(&bias->data[c * sizeof(int32_t)]);
            seedBuf[c] =
                rescaleIntoAccumulatorScale(biasIntC, biasQC->scale, sAcc, biasQC->roundingMode);
        }
        seedPtr = seedBuf;
    }

    matmulIntCoreGrouped(aTensor, bTensor, outputTensor, seedPtr, weightGroups, aScale, sAcc,
                         bQC->roundingMode);

    symInt32QConfig_t *outputQC = outputTensor->quantization->qConfig;
    outputQC->scale = sAcc;
}

size_t getMatmulInstructionCounter() {
    return matmulInstructionCounter;
}
