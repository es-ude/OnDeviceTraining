#define SOURCE_FILE "MATMUL"

#ifdef TRACK_INSTRUCTIONS
#define MATMUL_FUNC_FLOAT matmulFloatTensorsWithInstructionCounter
#define MATMUL_FUNC_SYM_INT32 matmulSymIntTensorsWithInstructionCounter
#else
#define MATMUL_FUNC_FLOAT matmulFloatTensors
#define MATMUL_FUNC_SYM_INT32 matmulSymIntTensors
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Arithmetic.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "DTypes.h"
#include "Matmul.h"
#include "Mul.h"
#include "Rounding.h"
#include "ScaleCanon.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"

size_t matmulInstructionCounter = 0;

/* The int32 dot-product loop nest, instantiated unchecked and overflow-checked
 * (SymAccumGuard.h) with `checked` a compile-time constant. */
SYM_ACCUM_ALWAYS_INLINE void matmulIntLoops(tensor_t *aTensor, tensor_t *bTensor,
                                            tensor_t *outputTensor, const int32_t *biasSeed,
                                            size_t aRows, size_t aColumns, size_t bColumns,
                                            const bool checked, const symAccumSite_t *site) {
    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;
    size_t bNumberOfDims = bTensor->shape->numberOfDimensions;
    size_t *bDims = bTensor->shape->dimensions;
    size_t resultCounter = 0;

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

                result = symMac(result, aValue, bValue, checked, site);
            }

            size_t outputByteIndex = resultCounter * sizeof(int32_t);
            writeInt32ToByteArray(result, &outputTensor->data[outputByteIndex]);
            resultCounter++;
        }
    }
}

SYM_ACCUM_NOINLINE void matmulIntLoopsSafe(tensor_t *a, tensor_t *b, tensor_t *out,
                                           const int32_t *biasSeed, size_t aRows, size_t aColumns,
                                           size_t bColumns, const symAccumSite_t *site) {
    matmulIntLoops(a, b, out, biasSeed, aRows, aColumns, bColumns, false, site);
}

SYM_ACCUM_NOINLINE void matmulIntLoopsChecked(tensor_t *a, tensor_t *b, tensor_t *out,
                                              const int32_t *biasSeed, size_t aRows,
                                              size_t aColumns, size_t bColumns,
                                              const symAccumSite_t *site) {
    matmulIntLoops(a, b, out, biasSeed, aRows, aColumns, bColumns, true, site);
}

/* SYM_INT32 operands of declared widths aBits/bBits: the static bound over
 * (aColumns terms, bias seeds) picks the unchecked or the checked loop. Returns
 * the static width of the accumulators written (symAccumResultBits). */
static uint8_t matmulIntCore(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                             const int32_t *biasSeed, uint8_t aBits, uint8_t bBits) {
    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t bNumberOfDims = bTensor->shape->numberOfDimensions;

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

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %zu, bRows: %zu", aColumns, bRows);
        exit(1);
    }

    symAccumSite_t site = symAccumSiteInit("matmul SYM_INT32", aColumns, aBits, bBits, 0);
    symAccumSiteSetSeeds(&site, biasSeed, bColumns);
    if (symAccumProvenSafe(&site)) {
        matmulIntLoopsSafe(aTensor, bTensor, outputTensor, biasSeed, aRows, aColumns, bColumns,
                           &site);
    } else {
        matmulIntLoopsChecked(aTensor, bTensor, outputTensor, biasSeed, aRows, aColumns, bColumns,
                              &site);
    }
    return symAccumResultBits(&site);
}

static void matmulFloatCore(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                            const uint8_t *biasSeed) {
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
        PRINT_DEBUG("aColumns: %zu, bRows: %zu", aColumns, bRows);
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
    matmulFloatCore(aTensor, bTensor, outputTensor, bias != NULL ? bias->data : NULL);
}

void matmulFloat32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    MATMUL_FUNC_FLOAT(aTensor, bTensor, outputTensor);
}

/* The output carries raw accumulator codes wider than the operands, so it
 * declares the width the guard proved; a consumer reading it as a SYM operand
 * sizes its own proof from that. */
static void matmulSymProduct(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    uint8_t aBits = ((symInt32QConfig_t *)aTensor->quantization->qConfig)->qMaxBits;
    uint8_t bBits = ((symInt32QConfig_t *)bTensor->quantization->qConfig)->qMaxBits;
    ((symInt32QConfig_t *)outputTensor->quantization->qConfig)->qMaxBits =
        matmulIntCore(aTensor, bTensor, outputTensor, NULL, aBits, bBits);
#ifdef TRACK_INSTRUCTIONS
    ++matmulInstructionCounter;
#endif
}

void matmulSymIntTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulSymProduct(aTensor, bTensor, outputTensor);

    symInt32QConfig_t *aSymInt32QC = aTensor->quantization->qConfig;
    symInt32QConfig_t *bSymInt32QC = bTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;

    storeCanonScale(&outputSymInt32QC->scale,
                    canonProductScale(loadCanonScale(&aSymInt32QC->scale),
                                      loadCanonScale(&bSymInt32QC->scale), ODT_SCALE_NEAREST));
}

void matmulSymIntTensorsWithInstructionCounter(tensor_t *aTensor, tensor_t *bTensor,
                                               tensor_t *outputTensor) {
    matmulSymProduct(aTensor, bTensor, outputTensor);

    symInt32QConfig_t *aSymInt32QC = aTensor->quantization->qConfig;
    symInt32QConfig_t *bSymInt32QC = bTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    storeCanonScale(&outputSymInt32QC->scale,
                    canonProductScale(loadCanonScale(&aSymInt32QC->scale),
                                      loadCanonScale(&bSymInt32QC->scale), ODT_SCALE_NEAREST));

    ++matmulInstructionCounter;
}

void matmulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    MATMUL_FUNC_SYM_INT32(aTensor, bTensor, outputTensor);
}

void matmulSymInt32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                   tensor_t *bias, rngStream_t *rs) {
    uint8_t aBits = ((symInt32QConfig_t *)aTensor->quantization->qConfig)->qMaxBits;
    uint8_t bBits = ((symInt32QConfig_t *)bTensor->quantization->qConfig)->qMaxBits;
    /* The stored product scale, rounded once from the exact mantissa product;
     * the bias seed below is refolded onto this same value. */
    const float outputScale = canonProductScale(
        loadCanonScale(&((symInt32QConfig_t *)aTensor->quantization->qConfig)->scale),
        loadCanonScale(&((symInt32QConfig_t *)bTensor->quantization->qConfig)->scale),
        ODT_SCALE_NEAREST);
    uint8_t resultBits;
    if (bias == NULL) {
        resultBits = matmulIntCore(aTensor, bTensor, outputTensor, NULL, aBits, bBits);
    } else {
        size_t bColumns =
            (bTensor->shape->numberOfDimensions < 2) ? 1 : getDimensionsByIndex(bTensor, 1);

        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        float biasScale = loadCanonScale(&biasQC->scale);

        /* Rescale the bias into the accumulator's scale (guarded float->int32 cast). */
        int32_t seed[bColumns];
        for (size_t c = 0; c < bColumns; c++) {
            int32_t biasIntC = readBytesAsInt32(&bias->data[c * sizeof(int32_t)]);
            seed[c] = rescaleIntoAccumulatorScale(biasIntC, biasScale, outputScale,
                                                  biasQC->roundingMode, rs);
        }
        resultBits = matmulIntCore(aTensor, bTensor, outputTensor, seed, aBits, bBits);
    }

    symInt32QConfig_t *outputQC = outputTensor->quantization->qConfig;
    storeCanonScale(&outputQC->scale, outputScale);
    outputQC->qMaxBits = resultBits;
}

/* Sibling of matmulIntCore that folds per-group partials. Groups follow the
 * weight's storage index, so any orientation of the reduction axis works,
 * including Linear dx's storage-strided walk. Fold once per visited-group change
 * and at the tail. */
SYM_ACCUM_ALWAYS_INLINE void
matmulGroupedLoops(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                   const int32_t *biasSeed, const symQConfig_t *weightGroups, float aScale,
                   float sAcc, roundingMode_t roundingMode, rngStream_t *rs, size_t aRows,
                   size_t aColumns, size_t bColumns, const bool checked,
                   const symAccumSite_t *partialSite, const symAccumSite_t *accSite) {
    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;
    size_t *bDims = bTensor->shape->dimensions;
    size_t *bOrder = bTensor->shape->orderOfDimensions;

    size_t resultCounter = 0;
    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            int32_t acc = biasSeed ? biasSeed[columnIndex] : 0;
            int32_t partial = 0;
            size_t currentGroup = SIZE_MAX;
            for (size_t i = 0; i < aColumns; i++) {
                size_t bIndices[] = {i, columnIndex};
                size_t wStorageIdx = calcElementIndexByIndices(2, bDims, bIndices, bOrder);
                /* Divide the visited index: a strided walk can jump groups, so a
                 * precomputed run length would overshoot. */
                size_t g = wStorageIdx / weightGroups->groupSize;

                if (g != currentGroup) {
                    if (currentGroup != SIZE_MAX) {
                        /* One rounding per boundary, by the op's rounding mode. */
                        acc =
                            symAdd(acc,
                                   rescalePartialIntoAccumulatorScale(
                                       partial,
                                       aScale * loadCanonScale(&weightGroups->scales[currentGroup]),
                                       sAcc, roundingMode, rs),
                                   checked, accSite);
                        partial = 0;
                    }
                    currentGroup = g;
                }

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
                int32_t bValue = readBytesAsInt32(&bTensor->data[wStorageIdx * sizeof(int32_t)]);
                partial = symMac(partial, aValue, bValue, checked, partialSite);
            }
            /* Tail combine; per-channel forward weights only combine here. */
            if (currentGroup != SIZE_MAX) {
                acc = symAdd(acc,
                             rescalePartialIntoAccumulatorScale(
                                 partial,
                                 aScale * loadCanonScale(&weightGroups->scales[currentGroup]), sAcc,
                                 roundingMode, rs),
                             checked, accSite);
            }

            size_t outputByteIndex = resultCounter * sizeof(int32_t);
            writeInt32ToByteArray(acc, &outputTensor->data[outputByteIndex]);
            resultCounter++;
        }
    }
}

#define MATMUL_GROUPED_PARAMS                                                                      \
    tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor, const int32_t *biasSeed,         \
        const symQConfig_t *weightGroups, float aScale, float sAcc, roundingMode_t roundingMode,   \
        rngStream_t *rs, size_t aRows, size_t aColumns, size_t bColumns,                           \
        const symAccumSite_t *partialSite, const symAccumSite_t *accSite
#define MATMUL_GROUPED_ARGS(checked)                                                               \
    aTensor, bTensor, outputTensor, biasSeed, weightGroups, aScale, sAcc, roundingMode, rs, aRows, \
        aColumns, bColumns, checked, partialSite, accSite

SYM_ACCUM_NOINLINE void matmulGroupedLoopsSafe(MATMUL_GROUPED_PARAMS) {
    matmulGroupedLoops(MATMUL_GROUPED_ARGS(false));
}

SYM_ACCUM_NOINLINE void matmulGroupedLoopsChecked(MATMUL_GROUPED_PARAMS) {
    matmulGroupedLoops(MATMUL_GROUPED_ARGS(true));
}

/* Returns the static width of the RESCALED accumulator it wrote (the output
 * wire), not of the raw group partial. */
static uint8_t matmulIntCoreGrouped(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                    const int32_t *biasSeed,
                                    const symGroupedWeight_t *groupedWeight, float aScale,
                                    float sAcc, roundingMode_t roundingMode, rngStream_t *rs) {
    const symQConfig_t *weightGroups = groupedWeight->groups;
    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;

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
        PRINT_DEBUG("aColumns: %zu, bRows: %zu", aColumns, bRows);
        exit(1);
    }

    /* Two int32 accumulators, two bounds. The raw group partial holds at most
     * min(groupSize, N) products (a group never contributes more elements than
     * it holds; a revisit after a strided hop starts a fresh partial). The
     * running acc = seed + the rescaled partials: each rescale factor is
     * scales[g]/max(scales) <= 1, so |rescaled partial| <= |partial| up to the
     * float32 evaluation slack (the rescaled symAccumSiteTermLimit) and the sum is
     * bounded by N slacked products plus the seed. Proving the acc bound proves
     * the partial bound too (fewer terms, no seed), so one flag drives both. */
    /* The weight's operand range is its code domain's (SymGroupedWeight.h): a
     * stored SYM weight's two's-complement qBits range, a stored ASYM weight's
     * zp-shifted image +-(2^qBits - 1) -- the scratch's declared qMaxBits is
     * not read. The abort banner reports each range's holding width (the
     * image's is qBits + 1). */
    int32_t aMin, aMax, bMin, bMax;
    symAccumWidthRange(((symInt32QConfig_t *)aTensor->quantization->qConfig)->qMaxBits, &aMin,
                       &aMax);
    symGroupedWeightRange(groupedWeight, &bMin, &bMax);
    size_t partialTerms = weightGroups->groupSize < aColumns ? weightGroups->groupSize : aColumns;
    symAccumSite_t partialSite = symAccumSiteInitRanges(
        "matmul SYM_INT32 grouped (raw group partial)", partialTerms, aMin, aMax, bMin, bMax, 0);
    symAccumSite_t accSite = symAccumSiteInitRangesRescaled(
        "matmul SYM_INT32 grouped (rescaled accumulator)", aColumns, aMin, aMax, bMin, bMax, 0);
    symAccumSiteSetSeeds(&accSite, biasSeed, bColumns);
    if (symAccumProvenSafe(&accSite)) {
        matmulGroupedLoopsSafe(aTensor, bTensor, outputTensor, biasSeed, weightGroups, aScale, sAcc,
                               roundingMode, rs, aRows, aColumns, bColumns, &partialSite, &accSite);
    } else {
        matmulGroupedLoopsChecked(aTensor, bTensor, outputTensor, biasSeed, weightGroups, aScale,
                                  sAcc, roundingMode, rs, aRows, aColumns, bColumns, &partialSite,
                                  &accSite);
    }
    return symAccumResultBits(&accSite);
}

void matmulSymInt32TensorsGroupedWeight(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias,
                                        tensor_t *outputTensor,
                                        const symGroupedWeight_t *groupedWeight, rngStream_t *rs) {
    const symQConfig_t *weightGroups = groupedWeight->groups;

    /* s_acc uses the maximum group scale so every rescale factor is <= 1. */
    float aScale = loadCanonScale(&((symInt32QConfig_t *)aTensor->quantization->qConfig)->scale);
    float maxScale = loadCanonScale(&weightGroups->scales[0]);
    for (size_t g = 1; g < weightGroups->numGroups; g++) {
        const float groupScale = loadCanonScale(&weightGroups->scales[g]);
        if (groupScale > maxScale) {
            maxScale = groupScale;
        }
    }
    /* Stored, so canonical. CEIL, not the product rule's NEAREST: the rescaled
     * accumulator proof (symAccumSiteInitRangesRescaled) needs every group
     * factor (aScale * scales[g]) / sAcc <= 1 up to float slack 2^-22, and a
     * NEAREST sAcc can sit 2^-15 below the exact product. */
    const float sAcc = canonProductScale(aScale, maxScale, ODT_SCALE_CEIL);

    /* b's scratch config carries the op's rounding mode (set by the executeOp
     * prologue; direct callers set it explicitly). */
    symInt32QConfig_t *bQC = bTensor->quantization->qConfig;
    size_t bColumns = getDimensionsByIndex(bTensor, 1);

    int32_t *seedPtr = NULL;
    int32_t seedBuf[bColumns > 0 ? bColumns : 1];
    if (bias != NULL) {
        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        const float biasScale = loadCanonScale(&biasQC->scale);
        for (size_t c = 0; c < bColumns; c++) {
            int32_t biasIntC = readBytesAsInt32(&bias->data[c * sizeof(int32_t)]);
            seedBuf[c] =
                rescaleIntoAccumulatorScale(biasIntC, biasScale, sAcc, biasQC->roundingMode, rs);
        }
        seedPtr = seedBuf;
    }

    uint8_t resultBits = matmulIntCoreGrouped(aTensor, bTensor, outputTensor, seedPtr,
                                              groupedWeight, aScale, sAcc, bQC->roundingMode, rs);

    symInt32QConfig_t *outputQC = outputTensor->quantization->qConfig;
    storeCanonScale(&outputQC->scale, sAcc);
    outputQC->qMaxBits = resultBits; /* the rescaled accumulator's width */
}

/* matmulIntCoreGrouped's BFP twin (contract in Matmul.h). Differences from the SYM grouped core:
 * BOTH operands are group-tracked (not just the weight), the boundary combine is a rounding-free
 * power-of-two fold into a float accumulator (never rescaleIntoAccumulatorScale), and the output
 * is raw FLOAT32. */
void matmulBfpTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias,
                      tensor_t *outputTensor) {
    bfpQConfig_t *aQC = aTensor->quantization->qConfig;
    bfpQConfig_t *bQC = bTensor->quantization->qConfig;

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

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %zu, bRows: %zu", aColumns, bRows);
        exit(1);
    }

    bfpQConfig_t *biasQC = bias != NULL ? bias->quantization->qConfig : NULL;

    bfpValidateBlockHeadroom(aQC, bQC, aColumns, "matmulBfpTensors");

    int32_t aExpBias = bfpExponentBias(aQC);
    int32_t bExpBias = bfpExponentBias(bQC);
    bfpGroupRange_t aRange = {0};
    bfpGroupRange_t bRange = {0};

    size_t resultCounter = 0;
    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            float acc = 0.f;
            if (bias != NULL) {
                /* Value-seed: dequantized to float BEFORE the reduction, so
                 * it never touches the int32 partial (headroom-exempt). */
                int32_t biasMantissa = readBytesAsInt32(&bias->data[columnIndex * sizeof(int32_t)]);
                acc = (float)biasMantissa * bfpGroupScale(biasQC, bfpGroupOf(biasQC, columnIndex));
            }
            int32_t partial = 0;
            size_t currentAGroup = 0;
            size_t currentBGroup = 0;
            for (size_t i = 0; i < aColumns; i++) {
                size_t aValueIndex;
                if (aNumberOfDims == 1) {
                    aValueIndex = i;
                } else {
                    size_t aIndices[] = {rowIndex, i};
                    aValueIndex = calcElementIndexByIndices(aNumberOfDims, aDims, aIndices,
                                                            aTensor->shape->orderOfDimensions);
                }
                size_t bValueIndex;
                if (bNumberOfDims == 1) {
                    bValueIndex = i;
                } else {
                    size_t bIndices[] = {i, columnIndex};
                    bValueIndex = calcElementIndexByIndices(bNumberOfDims, bDims, bIndices,
                                                            bTensor->shape->orderOfDimensions);
                }
                size_t aGroup = bfpGroupAt(&aRange, aQC, aValueIndex);
                size_t bGroup = bfpGroupAt(&bRange, bQC, bValueIndex);
                if (i == 0) {
                    currentAGroup = aGroup;
                    currentBGroup = bGroup;
                } else if (aGroup != currentAGroup || bGroup != currentBGroup) {
                    /* Boundary fold: the finished same-exponent segment's raw
                     * int32 partial enters the float accumulator via a pure
                     * exponent shift -- rounding-free by kernel contract. */
                    acc +=
                        ldexpf((float)partial, (int)aQC->exponents[currentAGroup] - aExpBias +
                                                   (int)bQC->exponents[currentBGroup] - bExpBias);
                    partial = 0;
                    currentAGroup = aGroup;
                    currentBGroup = bGroup;
                }
                int32_t aValue = readBytesAsInt32(&aTensor->data[aValueIndex * sizeof(int32_t)]);
                int32_t bValue = readBytesAsInt32(&bTensor->data[bValueIndex * sizeof(int32_t)]);
                partial += mulInt32s(aValue, bValue);
            }
            if (aColumns > 0) {
                acc += ldexpf((float)partial, (int)aQC->exponents[currentAGroup] - aExpBias +
                                                  (int)bQC->exponents[currentBGroup] - bExpBias);
            }
            writeFloatToByteArray(acc, &outputTensor->data[resultCounter * sizeof(float)]);
            resultCounter++;
        }
    }
}

/* Max unbiased exponent of each reduction LINE of a matmul operand (line l =
 * the elements one output row reads from `a`, or one output column from
 * `b`), over the groups holding a nonzero code of the line (int16 marker
 * INT16_MIN: none). When the reduction axis is the physically innermost
 * dimension (1-D operands trivially), line l is the storage run
 * [l * k, (l + 1) * k) and one forward cursor serves all lines (no
 * division). Any other view gets the operand-wide max: coarser, still an
 * upper bound. */
static void matmulBfpLineMaxExponents(const tensor_t *t, size_t reductionAxis, size_t lines,
                                      size_t k, int16_t *lineMax) {
    const bfpQConfig_t *qC = t->quantization->qConfig;
    int32_t expBias = bfpExponentBias(qC);
    if (t->shape->numberOfDimensions < 2 || t->shape->orderOfDimensions[1] == reductionAxis) {
        bfpRowCursor_t c = {.group = 0, .groupEnd = qC->groupSize};
        for (size_t l = 0; l < lines; l++) {
            lineMax[l] = bfpLiveExpTo16(
                bfpRowMaxExponent(&c, qC, expBias, (const int32_t *)t->data, l * k, k));
        }
        return;
    }
    bfpRowCursor_t c = {.group = 0, .groupEnd = qC->groupSize};
    int e = bfpRowMaxExponent(&c, qC, expBias, (const int32_t *)t->data, 0, lines * k);
    for (size_t l = 0; l < lines; l++) {
        lineMax[l] = bfpLiveExpTo16(e);
    }
}

/* What the walk of matmulBfpRawTensors reads; the tables are the pre-pass's. */
typedef struct matmulBfpRawWalk {
    const tensor_t *aTensor;
    const tensor_t *bTensor;
    size_t aNumberOfDims, bNumberOfDims;
    size_t *aDims;
    size_t *bDims;
    size_t aRows, aColumns, bColumns, gsz, segmentLimit;
    const bfpQConfig_t *aQC;
    const bfpQConfig_t *bQC;
    int32_t aExpBias, bExpBias;
    const int32_t *aArr;
    const int32_t *bArr;
    const int32_t *biasArr; /* NULL without bias */
    const int16_t *biasE;   /* [bColumns] */
    const int16_t *rawE;    /* E_acc per raw target group */
    int32_t *mant;
} matmulBfpRawWalk_t;

/* The MAC walk (BFP_WALK_NOINLINE: no VLA in this frame). */
BFP_WALK_NOINLINE void matmulBfpRawTensorsWalk(const matmulBfpRawWalk_t *p) {
    const tensor_t *aTensor = p->aTensor;
    const tensor_t *bTensor = p->bTensor;
    size_t aNumberOfDims = p->aNumberOfDims, bNumberOfDims = p->bNumberOfDims;
    size_t *aDims = p->aDims;
    size_t *bDims = p->bDims;
    size_t aRows = p->aRows, aColumns = p->aColumns, bColumns = p->bColumns;
    size_t gsz = p->gsz, segmentLimit = p->segmentLimit;
    const bfpQConfig_t *aQC = p->aQC;
    const bfpQConfig_t *bQC = p->bQC;
    int32_t aExpBias = p->aExpBias, bExpBias = p->bExpBias;
    int32_t const *aArr = p->aArr;
    int32_t const *bArr = p->bArr;
    int32_t const *biasArr = p->biasArr;
    const int16_t *biasE = p->biasE;
    const int16_t *rawE = p->rawE;
    int32_t *mant = p->mant;

    bfpGroupRange_t aRange = {0};
    bfpGroupRange_t bRange = {0};
    size_t targetGroup = 0;
    size_t targetEnd = gsz;
    int eAcc = rawE[0];

    size_t resultCounter = 0;
    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            if (resultCounter >= targetEnd) {
                targetGroup++;
                targetEnd += gsz;
                eAcc = rawE[targetGroup];
            }
            /* The bias is one more segment (§13 item 6); the column changes
             * every output, so it is aligned per output. */
            int32_t acc = biasArr != NULL
                              ? bfpAlignPartial(biasArr[columnIndex], eAcc - biasE[columnIndex])
                              : 0;
            int32_t partial = 0;
            size_t segCount = 0;
            size_t currentAGroup = 0;
            size_t currentBGroup = 0;
            for (size_t i = 0; i < aColumns; i++) {
                size_t aValueIndex;
                if (aNumberOfDims == 1) {
                    aValueIndex = i;
                } else {
                    size_t aIndices[] = {rowIndex, i};
                    aValueIndex = calcElementIndexByIndices(aNumberOfDims, aDims, aIndices,
                                                            aTensor->shape->orderOfDimensions);
                }
                size_t bValueIndex;
                if (bNumberOfDims == 1) {
                    bValueIndex = i;
                } else {
                    size_t bIndices[] = {i, columnIndex};
                    bValueIndex = calcElementIndexByIndices(bNumberOfDims, bDims, bIndices,
                                                            bTensor->shape->orderOfDimensions);
                }
                size_t aGroup = bfpGroupAt(&aRange, aQC, aValueIndex);
                size_t bGroup = bfpGroupAt(&bRange, bQC, bValueIndex);
                if (i == 0) {
                    currentAGroup = aGroup;
                    currentBGroup = bGroup;
                } else if (aGroup != currentAGroup || bGroup != currentBGroup ||
                           segCount == segmentLimit) {
                    /* A segment closes on EITHER operand's group change or at
                     * the int32 limit (run split); its partial is aligned to
                     * E_acc. */
                    acc += bfpAlignPartial(partial,
                                           eAcc - ((int)aQC->exponents[currentAGroup] - aExpBias +
                                                   (int)bQC->exponents[currentBGroup] - bExpBias));
                    partial = 0;
                    segCount = 0;
                    currentAGroup = aGroup;
                    currentBGroup = bGroup;
                }
                partial += mulInt32s(aArr[aValueIndex], bArr[bValueIndex]);
                segCount++;
            }
            if (aColumns > 0) {
                acc += bfpAlignPartial(partial,
                                       eAcc - ((int)aQC->exponents[currentAGroup] - aExpBias +
                                               (int)bQC->exponents[currentBGroup] - bExpBias));
            }
            mant[resultCounter] = acc;
            resultCounter++;
        }
    }
}

/* Integer-only forward (arithmetic-bfp.md §13): matmulBfpTensors' walk
 * verbatim -- loop order, MAC count, calcElementIndexByIndices, bfpGroupAt on
 * both operands -- with the float fold replaced by the integer fold (see
 * Matmul.h). */
void matmulBfpRawTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias,
                         tensor_t *outputTensor) {
    bfpQConfig_t *aQC = aTensor->quantization->qConfig;
    bfpQConfig_t *bQC = bTensor->quantization->qConfig;

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

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %zu, bRows: %zu", aColumns, bRows);
        exit(1);
    }

    bfpQConfig_t *biasQC = bias != NULL ? bias->quantization->qConfig : NULL;

    size_t outElems = aRows * bColumns;
    bfpRawQConfig_t *rawQC = outputTensor->quantization->qConfig;
    size_t gsz = bfpRawRequireOutput(rawQC, outElems, "matmulBfpRawTensors");
    int16_t *rawE = rawQC->exponents;
    if (outElems == 0) {
        return;
    }

    /* Static headroom proof (§13 item 3) over the reduction; over-long
     * same-exponent runs split at the segment limit instead of failing. */
    int guard = bfpFoldGuardBits(aColumns, aQC->mantissaBits, bQC->mantissaBits,
                                 biasQC != NULL ? biasQC->mantissaBits : 0, "matmulBfpRawTensors");
    size_t segmentLimit = bfpSegmentLimit(aQC->mantissaBits, bQC->mantissaBits);

    int32_t const *aArr = (int32_t const *)aTensor->data;
    int32_t const *bArr = (int32_t const *)bTensor->data;
    int32_t const *biasArr = bias != NULL ? (int32_t const *)bias->data : NULL;
    int32_t *mant = (int32_t *)outputTensor->data;
    int32_t aExpBias = bfpExponentBias(aQC);
    int32_t bExpBias = bfpExponentBias(bQC);

    /* Pre-pass (stack tables sized by the topology, never by the MACs): per
     * output row / column the max exponent of the line it reduces over, per
     * column the bias exponent; E_acc of a raw target group = the max over its
     * outputs of max(rowTop + colTop, biasE) + g, walked in output order. A
     * line without nonzero codes and a zero bias code set nothing; a group
     * with nothing set takes the zero state's top 0. */
    int16_t aMax[aRows];
    int16_t bMax[bColumns];
    int16_t biasE[biasQC != NULL ? bColumns : 1];
    matmulBfpLineMaxExponents(aTensor, 1, aRows, aColumns, aMax);
    matmulBfpLineMaxExponents(bTensor, 0, bColumns, aColumns, bMax);
    if (biasQC != NULL) {
        bfpGroupRange_t biasRange = {0};
        int32_t biasExpBias = bfpExponentBias(biasQC);
        for (size_t c = 0; c < bColumns; c++) {
            biasE[c] =
                (int16_t)((int)biasQC->exponents[bfpGroupAt(&biasRange, biasQC, c)] - biasExpBias);
        }
    }
    {
        size_t t = 0;
        size_t tEnd = gsz;
        int top = INT_MIN;
        size_t o = 0;
        for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
            for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
                int e = bfpLiveExpSum(bfpLiveExpFrom16(aMax[rowIndex]),
                                      bfpLiveExpFrom16(bMax[columnIndex]));
                if (biasArr != NULL && biasArr[columnIndex] != 0 && biasE[columnIndex] > e) {
                    e = biasE[columnIndex];
                }
                if (e > top) {
                    top = e;
                }
                if (++o == tEnd) {
                    rawE[t++] = (int16_t)(bfpLiveTopOrZero(top) + guard);
                    tEnd += gsz;
                    top = INT_MIN;
                }
            }
        }
    }

    matmulBfpRawWalk_t walk = {.aTensor = aTensor,
                               .bTensor = bTensor,
                               .aNumberOfDims = aNumberOfDims,
                               .bNumberOfDims = bNumberOfDims,
                               .aDims = aDims,
                               .bDims = bDims,
                               .aRows = aRows,
                               .aColumns = aColumns,
                               .bColumns = bColumns,
                               .gsz = gsz,
                               .segmentLimit = segmentLimit,
                               .aQC = aQC,
                               .bQC = bQC,
                               .aExpBias = aExpBias,
                               .bExpBias = bExpBias,
                               .aArr = aArr,
                               .bArr = bArr,
                               .biasArr = biasArr,
                               .biasE = biasE,
                               .rawE = rawE,
                               .mant = mant};
    matmulBfpRawTensorsWalk(&walk);
}

size_t getMatmulInstructionCounter() {
    return matmulInstructionCounter;
}
