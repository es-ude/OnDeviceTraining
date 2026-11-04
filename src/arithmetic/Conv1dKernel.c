#define SOURCE_FILE "ODT_CONV1D_KERNEL"

#include "Conv1dKernel.h"

#include <math.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Mul.h"
#include "Rounding.h"
#include "SlidingWindow1d.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"

void conv1dKernelFloat32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                         kernel_t const *kernel, size_t groups, tensor_t *output) {
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("conv1dKernelFloat32: groups (%zu) must divide in_channels "
                    "(%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("conv1dKernelFloat32: output_length mismatch "
                    "(geometry=%zu, output tensor=%zu)",
                    geom.outputLength, outputLength);
        exit(1);
    }

    float const *xArr = (float const *)input->data;
    float const *wArr = (float const *)weight->data;
    float const *bArr = bias ? (float const *)bias->data : NULL;
    float *yArr = (float *)output->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    float sum = bArr ? bArr[oc] : 0.0f;

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;

                            float xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            float wv =
                                wArr[(oc * inChPerGroup + icOffset) * kernelSize + kernelIdx];
                            sum += xv * wv;
                        }
                    }

                    yArr[(b * outChannels + oc) * outputLength + outPos] = sum;
                }
            }
        }
    }
}

/* Conv shape and operand views shared by the SYM loop nests below. */
typedef struct convSymView {
    size_t batch, inChannels, inputLength, outChannels, outputLength, kernelSize;
    size_t groups, inChPerGroup, outChPerGroup;
    const windowGeometry1d_t *geom;
    const int32_t *xArr;
    const int32_t *wArr;
    int32_t *yArr;
    const int32_t *seed;
} convSymView_t;

/* Instantiated twice by conv1dKernelSymInt32 with `checked` a compile-time
 * constant: false is the historical body (sum + mulInt32s), true routes every
 * accumulate through the SYM accumulator guard (SymAccumGuard.h). */
SYM_ACCUM_ALWAYS_INLINE void conv1dSymLoops(const convSymView_t *v, const bool checked,
                                            const symAccumSite_t *site) {
    const size_t batch = v->batch, groups = v->groups, inChPerGroup = v->inChPerGroup;
    const size_t outChPerGroup = v->outChPerGroup, outChannels = v->outChannels;
    const size_t outputLength = v->outputLength, inChannels = v->inChannels;
    const size_t inputLength = v->inputLength, kernelSize = v->kernelSize;
    const windowGeometry1d_t geom = *v->geom;
    const int32_t *xArr = v->xArr, *wArr = v->wArr, *seed = v->seed;
    int32_t *yArr = v->yArr;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    int32_t sum = seed[oc];

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;

                            int32_t xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            int32_t wv =
                                wArr[(oc * inChPerGroup + icOffset) * kernelSize + kernelIdx];
                            sum = symMac(sum, xv, wv, checked, site);
                        }
                    }

                    yArr[(b * outChannels + oc) * outputLength + outPos] = sum;
                }
            }
        }
    }
}

SYM_ACCUM_NOINLINE void conv1dSymLoopsSafe(const convSymView_t *v, const symAccumSite_t *site) {
    conv1dSymLoops(v, false, site);
}

SYM_ACCUM_NOINLINE void conv1dSymLoopsChecked(const convSymView_t *v, const symAccumSite_t *site) {
    conv1dSymLoops(v, true, site);
}

/* Every operand is read through symInt32QConfig_t / as int32 codes, so a
 * different storage dtype must abort here instead of being reinterpreted. The
 * grouped entries pass the weight as a SYM_INT32 view too (its per-group
 * qConfig rides in weightGroups), so the same check holds for them. */
static void conv1dValidateSymOperand(const tensor_t *t, const char *kernelName, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("%s: %s must be SYM_INT32", kernelName, what);
        exit(1);
    }
}

static void conv1dValidateSymOperands(const tensor_t *input, const tensor_t *weight,
                                      const tensor_t *bias, const tensor_t *output,
                                      const char *kernelName) {
    conv1dValidateSymOperand(input, kernelName, "input");
    conv1dValidateSymOperand(weight, kernelName, "weight");
    if (bias != NULL) {
        conv1dValidateSymOperand(bias, kernelName, "bias");
    }
    conv1dValidateSymOperand(output, kernelName, "output");
}

void conv1dKernelSymInt32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                          kernel_t const *kernel, size_t groups, tensor_t *output,
                          rngStream_t *rs) {
    conv1dValidateSymOperands(input, weight, bias, output, "conv1dKernelSymInt32");
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("conv1dKernelSymInt32: groups (%zu) must divide in_channels "
                    "(%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("conv1dKernelSymInt32: output_length mismatch "
                    "(geometry=%zu, output tensor=%zu)",
                    geom.outputLength, outputLength);
        exit(1);
    }

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t const *wArr = (int32_t const *)weight->data;
    int32_t *yArr = (int32_t *)output->data;

    float inScale = ((symInt32QConfig_t *)input->quantization->qConfig)->scale;
    float wScale = ((symInt32QConfig_t *)weight->quantization->qConfig)->scale;
    float outputScale = inScale * wScale;

    /* Per-output-channel bias seed, refolded into the accumulator product scale
     * via the #189 guarded helper (NULL bias -> all zero). VLA over channels
     * (topology-bounded), mirroring matmulSymInt32TensorsWithBias's seed[bColumns]. */
    int32_t seed[outChannels];
    if (bias != NULL) {
        int32_t const *bArr = (int32_t const *)bias->data;
        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        for (size_t oc = 0; oc < outChannels; oc++) {
            seed[oc] = rescaleIntoAccumulatorScale(bArr[oc], biasQC->scale, outputScale,
                                                   biasQC->roundingMode, rs);
        }
    } else {
        for (size_t oc = 0; oc < outChannels; oc++) {
            seed[oc] = 0;
        }
    }

    convSymView_t view = {.batch = batch,
                          .inChannels = inChannels,
                          .inputLength = inputLength,
                          .outChannels = outChannels,
                          .outputLength = outputLength,
                          .kernelSize = kernelSize,
                          .groups = groups,
                          .inChPerGroup = inChPerGroup,
                          .outChPerGroup = outChPerGroup,
                          .geom = &geom,
                          .xArr = xArr,
                          .wArr = wArr,
                          .yArr = yArr,
                          .seed = seed};
    /* Per output element: at most inChPerGroup * kernelSize products (padding
     * only shrinks it), on top of the bias seed. */
    uint8_t xBits = ((symInt32QConfig_t *)input->quantization->qConfig)->qMaxBits;
    uint8_t wBits = ((symInt32QConfig_t *)weight->quantization->qConfig)->qMaxBits;
    symAccumSite_t site =
        symAccumSiteInit("conv1dKernelSymInt32", inChPerGroup * kernelSize, xBits, wBits, 0);
    symAccumSiteSetSeeds(&site, seed, outChannels);
    if (symAccumProvenSafe(&site)) {
        conv1dSymLoopsSafe(&view, &site);
    } else {
        conv1dSymLoopsChecked(&view, &site);
    }

    if (output->quantization->qConfig == NULL) {
        PRINT_ERROR("conv1dKernelSymInt32: output qConfig is NULL but SYM_INT32 expected (#187)");
        exit(1);
    }
    symInt32QConfig_t *outputQC = (symInt32QConfig_t *)output->quantization->qConfig;
    outputQC->scale = outputScale;
    /* The output carries raw accumulator codes that can exceed the operands'
     * width, so it declares the width the guard proved (#516): a consumer that
     * reads this wire as a SYM operand sizes its own proof from it. */
    outputQC->qMaxBits = symAccumResultBits(&site);
}

/* Group-quant PR2 (Task 4): entry guards mirror matmulValidateWeightGroups
 * (Matmul.c) exactly -- non-NULL and the per-tensor sentinel
 * {numGroups=1, groupSize=0} rejected (its groupSize=0 would divide-by-zero
 * below; per-tensor weights take conv1dKernelSymInt32). */
static void conv1dValidateWeightGroups(const symGroupedWeight_t *groupedWeight,
                                       size_t weightElemCount) {
    if (groupedWeight == NULL || groupedWeight->groups == NULL) {
        PRINT_ERROR("conv1dKernelSymInt32Grouped: weightGroups must not be NULL");
        exit(1);
    }
    const symQConfig_t *weightGroups = groupedWeight->groups;
    if (weightGroups->numGroups <= 1) {
        PRINT_ERROR("conv1dKernelSymInt32Grouped: weightGroups must be grouped (numGroups>1); "
                    "per-tensor weights take conv1dKernelSymInt32");
        exit(1);
    }
    validateSymQConfigShape(weightGroups, weightElemCount);
}

/* Grouped twin of conv1dSymLoops: two int32 accumulators (the raw group
 * partial and the running acc of rescaled partials), each with its own site. */
SYM_ACCUM_ALWAYS_INLINE void
conv1dSymGroupedLoops(const convSymView_t *v, const symQConfig_t *weightGroups, float inScale,
                      float sAcc, roundingMode_t roundingMode, rngStream_t *rs, const bool checked,
                      const symAccumSite_t *partialSite, const symAccumSite_t *accSite) {
    const size_t batch = v->batch, groups = v->groups, inChPerGroup = v->inChPerGroup;
    const size_t outChPerGroup = v->outChPerGroup, outChannels = v->outChannels;
    const size_t outputLength = v->outputLength, inChannels = v->inChannels;
    const size_t inputLength = v->inputLength, kernelSize = v->kernelSize;
    const windowGeometry1d_t geom = *v->geom;
    const int32_t *xArr = v->xArr, *wArr = v->wArr, *seed = v->seed;
    int32_t *yArr = v->yArr;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                size_t wBase = oc * inChPerGroup * kernelSize;

                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    int32_t acc = seed[oc];
                    int32_t partial = 0;
                    size_t currentGroup = SIZE_MAX;

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;
                            size_t wStorageIdx = wBase + icOffset * kernelSize + kernelIdx;
                            /* Per-element division, unlike Matmul's grouped kernel (which
                             * precomputes a run length to the next group boundary): padding
                             * (VALID/SAME/EXPLICIT) makes slice.validCount skip invalid kernel
                             * positions, so consecutive wStorageIdx values here can have gaps --
                             * a precomputed run length would overshoot past a skipped index.
                             * Dividing the actual visited index is gap-robust. */
                            size_t elemGroup = wStorageIdx / weightGroups->groupSize;

                            if (elemGroup != currentGroup) {
                                if (currentGroup != SIZE_MAX) {
                                    /* Group-boundary combine: fold the FINISHED
                                     * group's raw int32 partial into the running
                                     * accumulator scale (one rounding here,
                                     * honoring the op's rounding mode). */
                                    acc = symAdd(acc,
                                                 rescalePartialIntoAccumulatorScale(
                                                     partial,
                                                     inScale * weightGroups->scales[currentGroup],
                                                     sAcc, roundingMode, rs),
                                                 checked, accSite);
                                    partial = 0;
                                }
                                currentGroup = elemGroup;
                            }

                            int32_t xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            int32_t wv = wArr[wStorageIdx];
                            partial = symMac(partial, xv, wv, checked, partialSite);
                        }
                    }
                    /* Tail combine: the LAST group visited never crosses a
                     * further boundary, so its partial only ever gets folded
                     * in here. Per-channel weights (groupSize == the full
                     * inChPerGroup*kernelSize reduction) never hit the
                     * mid-loop branch at all -- this is their ONLY weight
                     * combine. */
                    if (currentGroup != SIZE_MAX) {
                        acc = symAdd(acc,
                                     rescalePartialIntoAccumulatorScale(
                                         partial, inScale * weightGroups->scales[currentGroup],
                                         sAcc, roundingMode, rs),
                                     checked, accSite);
                    }

                    yArr[(b * outChannels + oc) * outputLength + outPos] = acc;
                }
            }
        }
    }
}

#define CONV1D_GROUPED_PARAMS                                                                      \
    const convSymView_t *v, const symQConfig_t *weightGroups, float inScale, float sAcc,           \
        roundingMode_t roundingMode, rngStream_t *rs, const symAccumSite_t *partialSite,           \
        const symAccumSite_t *accSite
#define CONV1D_GROUPED_ARGS(checked)                                                               \
    v, weightGroups, inScale, sAcc, roundingMode, rs, checked, partialSite, accSite

SYM_ACCUM_NOINLINE void conv1dSymGroupedLoopsSafe(CONV1D_GROUPED_PARAMS) {
    conv1dSymGroupedLoops(CONV1D_GROUPED_ARGS(false));
}

SYM_ACCUM_NOINLINE void conv1dSymGroupedLoopsChecked(CONV1D_GROUPED_PARAMS) {
    conv1dSymGroupedLoops(CONV1D_GROUPED_ARGS(true));
}

void conv1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                 tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                 tensor_t *output, const symGroupedWeight_t *groupedWeight,
                                 rngStream_t *rs) {
    conv1dValidateSymOperands(input, weight, bias, output, "conv1dKernelSymInt32Grouped");
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("conv1dKernelSymInt32Grouped: groups (%zu) must divide in_channels "
                    "(%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    conv1dValidateWeightGroups(groupedWeight, outChannels * inChPerGroup * kernelSize);
    const symQConfig_t *weightGroups = groupedWeight->groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("conv1dKernelSymInt32Grouped: output_length mismatch "
                    "(geometry=%zu, output tensor=%zu)",
                    geom.outputLength, outputLength);
        exit(1);
    }

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t const *wArr = (int32_t const *)weight->data;
    int32_t *yArr = (int32_t *)output->data;

    float inScale = ((symInt32QConfig_t *)input->quantization->qConfig)->scale;

    /* s_acc = inScale * max_g(weightGroups->scales[g]) (GGUF pattern, #189):
     * a single linear pass over scales[], NEVER scales[0] alone -- see
     * matmulSymInt32TensorsGroupedWeight's identical derivation. */
    float maxScale = weightGroups->scales[0];
    for (size_t g = 1; g < weightGroups->numGroups; g++) {
        if (weightGroups->scales[g] > maxScale) {
            maxScale = weightGroups->scales[g];
        }
    }
    float sAcc = inScale * maxScale;

    /* `weight`'s own poisoned symInt32QConfig_t carries the OP's rounding
     * mode (executeOp's prologue sets it via initSymInt32QConfig(arithmetic.
     * roundingMode, ...) BEFORE poisoning only .scale/.qMaxBits) -- the same
     * plumbing matmulSymInt32TensorsGroupedWeight reads via its `bQC`. */
    symInt32QConfig_t *wQC = (symInt32QConfig_t *)weight->quantization->qConfig;

    /* Per-output-channel bias seed, refolded into sAcc via the #189 guarded
     * helper (NULL bias -> all zero), mirroring conv1dKernelSymInt32. */
    int32_t seed[outChannels];
    if (bias != NULL) {
        int32_t const *bArr = (int32_t const *)bias->data;
        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        for (size_t oc = 0; oc < outChannels; oc++) {
            seed[oc] = rescaleIntoAccumulatorScale(bArr[oc], biasQC->scale, sAcc,
                                                   biasQC->roundingMode, rs);
        }
    } else {
        for (size_t oc = 0; oc < outChannels; oc++) {
            seed[oc] = 0;
        }
    }

    convSymView_t view = {.batch = batch,
                          .inChannels = inChannels,
                          .inputLength = inputLength,
                          .outChannels = outChannels,
                          .outputLength = outputLength,
                          .kernelSize = kernelSize,
                          .groups = groups,
                          .inChPerGroup = inChPerGroup,
                          .outChPerGroup = outChPerGroup,
                          .geom = &geom,
                          .xArr = xArr,
                          .wArr = wArr,
                          .yArr = yArr,
                          .seed = seed};
    /* Raw partial: at most min(groupSize, N) products; acc: the rescaled
     * partials (each factor <= 1, plus the float32 slack of
     * symAccumTermLimitRescaled) plus the seed, so N slacked products plus
     * seed bound it. Proving the acc proves the partial (fewer terms, no seed). */
    size_t nTerms = inChPerGroup * kernelSize;
    /* The weight's operand range is its code domain's (SymGroupedWeight.h): a
     * stored SYM weight's two's-complement qBits range, a stored ASYM weight's
     * zp-shifted image +-(2^qBits - 1) -- the scratch's declared qMaxBits is
     * not read. The abort banner reports each range's holding width (the
     * image's is qBits + 1). */
    int32_t xMin, xMax, wMin, wMax;
    symAccumWidthRange(((symInt32QConfig_t *)input->quantization->qConfig)->qMaxBits, &xMin, &xMax);
    symGroupedWeightRange(groupedWeight, &wMin, &wMax);
    size_t partialTerms = weightGroups->groupSize < nTerms ? weightGroups->groupSize : nTerms;
    symAccumSite_t partialSite = symAccumSiteInitRanges(
        "conv1dKernelSymInt32Grouped (raw group partial)", partialTerms, xMin, xMax, wMin, wMax, 0);
    symAccumSite_t accSite = symAccumSiteInitRangesRescaled(
        "conv1dKernelSymInt32Grouped (rescaled accumulator)", nTerms, xMin, xMax, wMin, wMax, 0);
    symAccumSiteSetSeeds(&accSite, seed, outChannels);
    if (symAccumProvenSafe(&accSite)) {
        conv1dSymGroupedLoopsSafe(&view, weightGroups, inScale, sAcc, wQC->roundingMode, rs,
                                  &partialSite, &accSite);
    } else {
        conv1dSymGroupedLoopsChecked(&view, weightGroups, inScale, sAcc, wQC->roundingMode, rs,
                                     &partialSite, &accSite);
    }

    if (output->quantization->qConfig == NULL) {
        PRINT_ERROR(
            "conv1dKernelSymInt32Grouped: output qConfig is NULL but SYM_INT32 expected (#187)");
        exit(1);
    }
    symInt32QConfig_t *outputQC = (symInt32QConfig_t *)output->quantization->qConfig;
    outputQC->scale = sAcc;
    outputQC->qMaxBits = symAccumResultBits(&accSite); /* the rescaled accumulator's width (#516) */
}

/* BFP epic PR2 (Task 4): conv1dKernelSymInt32Grouped's BFP twin (see
 * Conv1dKernel.h for the full contract). Differences from the SYM grouped
 * kernel: BOTH operands are group-tracked (not just the weight), the
 * boundary combine is a rounding-free ldexpf fold into a float accumulator
 * (never rescaleIntoAccumulatorScale), and the output is raw FLOAT32 (no
 * output-scale write -- the funnel epilogue owns packing). Since the
 * integer-only forward (conv1dKernelBfpRaw below) this float fold serves
 * the BACKWARD only: the ConvT1d dx adjoint keeps its FLOAT32 raw. */
void conv1dKernelBfp(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                     kernel_t const *kernel, size_t groups, tensor_t *output) {
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (input->quantization->type != BFP || weight->quantization->type != BFP) {
        PRINT_ERROR("conv1dKernelBfp: input and weight must be BFP (unpacked scratch form)");
        exit(1);
    }
    if (output->quantization->type != FLOAT32) {
        PRINT_ERROR("conv1dKernelBfp: output must be raw FLOAT32");
        exit(1);
    }
    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("conv1dKernelBfp: groups (%zu) must divide in_channels "
                    "(%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    bfpQConfig_t *inQC = input->quantization->qConfig;
    bfpQConfig_t *wQC = weight->quantization->qConfig;

    /* Group-shape fail-fast (validateSymQConfigShape precedent): bfpGroupOf
     * divides by groupSize with no relation to numGroups, so a mismatched
     * config would read exponents[] out of bounds. */
    validateBfpQConfigShape(inQC, calcNumberOfElementsByShape(input->shape));
    validateBfpQConfigShape(wQC, calcNumberOfElementsByShape(weight->shape));

    bfpQConfig_t *biasQC = NULL;
    if (bias != NULL) {
        if (bias->quantization->type != BFP) {
            PRINT_ERROR("conv1dKernelBfp: bias must be BFP");
            exit(1);
        }
        if (calcNumberOfElementsByShape(bias->shape) != outChannels) {
            PRINT_ERROR("conv1dKernelBfp: bias element count != out_channels");
            exit(1);
        }
        biasQC = bias->quantization->qConfig;
        validateBfpQConfigShape(biasQC, outChannels);
    }

    bfpValidateBlockHeadroom(inQC, wQC, inChPerGroup * kernelSize, "conv1dKernelBfp");

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("conv1dKernelBfp: output_length mismatch "
                    "(geometry=%zu, output tensor=%zu)",
                    geom.outputLength, outputLength);
        exit(1);
    }

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t const *wArr = (int32_t const *)weight->data;
    float *yArr = (float *)output->data;

    int32_t inExpBias = bfpExponentBias(inQC);
    int32_t wExpBias = bfpExponentBias(wQC);
    bfpGroupRange_t inRange = {0};
    bfpGroupRange_t wRange = {0};

    /* Per-output-channel bias seed, dequantized to float BEFORE the reduction
     * (value-seed, headroom-exempt -- it never touches the int32 partial).
     * VLA over channels (topology-bounded), mirroring the SYM kernels. */
    float seed[outChannels];
    if (bias != NULL) {
        int32_t const *bArr = (int32_t const *)bias->data;
        for (size_t oc = 0; oc < outChannels; oc++) {
            seed[oc] = (float)bArr[oc] * bfpGroupScale(biasQC, bfpGroupOf(biasQC, oc));
        }
    } else {
        for (size_t oc = 0; oc < outChannels; oc++) {
            seed[oc] = 0.0f;
        }
    }

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                size_t wBase = oc * inChPerGroup * kernelSize;

                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    float acc = seed[oc];
                    int32_t partial = 0;
                    size_t currentInGroup = 0;
                    size_t currentWGroup = SIZE_MAX;

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;
                            size_t wIdx = wBase + icOffset * kernelSize + kernelIdx;
                            size_t inIdx = (b * inChannels + ic) * inputLength + inputIdx;
                            /* Per-element group lookup on BOTH operands (see the
                             * SYM grouped kernel's gap rationale): clipped windows
                             * skip taps, so consecutive visited indices can have
                             * gaps on the weight AND -- across icOffset transitions
                             * -- jump by inputLength on the input. bfpGroupAt
                             * divides only when an index leaves the current
                             * group's range. */
                            size_t inGroup = bfpGroupAt(&inRange, inQC, inIdx);
                            size_t wGroup = bfpGroupAt(&wRange, wQC, wIdx);

                            if (currentWGroup == SIZE_MAX) {
                                currentInGroup = inGroup;
                                currentWGroup = wGroup;
                            } else if (inGroup != currentInGroup || wGroup != currentWGroup) {
                                /* Boundary fold on EITHER operand's group change:
                                 * the finished same-exponent segment's raw int32
                                 * partial enters the float accumulator via a pure
                                 * exponent shift -- rounding-free by contract. */
                                acc += ldexpf((float)partial,
                                              (int)inQC->exponents[currentInGroup] - inExpBias +
                                                  (int)wQC->exponents[currentWGroup] - wExpBias);
                                partial = 0;
                                currentInGroup = inGroup;
                                currentWGroup = wGroup;
                            }

                            partial += mulInt32s(xArr[inIdx], wArr[wIdx]);
                        }
                    }
                    /* Tail fold: the LAST segment never crosses a further
                     * boundary, so this is its only fold (and the ONLY fold
                     * when neither operand changes groups over the walk);
                     * empty windows never seed a segment at all. */
                    if (currentWGroup != SIZE_MAX) {
                        acc += ldexpf((float)partial,
                                      (int)inQC->exponents[currentInGroup] - inExpBias +
                                          (int)wQC->exponents[currentWGroup] - wExpBias);
                    }

                    yArr[(b * outChannels + oc) * outputLength + outPos] = acc;
                }
            }
        }
    }
}

/* ---- Integer-only forward (arithmetic-bfp.md §13): conv1dKernelBfpRaw ---- */

/* The row maxima of the pre-pass, read per output row (b, oc). */
typedef struct bfpRowTops {
    const int16_t *inMaxByBatchGroup; /* [batch * groups]: input rows of a conv group */
    const int16_t *wMax;              /* [outChannels]: the weight row */
    const int16_t *biasE;             /* [outChannels] or NULL */
    const uint16_t *convGroupOf;      /* [outChannels] */
    size_t groups;
} bfpRowTops_t;

static int bfpRowTop(const bfpRowTops_t *t, size_t b, size_t oc) {
    int e = t->inMaxByBatchGroup[b * t->groups + t->convGroupOf[oc]] + t->wMax[oc];
    if (t->biasE != NULL && t->biasE[oc] > e) {
        e = t->biasE[oc];
    }
    return e;
}

/* Contract in Conv1dKernel.h. The walk is conv1dKernelBfp's; only the fold
 * arithmetic and the output format differ. */
void conv1dKernelBfpRaw(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                        kernel_t const *kernel, size_t groups, tensor_t *output) {
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (input->quantization->type != BFP || weight->quantization->type != BFP) {
        PRINT_ERROR("conv1dKernelBfpRaw: input and weight must be BFP (unpacked scratch form)");
        exit(1);
    }
    if (output->quantization->type != BFP_RAW) {
        PRINT_ERROR("conv1dKernelBfpRaw: output must be BFP_RAW (the integer forward raw)");
        exit(1);
    }
    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("conv1dKernelBfpRaw: groups (%zu) must divide in_channels "
                    "(%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    bfpQConfig_t *inQC = input->quantization->qConfig;
    bfpQConfig_t *wQC = weight->quantization->qConfig;
    validateBfpQConfigShape(inQC, calcNumberOfElementsByShape(input->shape));
    validateBfpQConfigShape(wQC, calcNumberOfElementsByShape(weight->shape));

    bfpQConfig_t *biasQC = NULL;
    if (bias != NULL) {
        if (bias->quantization->type != BFP) {
            PRINT_ERROR("conv1dKernelBfpRaw: bias must be BFP");
            exit(1);
        }
        if (calcNumberOfElementsByShape(bias->shape) != outChannels) {
            PRINT_ERROR("conv1dKernelBfpRaw: bias element count != out_channels");
            exit(1);
        }
        biasQC = bias->quantization->qConfig;
        validateBfpQConfigShape(biasQC, outChannels);
    }

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("conv1dKernelBfpRaw: output_length mismatch "
                    "(geometry=%zu, output tensor=%zu)",
                    geom.outputLength, outputLength);
        exit(1);
    }

    size_t outElems = batch * outChannels * outputLength;
    bfpRawQConfig_t *rawQC = output->quantization->qConfig;
    size_t gsz = bfpRawRequireOutput(rawQC, outElems, "conv1dKernelBfpRaw");
    int16_t *rawE = rawQC->exponents;
    if (outElems == 0) {
        return;
    }

    /* Static headroom proof: g from the product bound over the nominal
     * reduction (padded taps are a superset of the visited ones); over-long
     * same-exponent runs split at the segment limit instead of failing. */
    size_t reductionLen = inChPerGroup * kernelSize;
    int guard = bfpFoldGuardBits(reductionLen, inQC->mantissaBits, wQC->mantissaBits,
                                 biasQC != NULL ? biasQC->mantissaBits : 0, "conv1dKernelBfpRaw");
    size_t segmentLimit = bfpSegmentLimit(inQC->mantissaBits, wQC->mantissaBits);

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t const *wArr = (int32_t const *)weight->data;
    int32_t const *bArr = bias != NULL ? (int32_t const *)bias->data : NULL;
    int32_t *mant = (int32_t *)output->data;

    int32_t inExpBias = bfpExponentBias(inQC);
    int32_t wExpBias = bfpExponentBias(wQC);

    /* Pre-pass (O(groups), stack tables sized by the topology, never by the
     * output): per (batch, conv group) the input rows' max exponent, per
     * output channel the weight row's max and the bias exponent. */
    int16_t inMaxByBatchGroup[batch * groups];
    int16_t wMax[outChannels];
    int16_t biasE[outChannels];
    uint16_t convGroupOf[outChannels];
    {
        bfpRowCursor_t c = {.group = 0, .groupEnd = inQC->groupSize};
        size_t row = 0;
        for (size_t b = 0; b < batch; b++) {
            for (size_t cg = 0; cg < groups; cg++) {
                int e = INT_MIN;
                for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                    int er = bfpRowMaxExponent(&c, inQC, inExpBias, row * inputLength, inputLength);
                    if (er > e) {
                        e = er;
                    }
                    row++;
                }
                inMaxByBatchGroup[b * groups + cg] = (int16_t)e;
            }
        }
    }
    {
        bfpRowCursor_t c = {.group = 0, .groupEnd = wQC->groupSize};
        bfpGroupRange_t biasRange = {0};
        size_t cg = 0;
        size_t cgEnd = outChPerGroup;
        for (size_t oc = 0; oc < outChannels; oc++) {
            if (oc >= cgEnd) {
                cg++;
                cgEnd += outChPerGroup;
            }
            convGroupOf[oc] = (uint16_t)cg;
            wMax[oc] =
                (int16_t)bfpRowMaxExponent(&c, wQC, wExpBias, oc * reductionLen, reductionLen);
            biasE[oc] = biasQC != NULL
                            ? (int16_t)((int)biasQC->exponents[bfpGroupAt(&biasRange, biasQC, oc)] -
                                        bfpExponentBias(biasQC))
                            : 0;
        }
    }
    bfpRowTops_t tops = {.inMaxByBatchGroup = inMaxByBatchGroup,
                         .wMax = wMax,
                         .biasE = biasQC != NULL ? biasE : NULL,
                         .convGroupOf = convGroupOf,
                         .groups = groups};

    /* E_acc per raw target group: the max row top over the output rows the
     * group covers (a group may straddle channels and batches), plus g. Rows
     * advance with cursors; the loop order below produces outputs in this
     * same flat order. */
    {
        size_t rowB = 0, rowOc = 0;
        size_t rowEnd = outputLength;
        for (size_t t = 0; t < rawQC->numGroups; t++) {
            size_t tStart = t * gsz;
            size_t tEnd = tStart + gsz;
            while (tStart >= rowEnd) {
                if (++rowOc == outChannels) {
                    rowOc = 0;
                    rowB++;
                }
                rowEnd += outputLength;
            }
            int eTop = bfpRowTop(&tops, rowB, rowOc);
            size_t rB = rowB, rOc = rowOc, rEnd = rowEnd;
            while (tEnd > rEnd) {
                if (++rOc == outChannels) {
                    rOc = 0;
                    rB++;
                }
                rEnd += outputLength;
                int e = bfpRowTop(&tops, rB, rOc);
                if (e > eTop) {
                    eTop = e;
                }
            }
            rawE[t] = (int16_t)(eTop + guard);
        }
    }

    bfpGroupRange_t inRange = {0};
    bfpGroupRange_t wRange = {0};
    size_t targetGroup = 0;
    size_t targetEnd = gsz;
    int eAcc = rawE[0];
    /* Bias as one more segment, hoisted: realigned only when the channel or
     * the target group's exponent changes. */
    int32_t biasSeed = 0;
    size_t seedOc = SIZE_MAX;
    int seedE = INT_MIN;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                size_t wBase = oc * inChPerGroup * kernelSize;

                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    size_t outIdx = (b * outChannels + oc) * outputLength + outPos;
                    if (outIdx >= targetEnd) {
                        targetGroup++;
                        targetEnd += gsz;
                        eAcc = rawE[targetGroup];
                    }
                    if (bArr != NULL && (oc != seedOc || eAcc != seedE)) {
                        biasSeed = bfpAlignPartial(bArr[oc], eAcc - biasE[oc]);
                        seedOc = oc;
                        seedE = eAcc;
                    }
                    int32_t acc = biasSeed;
                    int32_t partial = 0;
                    size_t segCount = 0;
                    size_t currentInGroup = 0;
                    size_t currentWGroup = SIZE_MAX;

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;
                            size_t wIdx = wBase + icOffset * kernelSize + kernelIdx;
                            size_t inIdx = (b * inChannels + ic) * inputLength + inputIdx;
                            size_t inGroup = bfpGroupAt(&inRange, inQC, inIdx);
                            size_t wGroup = bfpGroupAt(&wRange, wQC, wIdx);

                            if (currentWGroup == SIZE_MAX) {
                                currentInGroup = inGroup;
                                currentWGroup = wGroup;
                            } else if (inGroup != currentInGroup || wGroup != currentWGroup ||
                                       segCount == segmentLimit) {
                                /* A segment closes on EITHER operand's group
                                 * change or at the int32 limit (run split):
                                 * its partial is aligned to E_acc. */
                                acc += bfpAlignPartial(
                                    partial,
                                    eAcc - ((int)inQC->exponents[currentInGroup] - inExpBias +
                                            (int)wQC->exponents[currentWGroup] - wExpBias));
                                partial = 0;
                                segCount = 0;
                                currentInGroup = inGroup;
                                currentWGroup = wGroup;
                            }

                            partial += mulInt32s(xArr[inIdx], wArr[wIdx]);
                            segCount++;
                        }
                    }
                    if (currentWGroup != SIZE_MAX) {
                        acc += bfpAlignPartial(
                            partial, eAcc - ((int)inQC->exponents[currentInGroup] - inExpBias +
                                             (int)wQC->exponents[currentWGroup] - wExpBias));
                    }

                    mant[outIdx] = acc;
                }
            }
        }
    }
}
