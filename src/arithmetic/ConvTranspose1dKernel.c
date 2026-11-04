#define SOURCE_FILE "ODT_CONV_TRANSPOSE_1D_KERNEL"

#include "ConvTranspose1dKernel.h"

#include <math.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Mul.h"
#include "Rounding.h"
#include "SlidingWindow1d.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"

/* Shared VALID / adjoint-SAME(/EXPLICIT) geometry resolution for every ConvT
 * core (the scatter kernels AND the BFP gather): in VALID mode the output
 * length is pinned to convTranspose1dOutputLength ((inputLength-1)*stride +
 * dilation*(K-1) + outputPadding + 1) and outputPadding must stay below
 * max(stride, dilation) (PyTorch convention). SAME and EXPLICIT share the
 * adjoint path: the input-gradient of a forward Conv1d is a transposed conv
 * that scatters back through the forward's left padding; padLeft is recovered
 * from the forward-conv1d geometry on the adjoint OUTPUT length (= forward
 * input len), whose forward output len must equal the adjoint input len. For
 * EXPLICIT, windowGeometry1dCalc reports padLeft == kernel->padding; for SAME
 * the minimal {floor,ceil} split. outputPadding must be 0 there. Returns
 * padLeft; every violation exits with the calling kernel's name prefixed. */
static size_t convT1dResolveGeometry(kernel_t const *kernel, size_t inputLength,
                                     size_t outputLength, size_t outputPadding, const char *what) {
    size_t padLeft = 0;

    if (kernel->paddingType == VALID) {
        size_t expectedOutLen = convTranspose1dOutputLength(inputLength, kernel, outputPadding);
        if (expectedOutLen != outputLength) {
            PRINT_ERROR("%s: VALID output_length mismatch (expected=%zu, got=%zu)", what,
                        expectedOutLen, outputLength);
            exit(1);
        }

        if (outputPadding != 0 &&
            outputPadding >=
                ((kernel->stride > kernel->dilation) ? kernel->stride : kernel->dilation)) {
            PRINT_ERROR("%s: outputPadding (%zu) must be < max(stride=%zu, dilation=%zu)", what,
                        outputPadding, kernel->stride, kernel->dilation);
            exit(1);
        }
    } else if (kernel->paddingType == SAME || kernel->paddingType == EXPLICIT) {
        if (outputPadding != 0) {
            PRINT_ERROR("%s: outputPadding must be 0 in SAME/EXPLICIT mode (was %zu)", what,
                        outputPadding);
            exit(1);
        }
        windowGeometry1d_t fwdGeom = windowGeometry1dCalc(outputLength, kernel);
        if (fwdGeom.outputLength != inputLength) {
            PRINT_ERROR("%s: SAME/EXPLICIT adjoint input length (%zu) does not match forward "
                        "conv1d output length on the given output shape (%zu, fwd-out=%zu)",
                        what, inputLength, outputLength, fwdGeom.outputLength);
            exit(1);
        }
        padLeft = fwdGeom.padLeft;
    } else {
        PRINT_ERROR("%s: unsupported paddingType %d", what, (int)kernel->paddingType);
        exit(1);
    }
    return padLeft;
}

void convTranspose1dKernelFloat32(tensor_t const *input, tensor_t const *weight,
                                  tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                  size_t outputPadding, tensor_t *output) {
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("convTranspose1dKernelFloat32: groups (%zu) must divide "
                    "in_channels (%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t padLeft = convT1dResolveGeometry(kernel, inputLength, outputLength, outputPadding,
                                            "convTranspose1dKernelFloat32");

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    float const *xArr = (float const *)input->data;
    float const *wArr = (float const *)weight->data;
    float const *bArr = bias ? (float const *)bias->data : NULL;
    float *yArr = (float *)output->data;

    // Zero output (scatter loop accumulates with +=)
    size_t totalOut = batch * outChannels * outputLength;
    for (size_t i = 0; i < totalOut; i++) {
        yArr[i] = 0.0f;
    }

    long long padLeftSigned = (long long)padLeft;
    long long outputLengthSigned = (long long)outputLength;
    long long dilation = (long long)kernel->dilation;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    float xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    long long outBase = (long long)(inPos * kernel->stride) - padLeftSigned;

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        for (size_t k = 0; k < kernelSize; k++) {
                            long long outIdx = outBase + (long long)k * dilation;
                            if (outIdx < 0 || outIdx >= outputLengthSigned) {
                                continue;
                            }

                            float wv = wArr[(ic * outChPerGroup + ocOffset) * kernelSize + k];
                            yArr[(b * outChannels + oc) * outputLength + (size_t)outIdx] += xv * wv;
                        }
                    }
                }
            }
        }
    }

    // Bias add (separate pass; keeps the scatter loop a pure +=)
    if (bArr) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t oc = 0; oc < outChannels; oc++) {
                for (size_t l = 0; l < outputLength; l++) {
                    yArr[(b * outChannels + oc) * outputLength + l] += bArr[oc];
                }
            }
        }
    }
}

/* Shape and operand views shared by the SYM scatter loop nests below. */
typedef struct convTSymView {
    size_t batch, inChannels, inputLength, outChannels, outputLength, kernelSize;
    size_t groups, inChPerGroup, outChPerGroup, padLeft;
    const kernel_t *kernel;
    const int32_t *xArr;
    const int32_t *wArr;
    int32_t *yArr;
} convTSymView_t;

#define CONVT_SYM_VIEW_LOCALS(v)                                                                   \
    const size_t batch = (v)->batch, groups = (v)->groups, inChPerGroup = (v)->inChPerGroup;       \
    const size_t outChPerGroup = (v)->outChPerGroup, outChannels = (v)->outChannels;               \
    const size_t outputLength = (v)->outputLength, inChannels = (v)->inChannels;                   \
    const size_t inputLength = (v)->inputLength, kernelSize = (v)->kernelSize;                     \
    const size_t padLeft = (v)->padLeft;                                                           \
    kernel_t const *kernel = (v)->kernel;                                                          \
    const int32_t *xArr = (v)->xArr, *wArr = (v)->wArr;                                            \
    int32_t *yArr = (v)->yArr

/* Post-scatter bias seed pass, shared by both SYM scatter kernels. ALWAYS
 * checked: it is O(outputs), off the hot loop, and the scatter bound proves
 * only the products -- the seed can still tip a proven-safe scatter over, so
 * each add reports the seed it was adding. The output carries raw accumulator
 * codes that can exceed the operands' width, so it declares the width the
 * guard proved (#516) INCLUDING the seeds this pass added: they land after
 * the seedless scatter, so their signed range over the channels is written
 * into `resultSite` (a copy of the scatter site) for the stamp. The seeds
 * added here are the only correct ones (a second rescale under a stochastic
 * rounding mode would draw fresh randoms and could land on different values). */
static void convTAddBiasSeedsChecked(const convTSymView_t *v, tensor_t const *bias, float scale,
                                     rngStream_t *rs, const symAccumSite_t *scatterSite,
                                     symAccumSite_t *resultSite) {
    int32_t const *bArr = (int32_t const *)bias->data;
    symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
    int32_t seedMin = 0, seedMax = 0;
    for (size_t oc = 0; oc < v->outChannels; oc++) {
        int32_t seed =
            rescaleIntoAccumulatorScale(bArr[oc], biasQC->scale, scale, biasQC->roundingMode, rs);
        seedMin = (oc == 0 || seed < seedMin) ? seed : seedMin;
        seedMax = (oc == 0 || seed > seedMax) ? seed : seedMax;
        symAccumSite_t biasSite = *scatterSite;
        symAccumSiteSetSeeds(&biasSite, &seed, 1);
        for (size_t b = 0; b < v->batch; b++) {
            for (size_t l = 0; l < v->outputLength; l++) {
                size_t yIdx = (b * v->outChannels + oc) * v->outputLength + l;
                v->yArr[yIdx] = symAccumAddChecked(v->yArr[yIdx], seed, &biasSite);
            }
        }
    }
    symAccumSiteSetSeedRange(resultSite, seedMin, seedMax);
}

/* Instantiated twice by convTranspose1dKernelSymInt32 with `checked` a
 * compile-time constant: false is the historical scatter (+= mulInt32s), true
 * routes every accumulate through the SYM accumulator guard (SymAccumGuard.h). */
SYM_ACCUM_ALWAYS_INLINE void convTSymScatter(const convTSymView_t *v, const bool checked,
                                             const symAccumSite_t *site) {
    CONVT_SYM_VIEW_LOCALS(v);

    long long padLeftSigned = (long long)padLeft;
    long long outputLengthSigned = (long long)outputLength;
    long long dilation = (long long)kernel->dilation;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    int32_t xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    long long outBase = (long long)(inPos * kernel->stride) - padLeftSigned;

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        for (size_t k = 0; k < kernelSize; k++) {
                            long long outIdx = outBase + (long long)k * dilation;
                            if (outIdx < 0 || outIdx >= outputLengthSigned) {
                                continue;
                            }
                            int32_t wv = wArr[(ic * outChPerGroup + ocOffset) * kernelSize + k];
                            size_t yIdx = (b * outChannels + oc) * outputLength + (size_t)outIdx;
                            yArr[yIdx] = symMac(yArr[yIdx], xv, wv, checked, site);
                        }
                    }
                }
            }
        }
    }
}

SYM_ACCUM_NOINLINE void convTSymScatterSafe(const convTSymView_t *v, const symAccumSite_t *site) {
    convTSymScatter(v, false, site);
}

SYM_ACCUM_NOINLINE void convTSymScatterChecked(const convTSymView_t *v,
                                               const symAccumSite_t *site) {
    convTSymScatter(v, true, site);
}

/* Every operand is read through symInt32QConfig_t / as int32 codes, so a
 * different storage dtype must abort here instead of being reinterpreted. The
 * grouped entries pass the weight as a SYM_INT32 view too (its per-group
 * qConfig rides in weightGroups), so the same check holds for them. */
static void convT1dValidateSymOperand(const tensor_t *t, const char *kernelName, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("%s: %s must be SYM_INT32", kernelName, what);
        exit(1);
    }
}

static void convT1dValidateSymOperands(const tensor_t *input, const tensor_t *weight,
                                       const tensor_t *bias, const tensor_t *output,
                                       const char *kernelName) {
    convT1dValidateSymOperand(input, kernelName, "input");
    convT1dValidateSymOperand(weight, kernelName, "weight");
    if (bias != NULL) {
        convT1dValidateSymOperand(bias, kernelName, "bias");
    }
    convT1dValidateSymOperand(output, kernelName, "output");
}

void convTranspose1dKernelSymInt32(tensor_t const *input, tensor_t const *weight,
                                   tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                   size_t outputPadding, tensor_t *output, rngStream_t *rs) {
    convT1dValidateSymOperands(input, weight, bias, output, "convTranspose1dKernelSymInt32");
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("convTranspose1dKernelSymInt32: groups (%zu) must divide "
                    "in_channels (%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t padLeft = convT1dResolveGeometry(kernel, inputLength, outputLength, outputPadding,
                                            "convTranspose1dKernelSymInt32");

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t const *wArr = (int32_t const *)weight->data;
    int32_t *yArr = (int32_t *)output->data;

    float inScale = ((symInt32QConfig_t *)input->quantization->qConfig)->scale;
    float wScale = ((symInt32QConfig_t *)weight->quantization->qConfig)->scale;
    float outputScale = inScale * wScale;

    size_t totalOut = batch * outChannels * outputLength;
    for (size_t i = 0; i < totalOut; i++) {
        yArr[i] = 0;
    }

    convTSymView_t view = {.batch = batch,
                           .inChannels = inChannels,
                           .inputLength = inputLength,
                           .outChannels = outChannels,
                           .outputLength = outputLength,
                           .kernelSize = kernelSize,
                           .groups = groups,
                           .inChPerGroup = inChPerGroup,
                           .outChPerGroup = outChPerGroup,
                           .padLeft = padLeft,
                           .kernel = kernel,
                           .xArr = xArr,
                           .wArr = wArr,
                           .yArr = yArr};
    /* Scatter: at most inChPerGroup * kernelSize products reach one output
     * element. The bias seed lands afterwards (checked pass below). */
    uint8_t xBits = ((symInt32QConfig_t *)input->quantization->qConfig)->qMaxBits;
    uint8_t wBits = ((symInt32QConfig_t *)weight->quantization->qConfig)->qMaxBits;
    symAccumSite_t site = symAccumSiteInit("convTranspose1dKernelSymInt32",
                                           inChPerGroup * kernelSize, xBits, wBits, 0);
    if (symAccumProvenSafe(&site)) {
        convTSymScatterSafe(&view, &site);
    } else {
        convTSymScatterChecked(&view, &site);
    }

    /* Bias seed pass (refold), separate from the pure-+= scatter. NULL for Conv1d dx;
     * exercised by Conv1dTransposed forward in PR3. The stamp counts its seeds. */
    symAccumSite_t resultSite = site;
    if (bias != NULL) {
        convTAddBiasSeedsChecked(&view, bias, outputScale, rs, &site, &resultSite);
    }

    if (output->quantization->qConfig == NULL) {
        PRINT_ERROR("convTranspose1dKernelSymInt32: output qConfig is NULL but SYM_INT32 "
                    "expected (#187)");
        exit(1);
    }
    symInt32QConfig_t *outputQC = (symInt32QConfig_t *)output->quantization->qConfig;
    outputQC->scale = outputScale;
    outputQC->qMaxBits = symAccumResultBits(&resultSite);
}

/* Group-quant PR3 (Task 2): entry guards mirror conv1dValidateWeightGroups
 * (Conv1dKernel.c) exactly -- non-NULL and the per-tensor sentinel
 * {numGroups=1, groupSize=0} rejected (its groupSize=0 would divide-by-zero
 * below; per-tensor weights take
 * convTranspose1dKernelSymInt32). */
static void convT1dValidateWeightGroups(const symGroupedWeight_t *groupedWeight,
                                        size_t weightElemCount) {
    if (groupedWeight == NULL || groupedWeight->groups == NULL) {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: weightGroups must not be NULL");
        exit(1);
    }
    const symQConfig_t *weightGroups = groupedWeight->groups;
    if (weightGroups->numGroups <= 1) {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: weightGroups must be grouped "
                    "(numGroups>1); per-tensor weights take convTranspose1dKernelSymInt32");
        exit(1);
    }
    validateSymQConfigShape(weightGroups, weightElemCount);
}

/* Grouped twin of convTSymScatter: each product is rescaled into sAcc (factor
 * <= 1, so |rescaled| <= |product| up to the float32 evaluation slack) before it
 * joins the output element, hence the N * 2^(qa-1) * 2^(qb-1) bound with that
 * slack (symAccumTermLimitRescaled) covers the int32 += (and, on the checked
 * path, the product itself for wide widths). */
SYM_ACCUM_ALWAYS_INLINE void convTSymGroupedScatter(const convTSymView_t *v,
                                                    const symQConfig_t *weightGroups, float inScale,
                                                    float sAcc, roundingMode_t roundingMode,
                                                    rngStream_t *rs, const bool checked,
                                                    const symAccumSite_t *site) {
    CONVT_SYM_VIEW_LOCALS(v);

    long long padLeftSigned = (long long)padLeft;
    long long outputLengthSigned = (long long)outputLength;
    long long dilation = (long long)kernel->dilation;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    int32_t xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    long long outBase = (long long)(inPos * kernel->stride) - padLeftSigned;

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        for (size_t k = 0; k < kernelSize; k++) {
                            long long outIdx = outBase + (long long)k * dilation;
                            if (outIdx < 0 || outIdx >= outputLengthSigned) {
                                continue;
                            }
                            size_t wStorageIdx = (ic * outChPerGroup + ocOffset) * kernelSize + k;
                            /* Per-PRODUCT rescale into sAcc -- NOT the gather
                             * cores' running group-partial: consecutive
                             * products here land in DIFFERENT output elements
                             * (outIdx moves with k), so there is no
                             * per-(target, group) run across which a raw int32
                             * partial could be carried without a second
                             * accumulator buffer. Cost: one rounding per
                             * product, |err| <= 0.5*C*sAcc per output element
                             * (C = contributing products). Per-element group
                             * division (gap-robust): padding clips k taps, so
                             * visited wStorageIdx values can have gaps -- a
                             * precomputed run length would overshoot. */
                            size_t elemGroup = wStorageIdx / weightGroups->groupSize;
                            int32_t product = symProduct(xv, wArr[wStorageIdx], checked, site);
                            size_t yIdx = (b * outChannels + oc) * outputLength + (size_t)outIdx;
                            yArr[yIdx] =
                                symAdd(yArr[yIdx],
                                       rescalePartialIntoAccumulatorScale(
                                           product, inScale * weightGroups->scales[elemGroup], sAcc,
                                           roundingMode, rs),
                                       checked, site);
                        }
                    }
                }
            }
        }
    }
}

#define CONVT_GROUPED_PARAMS                                                                       \
    const convTSymView_t *v, const symQConfig_t *weightGroups, float inScale, float sAcc,          \
        roundingMode_t roundingMode, rngStream_t *rs, const symAccumSite_t *site
#define CONVT_GROUPED_ARGS(checked) v, weightGroups, inScale, sAcc, roundingMode, rs, checked, site

SYM_ACCUM_NOINLINE void convTSymGroupedScatterSafe(CONVT_GROUPED_PARAMS) {
    convTSymGroupedScatter(CONVT_GROUPED_ARGS(false));
}

SYM_ACCUM_NOINLINE void convTSymGroupedScatterChecked(CONVT_GROUPED_PARAMS) {
    convTSymGroupedScatter(CONVT_GROUPED_ARGS(true));
}

void convTranspose1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                          tensor_t const *bias, kernel_t const *kernel,
                                          size_t groups, size_t outputPadding, tensor_t *output,
                                          const symGroupedWeight_t *groupedWeight,
                                          rngStream_t *rs) {
    convT1dValidateSymOperands(input, weight, bias, output, "convTranspose1dKernelSymInt32Grouped");
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: groups (%zu) must divide "
                    "in_channels (%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t padLeft = convT1dResolveGeometry(kernel, inputLength, outputLength, outputPadding,
                                            "convTranspose1dKernelSymInt32Grouped");

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    convT1dValidateWeightGroups(groupedWeight, inChannels * outChPerGroup * kernelSize);
    const symQConfig_t *weightGroups = groupedWeight->groups;

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t const *wArr = (int32_t const *)weight->data;
    int32_t *yArr = (int32_t *)output->data;

    float inScale = ((symInt32QConfig_t *)input->quantization->qConfig)->scale;

    /* sAcc = inScale * max_g(weightGroups->scales[g]) (GGUF pattern, #189):
     * a single linear pass over scales[], NEVER scales[0] alone -- same
     * derivation as conv1dKernelSymInt32Grouped / matmul's grouped core. */
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
     * plumbing the gather cores read via their `wQC`/`bQC`. */
    symInt32QConfig_t *wQC = (symInt32QConfig_t *)weight->quantization->qConfig;

    size_t totalOut = batch * outChannels * outputLength;
    for (size_t i = 0; i < totalOut; i++) {
        yArr[i] = 0;
    }

    convTSymView_t view = {.batch = batch,
                           .inChannels = inChannels,
                           .inputLength = inputLength,
                           .outChannels = outChannels,
                           .outputLength = outputLength,
                           .kernelSize = kernelSize,
                           .groups = groups,
                           .inChPerGroup = inChPerGroup,
                           .outChPerGroup = outChPerGroup,
                           .padLeft = padLeft,
                           .kernel = kernel,
                           .xArr = xArr,
                           .wArr = wArr,
                           .yArr = yArr};
    /* The weight's operand range is its code domain's (SymGroupedWeight.h): a
     * stored SYM weight's two's-complement qBits range, a stored ASYM weight's
     * zp-shifted image +-(2^qBits - 1) -- the scratch's declared qMaxBits is
     * not read. The abort banner reports each range's holding width (the
     * image's is qBits + 1). */
    int32_t xMin, xMax, wMin, wMax;
    symAccumWidthRange(((symInt32QConfig_t *)input->quantization->qConfig)->qMaxBits, &xMin, &xMax);
    symGroupedWeightRange(groupedWeight, &wMin, &wMax);
    symAccumSite_t site =
        symAccumSiteInitRangesRescaled("convTranspose1dKernelSymInt32Grouped",
                                       inChPerGroup * kernelSize, xMin, xMax, wMin, wMax, 0);
    if (symAccumProvenSafe(&site)) {
        convTSymGroupedScatterSafe(&view, weightGroups, inScale, sAcc, wQC->roundingMode, rs,
                                   &site);
    } else {
        convTSymGroupedScatterChecked(&view, weightGroups, inScale, sAcc, wQC->roundingMode, rs,
                                      &site);
    }

    /* Bias seed pass (refold into sAcc), separate from the pure-+= scatter --
     * same pass order and primitive as convTranspose1dKernelSymInt32, only
     * the target scale differs (sAcc instead of inScale*wScale). */
    symAccumSite_t resultSite = site;
    if (bias != NULL) {
        convTAddBiasSeedsChecked(&view, bias, sAcc, rs, &site, &resultSite);
    }

    if (output->quantization->qConfig == NULL) {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: output qConfig is NULL but "
                    "SYM_INT32 expected (#187)");
        exit(1);
    }
    symInt32QConfig_t *outputQC = (symInt32QConfig_t *)output->quantization->qConfig;
    outputQC->scale = sAcc;
    outputQC->qMaxBits = symAccumResultBits(&resultSite); /* rescaled site + seeds (#516) */
}

/* BFP epic PR2 (Task 5, D9): the ConvT1d forward under ARITH_BFP is GATHER-
 * formulated (see ConvTranspose1dKernel.h for the full contract) because the
 * scatter form above has no per-(target, group) run across which one raw
 * int32 block partial could be carried -- consecutive scatter products land
 * in DIFFERENT output elements. Enumerating each output element's
 * contributors via convTranspose1dTapsAt restores the Task 3/4 block-partial
 * contract; the SYM scatter cores stay untouched. */
void convTranspose1dKernelBfpGather(tensor_t const *input, tensor_t const *weight,
                                    tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                    size_t outputPadding, tensor_t *output) {
    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (input->quantization->type != BFP || weight->quantization->type != BFP) {
        PRINT_ERROR("convTranspose1dKernelBfpGather: input and weight must be BFP "
                    "(unpacked scratch form)");
        exit(1);
    }
    if (output->quantization->type != FLOAT32) {
        PRINT_ERROR("convTranspose1dKernelBfpGather: output must be raw FLOAT32");
        exit(1);
    }
    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("convTranspose1dKernelBfpGather: groups (%zu) must divide "
                    "in_channels (%zu) and out_channels (%zu)",
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
            PRINT_ERROR("convTranspose1dKernelBfpGather: bias must be BFP");
            exit(1);
        }
        if (calcNumberOfElementsByShape(bias->shape) != outChannels) {
            PRINT_ERROR("convTranspose1dKernelBfpGather: bias element count != out_channels");
            exit(1);
        }
        biasQC = bias->quantization->qConfig;
        validateBfpQConfigShape(biasQC, outChannels);
    }

    bfpValidateBlockHeadroom(inQC, wQC, inChPerGroup * kernelSize,
                             "convTranspose1dKernelBfpGather");

    size_t padLeft = convT1dResolveGeometry(kernel, inputLength, outputLength, outputPadding,
                                            "convTranspose1dKernelBfpGather");

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t const *wArr = (int32_t const *)weight->data;
    float *yArr = (float *)output->data;

    int32_t inExpBias = bfpExponentBias(inQC);
    int32_t wExpBias = bfpExponentBias(wQC);
    bfpGroupRange_t inRange = {0};
    bfpGroupRange_t wRange = {0};

    /* Per-output-channel bias seed, dequantized to float BEFORE the reduction
     * (value-seed, headroom-exempt -- it never touches the int32 partial).
     * Seeded INLINE per output element, not added in a separate pass like the
     * scatter cores' bias refold: the gather owns a per-element accumulator,
     * so a second pass would buy nothing. VLA over channels (topology-
     * bounded), mirroring conv1dKernelBfp. */
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

                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    /* Stack VLA, topology-bounded (allocation-locality rule);
                     * outputPadding tail positions get tapCount == 0 and stay
                     * at the bias seed -- exactly where the scatter never
                     * writes. */
                    convTransposeTap_t taps[kernelSize];
                    size_t tapCount =
                        convTranspose1dTapsAt(outPos, inputLength, kernelSize, kernel->stride,
                                              kernel->dilation, padLeft, taps);
                    float acc = seed[oc];
                    int32_t partial = 0;
                    size_t currentInGroup = 0;
                    size_t currentWGroup = SIZE_MAX;

                    /* Taps OUTER, icOffset INNER (the BFP D9 normative reduction
                     * order the gold emulation mirrors). Per-element group
                     * lookup on BOTH operands: tap hops make both index
                     * sequences non-contiguous, so no run length could be
                     * precomputed (the SYM grouped kernels' gap rationale);
                     * bfpGroupAt divides only when an index leaves the current
                     * group's range. */
                    for (size_t t = 0; t < tapCount; t++) {
                        for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                            size_t ic = inLo + icOffset;
                            size_t inIdx = (b * inChannels + ic) * inputLength + taps[t].inPos;
                            size_t wIdx =
                                (ic * outChPerGroup + ocOffset) * kernelSize + taps[t].kernelIdx;
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
                                acc += bfpLdexpInt32(
                                    partial, (int)inQC->exponents[currentInGroup] - inExpBias +
                                                 (int)wQC->exponents[currentWGroup] - wExpBias);
                                partial = 0;
                                currentInGroup = inGroup;
                                currentWGroup = wGroup;
                            }

                            partial += mulInt32s(xArr[inIdx], wArr[wIdx]);
                        }
                    }
                    /* Tail fold: the LAST segment never crosses a further
                     * boundary, so this is its only fold; tap-free positions
                     * never seed a segment at all. */
                    if (currentWGroup != SIZE_MAX) {
                        acc += bfpLdexpInt32(partial,
                                             (int)inQC->exponents[currentInGroup] - inExpBias +
                                                 (int)wQC->exponents[currentWGroup] - wExpBias);
                    }

                    yArr[(b * outChannels + oc) * outputLength + outPos] = acc;
                }
            }
        }
    }
}
