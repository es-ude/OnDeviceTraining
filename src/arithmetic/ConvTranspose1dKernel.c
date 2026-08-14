#define SOURCE_FILE "ODT_CONV_TRANSPOSE_1D_KERNEL"

#include "ConvTranspose1dKernel.h"

#include <math.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Mul.h"
#include "Rounding.h"
#include "SlidingWindow1d.h"

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

void convTranspose1dKernelSymInt32(tensor_t const *input, tensor_t const *weight,
                                   tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                   size_t outputPadding, tensor_t *output) {
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
                            yArr[(b * outChannels + oc) * outputLength + (size_t)outIdx] +=
                                mulInt32s(xv, wv);
                        }
                    }
                }
            }
        }
    }

    /* Bias seed pass (refold), separate from the pure-+= scatter. NULL for Conv1d dx;
     * exercised by Conv1dTransposed forward in PR3. */
    if (bias != NULL) {
        int32_t const *bArr = (int32_t const *)bias->data;
        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        for (size_t oc = 0; oc < outChannels; oc++) {
            int32_t seed = rescaleIntoAccumulatorScale(bArr[oc], biasQC->scale, outputScale,
                                                       biasQC->roundingMode);
            for (size_t b = 0; b < batch; b++) {
                for (size_t l = 0; l < outputLength; l++) {
                    yArr[(b * outChannels + oc) * outputLength + l] += seed;
                }
            }
        }
    }

    if (output->quantization->qConfig == NULL) {
        PRINT_ERROR("convTranspose1dKernelSymInt32: output qConfig is NULL but SYM_INT32 "
                    "expected (#187)");
        exit(1);
    }
    ((symInt32QConfig_t *)output->quantization->qConfig)->scale = outputScale;
}

/* Group-quant PR3 (Task 2): entry guards mirror conv1dValidateWeightGroups
 * (Conv1dKernel.c) exactly -- non-NULL, qBits within the operand contract,
 * and the per-tensor sentinel {numGroups=1, groupSize=0} rejected (its
 * groupSize=0 would divide-by-zero below; per-tensor weights take
 * convTranspose1dKernelSymInt32). */
static void convT1dValidateWeightGroups(const symQConfig_t *weightGroups, size_t weightElemCount) {
    if (weightGroups == NULL) {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: weightGroups must not be NULL");
        exit(1);
    }
    if (weightGroups->qBits > ODT_SYM_OPERAND_QMAXBITS) {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: weightGroups qBits (%u) exceeds "
                    "operand contract (%u) — int32 product accumulation would overflow (#227)",
                    (unsigned)weightGroups->qBits, (unsigned)ODT_SYM_OPERAND_QMAXBITS);
        exit(1);
    }
    if (weightGroups->numGroups <= 1) {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: weightGroups must be grouped "
                    "(numGroups>1); per-tensor weights take convTranspose1dKernelSymInt32");
        exit(1);
    }
    validateSymQConfigShape(weightGroups, weightElemCount);
}

void convTranspose1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                          tensor_t const *bias, kernel_t const *kernel,
                                          size_t groups, size_t outputPadding, tensor_t *output,
                                          const symQConfig_t *weightGroups) {
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

    convT1dValidateWeightGroups(weightGroups, inChannels * outChPerGroup * kernelSize);

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
                            int32_t product = mulInt32s(xv, wArr[wStorageIdx]);
                            yArr[(b * outChannels + oc) * outputLength + (size_t)outIdx] +=
                                rescaleIntoAccumulatorScale(
                                    product, inScale * weightGroups->scales[elemGroup], sAcc,
                                    wQC->roundingMode);
                        }
                    }
                }
            }
        }
    }

    /* Bias seed pass (refold into sAcc), separate from the pure-+= scatter --
     * same pass order and primitive as convTranspose1dKernelSymInt32, only
     * the target scale differs (sAcc instead of inScale*wScale). */
    if (bias != NULL) {
        int32_t const *bArr = (int32_t const *)bias->data;
        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        for (size_t oc = 0; oc < outChannels; oc++) {
            int32_t seed =
                rescaleIntoAccumulatorScale(bArr[oc], biasQC->scale, sAcc, biasQC->roundingMode);
            for (size_t b = 0; b < batch; b++) {
                for (size_t l = 0; l < outputLength; l++) {
                    yArr[(b * outChannels + oc) * outputLength + l] += seed;
                }
            }
        }
    }

    if (output->quantization->qConfig == NULL) {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: output qConfig is NULL but "
                    "SYM_INT32 expected (#187)");
        exit(1);
    }
    ((symInt32QConfig_t *)output->quantization->qConfig)->scale = sAcc;
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

                    /* Taps OUTER, icOffset INNER (the D9 normative reduction
                     * order the gold emulation mirrors). Per-element group
                     * lookup on BOTH operands: tap hops make both index
                     * sequences non-contiguous, so no run length could be
                     * precomputed (the SYM grouped kernels' gap rationale). */
                    for (size_t t = 0; t < tapCount; t++) {
                        for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                            size_t ic = inLo + icOffset;
                            size_t inIdx = (b * inChannels + ic) * inputLength + taps[t].inPos;
                            size_t wIdx =
                                (ic * outChPerGroup + ocOffset) * kernelSize + taps[t].kernelIdx;
                            size_t inGroup = bfpGroupOf(inQC, inIdx);
                            size_t wGroup = bfpGroupOf(wQC, wIdx);

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
                     * boundary, so this is its only fold; tap-free positions
                     * never seed a segment at all. */
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
