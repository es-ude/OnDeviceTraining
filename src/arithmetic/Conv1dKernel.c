#define SOURCE_FILE "ODT_CONV1D_KERNEL"

#include "Conv1dKernel.h"

#include <math.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Mul.h"
#include "Rounding.h"
#include "SlidingWindow1d.h"

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

void conv1dKernelSymInt32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                          kernel_t const *kernel, size_t groups, tensor_t *output) {
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
                                                   biasQC->roundingMode);
        }
    } else {
        for (size_t oc = 0; oc < outChannels; oc++) {
            seed[oc] = 0;
        }
    }

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
                            sum += mulInt32s(xv, wv);
                        }
                    }

                    yArr[(b * outChannels + oc) * outputLength + outPos] = sum;
                }
            }
        }
    }

    if (output->quantization->qConfig == NULL) {
        PRINT_ERROR("conv1dKernelSymInt32: output qConfig is NULL but SYM_INT32 expected (#187)");
        exit(1);
    }
    ((symInt32QConfig_t *)output->quantization->qConfig)->scale = outputScale;
}

/* Group-quant PR2 (Task 4): entry guards mirror matmulValidateWeightGroups
 * (Matmul.c) exactly -- non-NULL, qBits within the operand contract, and the
 * per-tensor sentinel {numGroups=1, groupSize=0} rejected (its groupSize=0
 * would divide-by-zero below; per-tensor weights take conv1dKernelSymInt32). */
static void conv1dValidateWeightGroups(const symQConfig_t *weightGroups, size_t weightElemCount) {
    if (weightGroups == NULL) {
        PRINT_ERROR("conv1dKernelSymInt32Grouped: weightGroups must not be NULL");
        exit(1);
    }
    if (weightGroups->qBits > ODT_SYM_OPERAND_QMAXBITS) {
        PRINT_ERROR("conv1dKernelSymInt32Grouped: weightGroups qBits (%u) exceeds operand "
                    "contract (%u) — int32 product accumulation would overflow (#227)",
                    (unsigned)weightGroups->qBits, (unsigned)ODT_SYM_OPERAND_QMAXBITS);
        exit(1);
    }
    if (weightGroups->numGroups <= 1) {
        PRINT_ERROR("conv1dKernelSymInt32Grouped: weightGroups must be grouped (numGroups>1); "
                    "per-tensor weights take conv1dKernelSymInt32");
        exit(1);
    }
    validateSymQConfigShape(weightGroups, weightElemCount);
}

void conv1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                 tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                 tensor_t *output, const symQConfig_t *weightGroups) {
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

    conv1dValidateWeightGroups(weightGroups, outChannels * inChPerGroup * kernelSize);

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
            seed[oc] =
                rescaleIntoAccumulatorScale(bArr[oc], biasQC->scale, sAcc, biasQC->roundingMode);
        }
    } else {
        for (size_t oc = 0; oc < outChannels; oc++) {
            seed[oc] = 0;
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
                                    acc += rescaleIntoAccumulatorScale(
                                        partial, inScale * weightGroups->scales[currentGroup], sAcc,
                                        wQC->roundingMode);
                                    partial = 0;
                                }
                                currentGroup = elemGroup;
                            }

                            int32_t xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            int32_t wv = wArr[wStorageIdx];
                            partial += mulInt32s(xv, wv);
                        }
                    }
                    /* Tail combine: the LAST group visited never crosses a
                     * further boundary, so its partial only ever gets folded
                     * in here. Per-channel weights (groupSize == the full
                     * inChPerGroup*kernelSize reduction) never hit the
                     * mid-loop branch at all -- this is their ONLY weight
                     * combine. */
                    if (currentGroup != SIZE_MAX) {
                        acc += rescaleIntoAccumulatorScale(
                            partial, inScale * weightGroups->scales[currentGroup], sAcc,
                            wQC->roundingMode);
                    }

                    yArr[(b * outChannels + oc) * outputLength + outPos] = acc;
                }
            }
        }
    }

    if (output->quantization->qConfig == NULL) {
        PRINT_ERROR(
            "conv1dKernelSymInt32Grouped: output qConfig is NULL but SYM_INT32 expected (#187)");
        exit(1);
    }
    ((symInt32QConfig_t *)output->quantization->qConfig)->scale = sAcc;
}

/* BFP epic PR2 (Task 4): conv1dKernelSymInt32Grouped's BFP twin (see
 * Conv1dKernel.h for the full contract). Differences from the SYM grouped
 * kernel: BOTH operands are group-tracked (not just the weight), the
 * boundary combine is a rounding-free ldexpf fold into a float accumulator
 * (never rescaleIntoAccumulatorScale), and the output is raw FLOAT32 (no
 * output-scale write -- the funnel epilogue owns packing). */
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
                            /* Per-element division on BOTH operands (see the SYM
                             * grouped kernel's gap rationale): clipped windows skip
                             * taps, so consecutive visited indices can have gaps on
                             * the weight AND -- across icOffset transitions -- jump
                             * by inputLength on the input. */
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
