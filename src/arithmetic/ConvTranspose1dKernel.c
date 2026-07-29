#define SOURCE_FILE "ODT_CONV_TRANSPOSE_1D_KERNEL"

#include "ConvTranspose1dKernel.h"

#include "Common.h"
#include "Mul.h"
#include "Rounding.h"
#include "SlidingWindow1d.h"

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

    // Geometry: in VALID mode the adjoint output_length is determined by
    // (inputLength-1)*stride + dilation*(K-1) + outputPadding + 1.
    // In SAME mode the adjoint is the inverse of a forward Conv1d that took
    // an input of length outputLength (caller's propLoss target) and produced
    // an output of length inputLength. windowGeometry1dCalc takes the forward
    // input length and returns padLeft/padRight implicitly.
    size_t padLeft = 0;

    if (kernel->paddingType == VALID) {
        size_t expectedOutLen = convTranspose1dOutputLength(inputLength, kernel, outputPadding);
        if (expectedOutLen != outputLength) {
            PRINT_ERROR("convTranspose1dKernelFloat32: VALID output_length mismatch "
                        "(expected=%zu, got=%zu)",
                        expectedOutLen, outputLength);
            exit(1);
        }

        if (outputPadding != 0 &&
            outputPadding >=
                ((kernel->stride > kernel->dilation) ? kernel->stride : kernel->dilation)) {
            PRINT_ERROR("convTranspose1dKernelFloat32: outputPadding (%zu) must be "
                        "< max(stride=%zu, dilation=%zu)",
                        outputPadding, kernel->stride, kernel->dilation);
            exit(1);
        }
    } else if (kernel->paddingType == SAME || kernel->paddingType == EXPLICIT) {
        // SAME and EXPLICIT share the adjoint path. The input-gradient of a
        // forward Conv1d is a transposed conv that scatters back through the
        // forward's left padding; we recover padLeft from the forward-conv1d
        // geometry on the adjoint output length (= forward input len), whose
        // forward output len must equal the adjoint input len. For EXPLICIT,
        // windowGeometry1dCalc reports padLeft == kernel->padding; for SAME it
        // reports the minimal {floor,ceil} split. This branch is reached as the
        // Conv1d backward adjoint (conv1dBackward passes the forward kernel).
        if (outputPadding != 0) {
            PRINT_ERROR("convTranspose1dKernelFloat32: outputPadding must be 0 in "
                        "SAME/EXPLICIT mode (was %zu)",
                        outputPadding);
            exit(1);
        }
        windowGeometry1d_t fwdGeom = windowGeometry1dCalc(outputLength, kernel);
        if (fwdGeom.outputLength != inputLength) {
            PRINT_ERROR("convTranspose1dKernelFloat32: SAME/EXPLICIT adjoint input length "
                        "(%zu) does not match forward conv1d output length on the "
                        "given output shape (%zu, fwd-out=%zu)",
                        inputLength, outputLength, fwdGeom.outputLength);
            exit(1);
        }
        padLeft = fwdGeom.padLeft;
    } else {
        PRINT_ERROR("convTranspose1dKernelFloat32: unsupported paddingType %d",
                    (int)kernel->paddingType);
        exit(1);
    }

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

    size_t padLeft = 0;

    if (kernel->paddingType == VALID) {
        size_t expectedOutLen = convTranspose1dOutputLength(inputLength, kernel, outputPadding);
        if (expectedOutLen != outputLength) {
            PRINT_ERROR("convTranspose1dKernelSymInt32: VALID output_length mismatch "
                        "(expected=%zu, got=%zu)",
                        expectedOutLen, outputLength);
            exit(1);
        }
        if (outputPadding != 0 &&
            outputPadding >=
                ((kernel->stride > kernel->dilation) ? kernel->stride : kernel->dilation)) {
            PRINT_ERROR("convTranspose1dKernelSymInt32: outputPadding (%zu) must be "
                        "< max(stride=%zu, dilation=%zu)",
                        outputPadding, kernel->stride, kernel->dilation);
            exit(1);
        }
    } else if (kernel->paddingType == SAME || kernel->paddingType == EXPLICIT) {
        if (outputPadding != 0) {
            PRINT_ERROR("convTranspose1dKernelSymInt32: outputPadding must be 0 in "
                        "SAME/EXPLICIT mode (was %zu)",
                        outputPadding);
            exit(1);
        }
        windowGeometry1d_t fwdGeom = windowGeometry1dCalc(outputLength, kernel);
        if (fwdGeom.outputLength != inputLength) {
            PRINT_ERROR("convTranspose1dKernelSymInt32: SAME/EXPLICIT adjoint input length "
                        "(%zu) does not match forward conv1d output length on the "
                        "given output shape (%zu, fwd-out=%zu)",
                        inputLength, outputLength, fwdGeom.outputLength);
            exit(1);
        }
        padLeft = fwdGeom.padLeft;
    } else {
        PRINT_ERROR("convTranspose1dKernelSymInt32: unsupported paddingType %d",
                    (int)kernel->paddingType);
        exit(1);
    }

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

    size_t padLeft = 0;

    if (kernel->paddingType == VALID) {
        size_t expectedOutLen = convTranspose1dOutputLength(inputLength, kernel, outputPadding);
        if (expectedOutLen != outputLength) {
            PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: VALID output_length mismatch "
                        "(expected=%zu, got=%zu)",
                        expectedOutLen, outputLength);
            exit(1);
        }
        if (outputPadding != 0 &&
            outputPadding >=
                ((kernel->stride > kernel->dilation) ? kernel->stride : kernel->dilation)) {
            PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: outputPadding (%zu) must be "
                        "< max(stride=%zu, dilation=%zu)",
                        outputPadding, kernel->stride, kernel->dilation);
            exit(1);
        }
    } else if (kernel->paddingType == SAME || kernel->paddingType == EXPLICIT) {
        if (outputPadding != 0) {
            PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: outputPadding must be 0 in "
                        "SAME/EXPLICIT mode (was %zu)",
                        outputPadding);
            exit(1);
        }
        windowGeometry1d_t fwdGeom = windowGeometry1dCalc(outputLength, kernel);
        if (fwdGeom.outputLength != inputLength) {
            PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: SAME/EXPLICIT adjoint input "
                        "length (%zu) does not match forward conv1d output length on the "
                        "given output shape (%zu, fwd-out=%zu)",
                        inputLength, outputLength, fwdGeom.outputLength);
            exit(1);
        }
        padLeft = fwdGeom.padLeft;
    } else {
        PRINT_ERROR("convTranspose1dKernelSymInt32Grouped: unsupported paddingType %d",
                    (int)kernel->paddingType);
        exit(1);
    }

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
