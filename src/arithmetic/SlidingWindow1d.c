#define SOURCE_FILE "ODT_SLIDING_WINDOW_1D"

#include "SlidingWindow1d.h"

windowGeometry1d_t windowGeometry1dCalc(size_t inputLength, kernel_t const *kernel) {
    windowGeometry1d_t g = {0};
    g.inputLength = inputLength;
    g.kernelSize = kernel->size;
    g.stride = kernel->stride;
    g.dilation = kernel->dilation;

    size_t effectiveKernel = kernel->dilation * (kernel->size - 1) + 1;

    switch (kernel->paddingType) {
    case VALID:
        g.padLeft = 0;
        g.padRight = 0;
        if (inputLength >= effectiveKernel) {
            g.outputLength = (inputLength - effectiveKernel) / kernel->stride + 1;
        } else {
            g.outputLength = 0;
        }
        break;
    case SAME: {
        // PyTorch padding="same": outputLength = ceil(inputLength / stride)
        g.outputLength = (inputLength + kernel->stride - 1) / kernel->stride;
        size_t needed = effectiveKernel + (g.outputLength - 1) * kernel->stride;
        size_t totalPad = (needed > inputLength) ? (needed - inputLength) : 0;
        g.padLeft = totalPad / 2;
        g.padRight = totalPad - g.padLeft;
        break;
    }
    case EXPLICIT: {
        // PyTorch padding=N: exactly `padding` zeros on EACH side (symmetric).
        // outputLength follows the standard conv formula on the padded input.
        g.padLeft = kernel->padding;
        g.padRight = kernel->padding;
        size_t paddedLength = inputLength + 2 * kernel->padding;
        if (paddedLength >= effectiveKernel) {
            g.outputLength = (paddedLength - effectiveKernel) / kernel->stride + 1;
        } else {
            g.outputLength = 0;
        }
        break;
    }
    }
    return g;
}

/* Unsigned, divide-free geometry. These run once per output position or
 * tap in every conv/pool kernel, so they avoid 64-bit math (a libgcc call on
 * Cortex-M) and skip the divide entirely for the common stride/dilation 1.
 * ceilDivSize uses (n - 1) / d + 1, not (n + d - 1) / d: the latter wraps once
 * n + d > SIZE_MAX (2^32 on the MCU); this form is exact for every n and d >= 1. */
static inline size_t ceilDivSize(size_t numerator, size_t denominator) {
    if (denominator == 1) {
        return numerator;
    }
    return (numerator == 0) ? 0 : (numerator - 1) / denominator + 1;
}

static inline size_t floorDivSize(size_t numerator, size_t denominator) {
    return (denominator == 1) ? numerator : numerator / denominator;
}

static windowSlice1d_t emptyWindowSlice(size_t kernelSize) {
    // sentinel firstValidKernelOffset == kernelSize lets the caller skip
    windowSlice1d_t s = {
        .firstValidInputIdx = 0, .firstValidKernelOffset = kernelSize, .validCount = 0};
    return s;
}

windowSlice1d_t windowSlice1dAt(windowGeometry1d_t const *geometry, size_t outputPos) {
    size_t const kernelSize = geometry->kernelSize;
    size_t const dilation = geometry->dilation;
    size_t const inputLength = geometry->inputLength;
    size_t const windowPos = outputPos * geometry->stride;

    // The window starts at windowPos - padLeft. firstK is the first kernel
    // offset whose input index is >= 0; firstIdx is that input index.
    size_t firstK = 0;
    size_t firstIdx;
    if (windowPos < geometry->padLeft) {
        size_t const neg = geometry->padLeft - windowPos;
        firstK = ceilDivSize(neg, dilation);
        if (firstK >= kernelSize) {
            return emptyWindowSlice(kernelSize);
        }
        firstIdx = firstK * dilation - neg;
    } else {
        firstIdx = windowPos - geometry->padLeft;
    }
    if (firstIdx >= inputLength) {
        return emptyWindowSlice(kernelSize);
    }

    // taps firstK .. firstK + validCount - 1 stay below inputLength
    size_t const reachable = floorDivSize(inputLength - 1 - firstIdx, dilation) + 1;
    size_t const remainingTaps = kernelSize - firstK;
    windowSlice1d_t s = {.firstValidInputIdx = firstIdx,
                         .firstValidKernelOffset = firstK,
                         .validCount = (reachable < remainingTaps) ? reachable : remainingTaps};
    return s;
}

void windowTapOutputRange1d(windowGeometry1d_t const *geometry, size_t k, size_t *outLo,
                            size_t *outHi) {
    size_t const stride = geometry->stride;
    size_t const inputLength = geometry->inputLength;
    size_t const tapPos = k * geometry->dilation;

    // Tap k of output outPos reads input outPos*stride + tapPos - padLeft.
    // lo: first outPos with that index >= 0; hi: first outPos with it >= inputLength.
    size_t lo = 0;
    size_t span; // inputLength - (tapPos - padLeft), clamped at 0
    if (tapPos < geometry->padLeft) {
        size_t const neg = geometry->padLeft - tapPos;
        lo = ceilDivSize(neg, stride);
        span = inputLength + neg;
    } else {
        size_t const offset = tapPos - geometry->padLeft;
        span = (offset < inputLength) ? inputLength - offset : 0;
    }
    size_t hi = ceilDivSize(span, stride);
    if (hi > geometry->outputLength) {
        hi = geometry->outputLength;
    }
    if (lo > hi) {
        lo = hi;
    }

    *outLo = lo;
    *outHi = hi;
}

void convTValidTapRange(size_t inPos, size_t stride, size_t dilation, size_t padLeft,
                        size_t outputLength, size_t kernelSize, size_t *kLo, size_t *kHi) {
    // Tap k writes output inPos*stride + k*dilation - padLeft; it is valid
    // while that index lies in [0, outputLength), i.e.
    // padLeft - base <= k*dilation < outputLength + padLeft - base.
    size_t const base = inPos * stride;
    size_t const limit = outputLength + padLeft;
    size_t lo = 0;
    size_t hi = 0;
    if (base < limit) {
        if (dilation == 0) {
            // every tap hits the same index base - padLeft
            hi = (base >= padLeft) ? kernelSize : 0;
        } else {
            if (base < padLeft) {
                lo = ceilDivSize(padLeft - base, dilation);
            }
            hi = ceilDivSize(limit - base, dilation);
            if (hi > kernelSize) {
                hi = kernelSize;
            }
            if (lo > hi) {
                lo = hi;
            }
        }
    }
    *kLo = lo;
    *kHi = hi;
}

size_t convTranspose1dTapsAt(size_t outPos, size_t inputLength, size_t kernelSize, size_t stride,
                             size_t dilation, size_t padLeft, convTransposeTap_t *taps) {
    size_t count = 0;
    size_t p = outPos + padLeft;
    for (size_t k = 0; k < kernelSize; k++) {
        size_t kd = k * dilation;
        if (kd > p) {
            break; /* k*dilation grows monotonically; later taps reach even further left */
        }
        size_t rem = p - kd;
        if (rem % stride != 0) {
            continue;
        }
        size_t inPos = rem / stride;
        if (inPos >= inputLength) {
            continue;
        }
        taps[count].inPos = inPos;
        taps[count].kernelIdx = k;
        count++;
    }
    return count;
}

size_t convTranspose1dOutputLength(size_t inputLength, kernel_t const *kernel,
                                   size_t outputPadding) {
    return (inputLength - 1) * kernel->stride + kernel->dilation * (kernel->size - 1) +
           outputPadding + 1;
}
