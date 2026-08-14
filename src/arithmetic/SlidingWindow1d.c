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

windowSlice1d_t windowSlice1dAt(windowGeometry1d_t const *geometry, size_t outputPos) {
    windowSlice1d_t s = {0};
    // signed math because input start can go negative when padLeft > 0
    long long inputStart = (long long)(outputPos * geometry->stride) - (long long)geometry->padLeft;
    long long dilation = (long long)geometry->dilation;
    long long inputLen = (long long)geometry->inputLength;

    // Right-side OOB: window starts past the end of input. C's integer division
    // truncates toward zero (not floor), so without this guard a small-magnitude
    // negative numerator below would round up to 0 and miss the empty-window
    // sentinel. Catch it explicitly.
    if (inputStart >= inputLen) {
        s.firstValidInputIdx = 0;
        s.firstValidKernelOffset = geometry->kernelSize;
        s.validCount = 0;
        return s;
    }

    // first kernel offset k where (inputStart + k*dilation) >= 0
    long long firstK = 0;
    if (inputStart < 0) {
        // ceil((-inputStart) / dilation)
        firstK = (-inputStart + dilation - 1) / dilation;
    }

    // last kernel offset k where (inputStart + k*dilation) < inputLen
    long long lastK = (long long)geometry->kernelSize - 1;
    long long maxKByInputLen = (inputLen - 1 - inputStart) / dilation; // floor
    if (maxKByInputLen < lastK) {
        lastK = maxKByInputLen;
    }

    if (firstK > lastK) {
        // empty window — sentinel firstValidKernelOffset == kernelSize lets caller skip
        s.firstValidInputIdx = 0;
        s.firstValidKernelOffset = geometry->kernelSize;
        s.validCount = 0;
        return s;
    }

    s.firstValidKernelOffset = (size_t)firstK;
    s.firstValidInputIdx = (size_t)(inputStart + firstK * dilation);
    s.validCount = (size_t)(lastK - firstK + 1);
    return s;
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
