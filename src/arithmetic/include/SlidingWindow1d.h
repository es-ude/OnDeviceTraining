#ifndef ODT_SLIDING_WINDOW_1D_H
#define ODT_SLIDING_WINDOW_1D_H

#include <stdlib.h>

#include "Kernel.h"

typedef struct windowGeometry1d {
    size_t inputLength;
    size_t outputLength;
    size_t kernelSize;
    size_t stride;
    size_t dilation;
    size_t padLeft;
    size_t padRight;
} windowGeometry1d_t;

windowGeometry1d_t windowGeometry1dCalc(size_t inputLength, kernel_t const *kernel);

/*! VALID-mode transposed-conv forward output length:
 *    Lout = (inputLength - 1)*stride + dilation*(kernelSize - 1) + outputPadding + 1
 *  SAME/EXPLICIT transpose geometry
 *  is NOT this — it is recovered via windowGeometry1dCalc(outputLength, kernel). */
size_t convTranspose1dOutputLength(size_t inputLength, kernel_t const *kernel,
                                   size_t outputPadding);

typedef struct windowSlice1d {
    size_t firstValidInputIdx;
    size_t firstValidKernelOffset;
    size_t validCount;
} windowSlice1d_t;

windowSlice1d_t windowSlice1dAt(windowGeometry1d_t const *geometry, size_t outputPos);

/*! Output positions [*outLo, *outHi) whose window covers kernel tap k
 *  (contiguous); the tap reads input index outPos*stride + k*dilation - padLeft.
 *  An empty range gives *outLo == *outHi. Lets a per-tap loop skip the
 *  per-outPos windowSlice1dAt membership test. */
void windowTapOutputRange1d(windowGeometry1d_t const *geometry, size_t k, size_t *outLo,
                            size_t *outHi);

/*! ConvT scatter: kernel taps [*kLo, *kHi) of input position inPos land inside
 *  the output; tap k writes output index inPos*stride + k*dilation - padLeft.
 *  The valid taps are contiguous, so a scatter loop can walk exactly this range
 *  instead of bounds-testing every tap in 64-bit signed math (a carry chain and
 *  a two-word compare per tap on Cortex-M). An empty range gives *kLo == *kHi.
 *  dilation 0 is accepted and means every tap hits the same index (all taps or
 *  none). */
void convTValidTapRange(size_t inPos, size_t stride, size_t dilation, size_t padLeft,
                        size_t outputLength, size_t kernelSize, size_t *kLo, size_t *kHi);

/*! Enumerate the contributors of ConvT1d output position
 *  outPos: kernel taps k with (outPos + padLeft - k*dilation) % stride == 0 and
 *  inPos = (outPos + padLeft - k*dilation) / stride in [0, inputLength).
 *  taps must hold kernelSize entries; returns the count (taps are emitted in
 *  ascending kernelIdx order). Used by the gather-formulated ConvT1d forward
 *  and backward. */
typedef struct convTransposeTap {
    size_t inPos;
    size_t kernelIdx;
} convTransposeTap_t;
size_t convTranspose1dTapsAt(size_t outPos, size_t inputLength, size_t kernelSize, size_t stride,
                             size_t dilation, size_t padLeft, convTransposeTap_t *taps);

#endif // ODT_SLIDING_WINDOW_1D_H
