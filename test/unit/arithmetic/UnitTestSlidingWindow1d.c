#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "SlidingWindow1d.h"
#include "unity.h"

void testGeometryValidNoPadding() {
    kernel_t kernel = {.size = 2, .stride = 1, .dilation = 1, .paddingType = VALID};
    windowGeometry1d_t g = windowGeometry1dCalc(4, &kernel);

    TEST_ASSERT_EQUAL_size_t(4, g.inputLength);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(2, g.kernelSize);
    TEST_ASSERT_EQUAL_size_t(1, g.stride);
    TEST_ASSERT_EQUAL_size_t(1, g.dilation);
    TEST_ASSERT_EQUAL_size_t(0, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(0, g.padRight);
}

void testGeometryValidWithStride() {
    kernel_t kernel = {.size = 2, .stride = 2, .dilation = 1, .paddingType = VALID};
    windowGeometry1d_t g = windowGeometry1dCalc(6, &kernel);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(0, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(0, g.padRight);
}

void testGeometryValidWithDilation() {
    kernel_t kernel = {.size = 2, .stride = 3, .dilation = 2, .paddingType = VALID};
    windowGeometry1d_t g = windowGeometry1dCalc(9, &kernel);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
}

void testGeometrySameSymmetricPadding() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    TEST_ASSERT_EQUAL_size_t(5, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(1, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(1, g.padRight);
}

void testGeometrySameAsymmetricPadding() {
    kernel_t kernel = {.size = 4, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    TEST_ASSERT_EQUAL_size_t(5, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(1, g.padLeft);  // total pad = 3, left gets floor(3/2)=1
    TEST_ASSERT_EQUAL_size_t(2, g.padRight); // right gets ceil(3/2)=2
}

void testGeometryKernelLargerThanInput() {
    kernel_t kernel = {.size = 5, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(2, &kernel);
    TEST_ASSERT_EQUAL_size_t(2, g.outputLength);
    // Total pad = 5 + (2-1)*1 - 2 = 4; padLeft = 2, padRight = 2
    TEST_ASSERT_EQUAL_size_t(2, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(2, g.padRight);
}

void testGeometryValidKernelLargerThanInput() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = VALID};
    windowGeometry1d_t g = windowGeometry1dCalc(2, &kernel);
    TEST_ASSERT_EQUAL_size_t(0, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(0, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(0, g.padRight);
}

void testSliceCenterFullWindow() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    // outputPos=2 is in the center: window covers inputs [1, 2, 3] — all valid
    windowSlice1d_t s = windowSlice1dAt(&g, 2);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(3, s.validCount);
}

void testSliceLeftEdgeWithPadding() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    // outputPos=0: window starts at inputStart = 0 - 1 = -1; kernel pos 0 -> -1 (OOB)
    //                                                         kernel pos 1 -> 0  (OK)
    //                                                         kernel pos 2 -> 1  (OK)
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(2, s.validCount);
}

void testSliceRightEdgeWithPadding() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    // outputPos=4: window starts at inputStart = 4 - 1 = 3; kernel pos 0 -> 3 (OK)
    //                                                       kernel pos 1 -> 4 (OK)
    //                                                       kernel pos 2 -> 5 (OOB)
    windowSlice1d_t s = windowSlice1dAt(&g, 4);
    TEST_ASSERT_EQUAL_size_t(3, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(2, s.validCount);
}

void testSliceWithDilation() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 2, .paddingType = VALID};
    // effective kernel = 5; inputLen=7 -> outputLen=3
    windowGeometry1d_t g = windowGeometry1dCalc(7, &kernel);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
    // outputPos=0: kernel positions map to input indices [0, 2, 4] — all valid
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(3, s.validCount);
}

void testSlicePartialWindowBothSides() {
    // Both edges OOB but middle is valid (NOT a true empty window — see testSliceTrulyEmpty)
    // pathological: kernel=5, dilation=1, SAME on input of length 2 (padLeft=padRight=2)
    kernel_t kernel = {.size = 5, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(2, &kernel);
    // outputPos=0: inputStart = 0 - 2 = -2;
    //   kernel pos 0 -> -2  (OOB)
    //   kernel pos 1 -> -1  (OOB)
    //   kernel pos 2 -> 0   (OK)
    //   kernel pos 3 -> 1   (OK)
    //   kernel pos 4 -> 2   (OOB, inputLength=2)
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(2, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(2, s.validCount);
}

void testSliceWithDilationAndPadding() {
    // kernel=2, dilation=2, SAME on inputLen=4:
    //   effectiveKernel = 3; outputLength = 4;
    //   totalPad = 3 + 3 - 4 = 2; padLeft=1, padRight=1
    // outputPos=0: inputStart = 0 - 1 = -1
    //   kernel pos 0 -> -1 + 0*2 = -1 (OOB)
    //   kernel pos 1 -> -1 + 1*2 =  1 (OK)
    // ceil((-(-1))/2) = ceil(1/2) = 1; floor would give 0 — this distinguishes ceil from floor.
    kernel_t kernel = {.size = 2, .stride = 1, .dilation = 2, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(4, &kernel);
    TEST_ASSERT_EQUAL_size_t(1, g.padLeft);  // sanity
    TEST_ASSERT_EQUAL_size_t(1, g.padRight); // sanity

    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(1, s.validCount);
}

void testSliceTrulyEmpty() {
    // Pathological geometry constructed by hand — exercise the firstK > lastK sentinel.
    // Window entirely on left padding: padLeft=100, kernel=3, dilation=1 -> all 3 kernel
    // positions land at inputStart..inputStart+2 = -100..-98, all OOB.
    windowGeometry1d_t g = {
        .inputLength = 5,
        .outputLength = 1,
        .kernelSize = 3,
        .stride = 1,
        .dilation = 1,
        .padLeft = 100,
        .padRight = 0,
    };
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(3, s.firstValidKernelOffset); // sentinel = kernelSize
    TEST_ASSERT_EQUAL_size_t(0, s.validCount);
}

void testSliceTrulyEmptyOnRightEdge() {
    // Symmetric counterpart to testSliceTrulyEmpty. Hand-built geometry where
    // outputPos pushes inputStart past the end of input.
    // padRight=10 + outputLength=10 + stride=1 - padLeft=0 = inputStart up to 9
    // for outputPos=2: inputStart = 2*1 - 0 = 2; inputLength = 2 -> inputStart >= inputLength
    windowGeometry1d_t g = {
        .inputLength = 2,
        .outputLength = 10,
        .kernelSize = 1,
        .stride = 1,
        .dilation = 2,
        .padLeft = 0,
        .padRight = 10,
    };
    windowSlice1d_t s = windowSlice1dAt(&g, 2);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidKernelOffset); // sentinel = kernelSize
    TEST_ASSERT_EQUAL_size_t(0, s.validCount);
}

void testGeometryExplicitPaddingStride2() {
    // enc1-shaped geometry: K=7, stride=2, explicit symmetric padding=3 on L=10.
    // padded = 10 + 2*3 = 16; outputLength = (16 - 7)/2 + 1 = 5; padLeft = padRight = 3.
    // This is DISTINCT from SAME, which for the same case yields the minimal
    // totalPad = 5 -> padLeft=2, padRight=3. Explicit padding is how we match a
    // PyTorch conv trained with padding=3.
    kernel_t kernel;
    initKernelExplicit(&kernel, 7, 3, 1, 2); // size, padding, dilation, stride
    windowGeometry1d_t g = windowGeometry1dCalc(10, &kernel);
    TEST_ASSERT_EQUAL_size_t(5, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(3, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(3, g.padRight);
}

void testGeometryExplicitPaddingOddKernelStride1MatchesSame() {
    // For odd kernel + stride 1, explicit (K-1)/2 padding equals SAME exactly.
    kernel_t kernel;
    initKernelExplicit(&kernel, 3, 1, 1, 1); // size, padding, dilation, stride
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    TEST_ASSERT_EQUAL_size_t(5, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(1, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(1, g.padRight);
}

void testGeometryExplicitZeroPaddingEqualsValid() {
    // Explicit padding of 0 must behave like VALID.
    kernel_t kernel;
    initKernelExplicit(&kernel, 2, 0, 1, 2); // size, padding, dilation, stride
    windowGeometry1d_t g = windowGeometry1dCalc(6, &kernel);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(0, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(0, g.padRight);
}

void testTransposeOutputLengthWithDilation() {
    // (4-1)*1 + 2*(3-1) + 0 + 1 = 8
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 2, .paddingType = VALID};
    TEST_ASSERT_EQUAL_size_t(8, convTranspose1dOutputLength(4, &kernel, 0));
}

void testTransposeOutputLengthStrideDilationOutputPadding() {
    // (5-1)*2 + 2*(3-1) + 1 + 1 = 14
    kernel_t kernel = {.size = 3, .stride = 2, .dilation = 2, .paddingType = VALID};
    TEST_ASSERT_EQUAL_size_t(14, convTranspose1dOutputLength(5, &kernel, 1));
}

/* ---- convTranspose1dTapsAt ----
 * Contributor enumeration for the gather-formulated ConvT1d: taps are
 * returned in ASCENDING kernelIdx order (the enumeration loop's k order), so
 * expectations below pin both the SET of (inPos, kernelIdx) pairs and that
 * order. */

static void assertTapsEqual(convTransposeTap_t const *taps, size_t count,
                            size_t const *expectedInPos, size_t const *expectedKernelIdx,
                            size_t expectedCount) {
    TEST_ASSERT_EQUAL_size_t(expectedCount, count);
    for (size_t i = 0; i < count; i++) {
        TEST_ASSERT_EQUAL_size_t(expectedInPos[i], taps[i].inPos);
        TEST_ASSERT_EQUAL_size_t(expectedKernelIdx[i], taps[i].kernelIdx);
    }
}

void testTransposeTapsKernel3Stride2EveryOutPos() {
    // K=3, stride=2, dilation=1, padLeft=0, Lin=4 -> Lout = (4-1)*2 + 2 + 1 = 9.
    // Hand-enumerated contributors of every output position: k with
    // (outPos - k) % 2 == 0 and (outPos - k)/2 in [0, 4).
    convTransposeTap_t taps[3];

    size_t in0[] = {0}, k0[] = {0};
    assertTapsEqual(taps, convTranspose1dTapsAt(0, 4, 3, 2, 1, 0, taps), in0, k0, 1);
    size_t in1[] = {0}, k1[] = {1};
    assertTapsEqual(taps, convTranspose1dTapsAt(1, 4, 3, 2, 1, 0, taps), in1, k1, 1);
    size_t in2[] = {1, 0}, k2[] = {0, 2};
    assertTapsEqual(taps, convTranspose1dTapsAt(2, 4, 3, 2, 1, 0, taps), in2, k2, 2);
    size_t in3[] = {1}, k3[] = {1};
    assertTapsEqual(taps, convTranspose1dTapsAt(3, 4, 3, 2, 1, 0, taps), in3, k3, 1);
    size_t in4[] = {2, 1}, k4[] = {0, 2};
    assertTapsEqual(taps, convTranspose1dTapsAt(4, 4, 3, 2, 1, 0, taps), in4, k4, 2);
    size_t in5[] = {2}, k5[] = {1};
    assertTapsEqual(taps, convTranspose1dTapsAt(5, 4, 3, 2, 1, 0, taps), in5, k5, 1);
    size_t in6[] = {3, 2}, k6[] = {0, 2};
    assertTapsEqual(taps, convTranspose1dTapsAt(6, 4, 3, 2, 1, 0, taps), in6, k6, 2);
    size_t in7[] = {3}, k7[] = {1};
    assertTapsEqual(taps, convTranspose1dTapsAt(7, 4, 3, 2, 1, 0, taps), in7, k7, 1);
    // outPos 8: k=0 would need inPos 4 (OOB) -- only the k=2 tap survives.
    size_t in8[] = {3}, k8[] = {2};
    assertTapsEqual(taps, convTranspose1dTapsAt(8, 4, 3, 2, 1, 0, taps), in8, k8, 1);
}

void testTransposeTapsDilation2() {
    // K=3, stride=2, dilation=2, padLeft=0, Lin=4 -> Lout = 6 + 4 + 1 = 11.
    // stride and dilation both even => odd output positions have NO contributors.
    convTransposeTap_t taps[3];

    size_t in0[] = {0}, k0[] = {0};
    assertTapsEqual(taps, convTranspose1dTapsAt(0, 4, 3, 2, 2, 0, taps), in0, k0, 1);
    size_t in4[] = {2, 1, 0}, k4[] = {0, 1, 2};
    assertTapsEqual(taps, convTranspose1dTapsAt(4, 4, 3, 2, 2, 0, taps), in4, k4, 3);
    TEST_ASSERT_EQUAL_size_t(0, convTranspose1dTapsAt(5, 4, 3, 2, 2, 0, taps));
    size_t in8[] = {3, 2}, k8[] = {1, 2};
    assertTapsEqual(taps, convTranspose1dTapsAt(8, 4, 3, 2, 2, 0, taps), in8, k8, 2);
    size_t in10[] = {3}, k10[] = {2};
    assertTapsEqual(taps, convTranspose1dTapsAt(10, 4, 3, 2, 2, 0, taps), in10, k10, 1);
}

void testTransposeTapsPadLeft1() {
    // K=3, stride=2, dilation=1, padLeft=1, Lin=4 (adjoint-SAME-shaped): the
    // effective position is outPos + padLeft, shifting every enumeration by 1.
    convTransposeTap_t taps[3];

    size_t in0[] = {0}, k0[] = {1};
    assertTapsEqual(taps, convTranspose1dTapsAt(0, 4, 3, 2, 1, 1, taps), in0, k0, 1);
    size_t in1[] = {1, 0}, k1[] = {0, 2};
    assertTapsEqual(taps, convTranspose1dTapsAt(1, 4, 3, 2, 1, 1, taps), in1, k1, 2);
    size_t in6[] = {3}, k6[] = {1};
    assertTapsEqual(taps, convTranspose1dTapsAt(6, 4, 3, 2, 1, 1, taps), in6, k6, 1);
    // outPos 7: p=8, k=0 would need inPos 4 (OOB) -- only the k=2 tap survives.
    size_t in7[] = {3}, k7[] = {2};
    assertTapsEqual(taps, convTranspose1dTapsAt(7, 4, 3, 2, 1, 1, taps), in7, k7, 1);
}

void testTransposeTapsMatchScatterEnumeration() {
    // Scatter-equivalence property: for every outPos, the taps must be
    // EXACTLY the (inPos, k) pairs the SYM scatter loop structure
    // (ConvTranspose1dKernel.c: outBase = inPos*stride - padLeft,
    // outIdx = outBase + k*dilation, bounds-checked) scatters into outPos.
    // Geometry chosen so both clip directions occur (outBase goes negative at
    // inPos=0 and outIdx overshoots outputLength at inPos=4).
    size_t const inputLength = 5;
    size_t const kernelSize = 4;
    size_t const stride = 3;
    size_t const dilation = 2;
    size_t const padLeft = 2;
    size_t const outputLength = 15;
    size_t totalTaps = 0;

    for (size_t outPos = 0; outPos < outputLength; outPos++) {
        convTransposeTap_t taps[4];
        size_t count =
            convTranspose1dTapsAt(outPos, inputLength, kernelSize, stride, dilation, padLeft, taps);
        totalTaps += count;

        size_t scatterCount = 0;
        for (size_t inPos = 0; inPos < inputLength; inPos++) {
            long long outBase = (long long)(inPos * stride) - (long long)padLeft;
            for (size_t k = 0; k < kernelSize; k++) {
                long long outIdx = outBase + (long long)(k * dilation);
                if (outIdx < 0 || outIdx >= (long long)outputLength) {
                    continue;
                }
                if ((size_t)outIdx != outPos) {
                    continue;
                }
                scatterCount++;
                int found = 0;
                for (size_t t = 0; t < count; t++) {
                    if (taps[t].inPos == inPos && taps[t].kernelIdx == k) {
                        found = 1;
                    }
                }
                TEST_ASSERT_TRUE_MESSAGE(found, "scatter-touched (inPos, k) missing from taps");
            }
        }
        // Same count + every scatter pair found + distinct kernelIdx per tap
        // => set equality per outPos.
        TEST_ASSERT_EQUAL_size_t(scatterCount, count);
    }
    // Non-vacuity: this geometry scatters 18 in-bounds products.
    TEST_ASSERT_EQUAL_size_t(18, totalTaps);
}

typedef struct tapRangeSweepStats {
    size_t ranges;
    size_t emptyRanges;
    size_t leftClipped;
    size_t clampEngaged;
} tapRangeSweepStats_t;

static void checkTapRangesAgainstSlices(kernel_t const *kernel, size_t inputLength,
                                        tapRangeSweepStats_t *stats) {
    windowGeometry1d_t g = windowGeometry1dCalc(inputLength, kernel);
    for (size_t k = 0; k < g.kernelSize; k++) {
        size_t lo = SIZE_MAX;
        size_t hi = SIZE_MAX;
        windowTapOutputRange1d(&g, k, &lo, &hi);

        // Explicit: the slice-membership check below cannot see an unclamped
        // hi, because slices past outputLength still report tap k as valid.
        TEST_ASSERT_TRUE(lo <= hi);
        TEST_ASSERT_TRUE(hi <= g.outputLength);

        stats->ranges++;
        stats->emptyRanges += (lo == hi);
        stats->leftClipped += (lo > 0);
        long long unclampedReach =
            (long long)(g.outputLength * g.stride + k * g.dilation) - (long long)g.padLeft;
        stats->clampEngaged +=
            (unclampedReach < (long long)inputLength && hi == g.outputLength && lo < hi);

        for (size_t outPos = 0; outPos < g.outputLength; outPos++) {
            windowSlice1d_t s = windowSlice1dAt(&g, outPos);
            bool coveredBySlice =
                k >= s.firstValidKernelOffset && k < s.firstValidKernelOffset + s.validCount;
            bool inRange = lo <= outPos && outPos < hi;
            TEST_ASSERT_EQUAL(coveredBySlice, inRange);
            if (inRange) {
                size_t inputIdx = outPos * g.stride + k * g.dilation - g.padLeft;
                TEST_ASSERT_EQUAL_size_t(
                    s.firstValidInputIdx + (k - s.firstValidKernelOffset) * g.dilation, inputIdx);
            }
        }
    }
}

void testTapOutputRangeMatchesSliceBruteForce() {
    tapRangeSweepStats_t stats = {0};
    for (size_t inputLength = 1; inputLength <= 12; inputLength++) {
        for (size_t kernelSize = 1; kernelSize <= 5; kernelSize++) {
            for (size_t stride = 1; stride <= 3; stride++) {
                for (size_t dilation = 1; dilation <= 3; dilation++) {
                    kernel_t kernel;
                    initKernel(&kernel, kernelSize, VALID, dilation, stride);
                    checkTapRangesAgainstSlices(&kernel, inputLength, &stats);
                    initKernel(&kernel, kernelSize, SAME, dilation, stride);
                    checkTapRangesAgainstSlices(&kernel, inputLength, &stats);
                    for (size_t padding = 0; padding <= 4; padding++) {
                        initKernelExplicit(&kernel, kernelSize, padding, dilation, stride);
                        checkTapRangesAgainstSlices(&kernel, inputLength, &stats);
                    }
                }
            }
        }
    }
    TEST_ASSERT_TRUE(stats.emptyRanges > 0);
    TEST_ASSERT_TRUE(stats.leftClipped > 0);
    TEST_ASSERT_TRUE(stats.clampEngaged > 0);
    TEST_ASSERT_TRUE(stats.ranges > stats.emptyRanges);
}

void testTapOutputRangeEmptyWhenStrideSkipsTheInput() {
    // L=1, K=2, stride 2, padding 1: outputLength 1, tap 0's only window
    // reads index -1, tap 1 reads index 0.
    kernel_t kernel;
    initKernelExplicit(&kernel, 2, 1, 1, 2); // size, padding, dilation, stride
    windowGeometry1d_t g = windowGeometry1dCalc(1, &kernel);
    TEST_ASSERT_EQUAL_size_t(1, g.outputLength);

    size_t lo = SIZE_MAX;
    size_t hi = SIZE_MAX;
    windowTapOutputRange1d(&g, 0, &lo, &hi);
    TEST_ASSERT_EQUAL_size_t(lo, hi);
    TEST_ASSERT_TRUE(hi <= g.outputLength);

    windowTapOutputRange1d(&g, 1, &lo, &hi);
    TEST_ASSERT_EQUAL_size_t(0, lo);
    TEST_ASSERT_EQUAL_size_t(1, hi);
}

/* ---- B6: differential tests against the pre-B6 (int64-divide) geometry ----
 * refWindowSlice1dAt / refWindowTapOutputRange1d are verbatim copies of the
 * long-long implementations that the divide-free rewrite replaced; the
 * exhaustive sweeps below pin the rewrite as bit-identical on every field. */

static windowSlice1d_t refWindowSlice1dAt(windowGeometry1d_t const *geometry, size_t outputPos) {
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

static void refWindowTapOutputRange1d(windowGeometry1d_t const *geometry, size_t k, size_t *outLo,
                                      size_t *outHi) {
    // signed: the tap offset relative to the window start is negative when
    // padLeft exceeds k*dilation
    long long tapOffset = (long long)(k * geometry->dilation) - (long long)geometry->padLeft;
    long long stride = (long long)geometry->stride;
    long long inputLen = (long long)geometry->inputLength;

    // first outPos with outPos*stride + tapOffset >= 0
    long long lo = 0;
    if (tapOffset < 0) {
        lo = (-tapOffset + stride - 1) / stride;
    }

    // first outPos with outPos*stride + tapOffset >= inputLen
    long long hi = 0;
    if (inputLen - tapOffset > 0) {
        hi = (inputLen - tapOffset + stride - 1) / stride;
    }
    if (hi > (long long)geometry->outputLength) {
        hi = (long long)geometry->outputLength;
    }
    if (lo > hi) {
        lo = hi;
    }

    *outLo = (size_t)lo;
    *outHi = (size_t)hi;
}

typedef void (*geometryCheck_t)(windowGeometry1d_t const *g, char const *label, size_t *checks);

/* inputLength 0..40 x kernelSize 1..9 x stride 1..4 x dilation 1..4 x
 * {VALID, SAME, EXPLICIT padding 0..5}. */
static size_t sweepGeometries(geometryCheck_t check, size_t *skipped) {
    size_t checks = 0;
    for (size_t inputLength = 0; inputLength <= 40; inputLength++) {
        for (size_t kernelSize = 1; kernelSize <= 9; kernelSize++) {
            for (size_t stride = 1; stride <= 4; stride++) {
                for (size_t dilation = 1; dilation <= 4; dilation++) {
                    for (int mode = -2; mode <= 5; mode++) {
                        kernel_t kernel;
                        char label[160];
                        if (mode == -2) {
                            initKernel(&kernel, kernelSize, VALID, dilation, stride);
                        } else if (mode == -1) {
                            initKernel(&kernel, kernelSize, SAME, dilation, stride);
                        } else {
                            initKernelExplicit(&kernel, kernelSize, (size_t)mode, dilation, stride);
                        }
                        snprintf(label, sizeof(label),
                                 "L=%zu K=%zu stride=%zu dil=%zu mode=%s pad=%d", inputLength,
                                 kernelSize, stride, dilation,
                                 mode == -2 ? "VALID" : (mode == -1 ? "SAME" : "EXPLICIT"),
                                 mode < 0 ? 0 : mode);
                        windowGeometry1d_t g = windowGeometry1dCalc(inputLength, &kernel);
                        if (g.padLeft > (size_t)INT32_MAX) {
                            // windowGeometry1dCalc(0, SAME) with stride > effective kernel
                            // wraps (outputLength - 1) * stride, giving padLeft ~ SIZE_MAX/2
                            // and outputLength 0; the reference then overflows long long
                            // (UB), so there is no defined result to compare against.
                            TEST_ASSERT_TRUE_MESSAGE(inputLength == 0 && mode == -1, label);
                            (*skipped)++;
                            continue;
                        }
                        check(&g, label, &checks);
                    }
                }
            }
        }
    }
    return checks;
}

static void checkSliceAgainstReference(windowGeometry1d_t const *g, char const *label,
                                       size_t *checks) {
    for (size_t outPos = 0; outPos <= g->outputLength + 2; outPos++) {
        windowSlice1d_t expected = refWindowSlice1dAt(g, outPos);
        windowSlice1d_t actual = windowSlice1dAt(g, outPos);
        (*checks)++;
        if (expected.firstValidInputIdx != actual.firstValidInputIdx ||
            expected.firstValidKernelOffset != actual.firstValidKernelOffset ||
            expected.validCount != actual.validCount) {
            char msg[320];
            snprintf(msg, sizeof(msg),
                     "%s outPos=%zu: ref {idx=%zu, k=%zu, n=%zu} != new {idx=%zu, k=%zu, n=%zu}",
                     label, outPos, expected.firstValidInputIdx, expected.firstValidKernelOffset,
                     expected.validCount, actual.firstValidInputIdx, actual.firstValidKernelOffset,
                     actual.validCount);
            TEST_FAIL_MESSAGE(msg);
        }
    }
}

static void checkTapRangeAgainstReference(windowGeometry1d_t const *g, char const *label,
                                          size_t *checks) {
    for (size_t k = 0; k < g->kernelSize; k++) {
        size_t expectedLo = SIZE_MAX, expectedHi = SIZE_MAX;
        size_t actualLo = SIZE_MAX - 1, actualHi = SIZE_MAX - 1;
        refWindowTapOutputRange1d(g, k, &expectedLo, &expectedHi);
        windowTapOutputRange1d(g, k, &actualLo, &actualHi);
        (*checks)++;
        if (expectedLo != actualLo || expectedHi != actualHi) {
            char msg[320];
            snprintf(msg, sizeof(msg), "%s k=%zu: ref [%zu, %zu) != new [%zu, %zu)", label, k,
                     expectedLo, expectedHi, actualLo, actualHi);
            TEST_FAIL_MESSAGE(msg);
        }
    }
}

void testWindowSlice1dAtMatchesReferenceExhaustive() {
    size_t skipped = 0;
    size_t checks = sweepGeometries(checkSliceAgainstReference, &skipped);
    printf("windowSlice1dAt: %zu (geometry, outPos) cases matched the reference; %zu wrapped L=0 "
           "SAME geometries skipped\n",
           checks, skipped);
    TEST_ASSERT_TRUE(checks > 0);
}

void testWindowTapOutputRange1dMatchesReferenceExhaustive() {
    size_t skipped = 0;
    size_t checks = sweepGeometries(checkTapRangeAgainstReference, &skipped);
    printf("windowTapOutputRange1d: %zu (geometry, k) cases matched the reference; %zu wrapped L=0 "
           "SAME geometries skipped\n",
           checks, skipped);
    TEST_ASSERT_TRUE(checks > 0);
}

/* The per-tap bounds test the ConvT scatters used before they switched to
 * convTValidTapRange: output index inPos*stride - padLeft + k*dilation, signed. */
static bool refConvTTapValid(size_t inPos, size_t stride, size_t dilation, size_t padLeft,
                             size_t outputLength, size_t k) {
    long long outIdx =
        (long long)(inPos * stride) - (long long)padLeft + (long long)k * (long long)dilation;
    return outIdx >= 0 && outIdx < (long long)outputLength;
}

static void checkConvTTapRange(size_t inPos, size_t stride, size_t dilation, size_t padLeft,
                               size_t outputLength, size_t kernelSize, size_t *checks) {
    size_t kLo = SIZE_MAX, kHi = SIZE_MAX;
    convTValidTapRange(inPos, stride, dilation, padLeft, outputLength, kernelSize, &kLo, &kHi);
    (*checks)++;
    char msg[200];
    snprintf(msg, sizeof(msg),
             "inPos=%zu stride=%zu dil=%zu padLeft=%zu Lout=%zu K=%zu -> [%zu,%zu)", inPos, stride,
             dilation, padLeft, outputLength, kernelSize, kLo, kHi);
    TEST_ASSERT_TRUE_MESSAGE(kLo <= kHi && kHi <= kernelSize, msg);
    for (size_t k = 0; k < kernelSize; k++) {
        bool expected = refConvTTapValid(inPos, stride, dilation, padLeft, outputLength, k);
        bool actual = (k >= kLo && k < kHi);
        TEST_ASSERT_EQUAL_MESSAGE(expected, actual, msg);
    }
}

/* inputLength 1..12 x K 1..7 x stride 1..3 x dilation 0..3 x padLeft 0..6 x
 * outputLength 0 .. VALID ConvT length + 2 (covers the +-2 band around the
 * geometry and every shorter adjoint length). dilation 0 is not a real
 * geometry, but the scatters tolerated it (every tap hits outBase), so the
 * helper must reproduce that instead of dividing by zero. */
void testConvTValidTapRangeMatchesBoundsTest() {
    size_t checks = 0;
    for (size_t inputLength = 1; inputLength <= 12; inputLength++) {
        for (size_t kernelSize = 1; kernelSize <= 7; kernelSize++) {
            for (size_t stride = 1; stride <= 3; stride++) {
                for (size_t dilation = 0; dilation <= 3; dilation++) {
                    size_t convTLength =
                        (inputLength - 1) * stride + dilation * (kernelSize - 1) + 1;
                    for (size_t padLeft = 0; padLeft <= 6; padLeft++) {
                        for (size_t outputLength = 0; outputLength <= convTLength + 2;
                             outputLength++) {
                            for (size_t inPos = 0; inPos < inputLength; inPos++) {
                                checkConvTTapRange(inPos, stride, dilation, padLeft, outputLength,
                                                   kernelSize, &checks);
                            }
                        }
                    }
                }
            }
        }
    }
    printf("convTValidTapRange: %zu (geometry, inPos) cases matched the per-tap bounds test\n",
           checks);
    TEST_ASSERT_TRUE(checks > 0);
}

/* Ceil-division wrap (advrev A2): with n + d - 1 > SIZE_MAX, (n + d - 1) / d wraps
 * on any size_t width. On the 32-bit MCU size_t that needs n + d > 2^32; on the
 * 64-bit host it needs a stride or dilation near SIZE_MAX, which these cases use.
 * The reference is a brute-force per-tap membership test in 128-bit math, which
 * does not overflow for the operands below (every product stays under 2^127). */
__extension__ typedef __int128 wide_t;

static bool tapIndexInRange(size_t pos, size_t step, size_t k, size_t dilation, size_t padLeft,
                            size_t length) {
    wide_t const idx = (wide_t)pos * step + (wide_t)k * dilation - (wide_t)padLeft;
    return idx >= 0 && idx < (wide_t)length;
}

static void assertSliceMatchesWideReference(windowGeometry1d_t const *g, size_t outPos) {
    size_t firstK = g->kernelSize;
    size_t count = 0;
    for (size_t k = 0; k < g->kernelSize; k++) {
        if (tapIndexInRange(outPos, g->stride, k, g->dilation, g->padLeft, g->inputLength)) {
            if (count == 0) {
                firstK = k;
            }
            count++;
        }
    }
    windowSlice1d_t s = windowSlice1dAt(g, outPos);
    TEST_ASSERT_EQUAL_size_t(count, s.validCount);
    TEST_ASSERT_EQUAL_size_t(firstK, s.firstValidKernelOffset);
    if (count > 0) {
        size_t const firstIdx =
            (size_t)((wide_t)outPos * g->stride + (wide_t)firstK * g->dilation - g->padLeft);
        TEST_ASSERT_EQUAL_size_t(firstIdx, s.firstValidInputIdx);
    }
}

void testWindowSlice1dAtCeilDivDoesNotWrapNearSizeMax() {
    size_t const half = SIZE_MAX / 2 + 1; /* 2^63 on the host */
    /* neg = padLeft = half + 5, dilation = half + 10: neg + dilation - 1 wraps to 14,
     * so the old ceil gave firstK = 0 and an empty slice; tap 1 reads input 5. */
    windowGeometry1d_t g = {.inputLength = 10,
                            .kernelSize = 3,
                            .stride = 1,
                            .dilation = half + 10,
                            .padLeft = half + 5,
                            .padRight = 0,
                            .outputLength = 1};
    assertSliceMatchesWideReference(&g, 0);
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(1, s.validCount);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(5, s.firstValidInputIdx);
}

void testWindowTapOutputRange1dCeilDivDoesNotWrapNearSizeMax() {
    size_t const half = SIZE_MAX / 2 + 1;
    /* tap 0, neg = half + 5, stride = half + 10: output 1 reads input 5. */
    windowGeometry1d_t g = {.inputLength = 10,
                            .kernelSize = 1,
                            .stride = half + 10,
                            .dilation = 1,
                            .padLeft = half + 5,
                            .padRight = 0,
                            .outputLength = 2};
    size_t lo = 0;
    size_t count = 0;
    for (size_t o = 0; o < g.outputLength; o++) {
        if (tapIndexInRange(o, g.stride, 0, g.dilation, g.padLeft, g.inputLength)) {
            if (count == 0) {
                lo = o;
            }
            count++;
        }
    }
    TEST_ASSERT_EQUAL_size_t(1, count);
    size_t outLo = 0;
    size_t outHi = 0;
    windowTapOutputRange1d(&g, 0, &outLo, &outHi);
    TEST_ASSERT_EQUAL_size_t(lo, outLo);
    TEST_ASSERT_EQUAL_size_t(lo + count, outHi);
}

void testConvTValidTapRangeCeilDivDoesNotWrapNearSizeMax() {
    struct {
        size_t inPos, stride, dilation, padLeft, outputLength, kernelSize;
    } const cases[] = {
        /* hi = ceil(2 / SIZE_MAX): 2 + SIZE_MAX - 1 wraps to 0, dropping tap 0 */
        {0, 1, SIZE_MAX, 0, 2, 4},
        /* lo = ceil(5 / dilation) with dilation > 2^63: tap 1 writes output 5 */
        {0, 1, SIZE_MAX / 2 + 11, SIZE_MAX / 2 + 6, 10, 3},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        size_t lo = 0;
        size_t count = 0;
        for (size_t k = 0; k < cases[c].kernelSize; k++) {
            if (tapIndexInRange(cases[c].inPos, cases[c].stride, k, cases[c].dilation,
                                cases[c].padLeft, cases[c].outputLength)) {
                if (count == 0) {
                    lo = k;
                }
                count++;
            }
        }
        TEST_ASSERT_EQUAL_size_t(1, count);
        size_t kLo = 0;
        size_t kHi = 0;
        convTValidTapRange(cases[c].inPos, cases[c].stride, cases[c].dilation, cases[c].padLeft,
                           cases[c].outputLength, cases[c].kernelSize, &kLo, &kHi);
        TEST_ASSERT_EQUAL_size_t(lo, kLo);
        TEST_ASSERT_EQUAL_size_t(lo + count, kHi);
    }
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testGeometryValidNoPadding);
    RUN_TEST(testGeometryValidWithStride);
    RUN_TEST(testGeometryValidWithDilation);
    RUN_TEST(testGeometrySameSymmetricPadding);
    RUN_TEST(testGeometrySameAsymmetricPadding);
    RUN_TEST(testGeometryKernelLargerThanInput);
    RUN_TEST(testGeometryValidKernelLargerThanInput);
    RUN_TEST(testGeometryExplicitPaddingStride2);
    RUN_TEST(testGeometryExplicitPaddingOddKernelStride1MatchesSame);
    RUN_TEST(testGeometryExplicitZeroPaddingEqualsValid);
    RUN_TEST(testTransposeOutputLengthWithDilation);
    RUN_TEST(testTransposeOutputLengthStrideDilationOutputPadding);
    RUN_TEST(testSliceCenterFullWindow);
    RUN_TEST(testSliceLeftEdgeWithPadding);
    RUN_TEST(testSliceRightEdgeWithPadding);
    RUN_TEST(testSliceWithDilation);
    RUN_TEST(testSlicePartialWindowBothSides);
    RUN_TEST(testSliceWithDilationAndPadding);
    RUN_TEST(testSliceTrulyEmpty);
    RUN_TEST(testSliceTrulyEmptyOnRightEdge);
    RUN_TEST(testTransposeTapsKernel3Stride2EveryOutPos);
    RUN_TEST(testTransposeTapsDilation2);
    RUN_TEST(testTransposeTapsPadLeft1);
    RUN_TEST(testTransposeTapsMatchScatterEnumeration);
    RUN_TEST(testTapOutputRangeMatchesSliceBruteForce);
    RUN_TEST(testTapOutputRangeEmptyWhenStrideSkipsTheInput);
    RUN_TEST(testWindowSlice1dAtMatchesReferenceExhaustive);
    RUN_TEST(testWindowTapOutputRange1dMatchesReferenceExhaustive);
    RUN_TEST(testConvTValidTapRangeMatchesBoundsTest);
    RUN_TEST(testWindowSlice1dAtCeilDivDoesNotWrapNearSizeMax);
    RUN_TEST(testWindowTapOutputRange1dCeilDivDoesNotWrapNearSizeMax);
    RUN_TEST(testConvTValidTapRangeCeilDivDoesNotWrapNearSizeMax);
    return UNITY_END();
}
