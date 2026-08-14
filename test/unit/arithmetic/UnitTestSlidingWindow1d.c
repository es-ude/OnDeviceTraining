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
    // PyTorch conv trained with padding=3 (see issue #177 ECG enc1).
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

void testTransposeOutputLengthBasic() {
    // (4-1)*1 + 1*(2-1) + 0 + 1 = 5
    kernel_t kernel = {.size = 2, .stride = 1, .dilation = 1, .paddingType = VALID};
    TEST_ASSERT_EQUAL_size_t(5, convTranspose1dOutputLength(4, &kernel, 0));
}

void testTransposeOutputLengthWithStride() {
    // (4-1)*2 + 1*(2-1) + 0 + 1 = 8
    kernel_t kernel = {.size = 2, .stride = 2, .dilation = 1, .paddingType = VALID};
    TEST_ASSERT_EQUAL_size_t(8, convTranspose1dOutputLength(4, &kernel, 0));
}

void testTransposeOutputLengthWithDilation() {
    // (4-1)*1 + 2*(3-1) + 0 + 1 = 8
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 2, .paddingType = VALID};
    TEST_ASSERT_EQUAL_size_t(8, convTranspose1dOutputLength(4, &kernel, 0));
}

void testTransposeOutputLengthWithOutputPadding() {
    // (4-1)*2 + 1*(2-1) + 1 + 1 = 9
    kernel_t kernel = {.size = 2, .stride = 2, .dilation = 1, .paddingType = VALID};
    TEST_ASSERT_EQUAL_size_t(9, convTranspose1dOutputLength(4, &kernel, 1));
}

void testTransposeOutputLengthStrideDilationOutputPadding() {
    // (5-1)*2 + 2*(3-1) + 1 + 1 = 14
    kernel_t kernel = {.size = 3, .stride = 2, .dilation = 2, .paddingType = VALID};
    TEST_ASSERT_EQUAL_size_t(14, convTranspose1dOutputLength(5, &kernel, 1));
}

/* ---- BFP epic PR2 (Task 5): convTranspose1dTapsAt -------------------------
 * Contributor enumeration for the gather-formulated ConvT1d (D9): taps are
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
    // Scatter-equivalence property (D9): for every outPos, the taps must be
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
    RUN_TEST(testTransposeOutputLengthBasic);
    RUN_TEST(testTransposeOutputLengthWithStride);
    RUN_TEST(testTransposeOutputLengthWithDilation);
    RUN_TEST(testTransposeOutputLengthWithOutputPadding);
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
    return UNITY_END();
}
