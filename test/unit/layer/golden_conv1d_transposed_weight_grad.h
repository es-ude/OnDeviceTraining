/* Goldens for UnitTestConv1dTransposed.c's
 * testConv1dTransposedLayerWeightGradGolden, recorded from the layer's float
 * and SYM weight-grad scatters before the ConvT valid-tap range refactor;
 * that refactor must reproduce them bit for bit. Regenerate only for an
 * intended numeric change. The SYM grad's fitted grid is the canonical
 * 15-bit dyadic CEIL value (DyadicScale.h). Arrays are the
 * [Cin, Cout/groups, K] weight-grad storage, flat. The SYM grad is the
 * funnel-restored wire (mantissas + scale + width). */
#ifndef GOLDEN_CONV1D_TRANSPOSED_WEIGHT_GRAD_H
#define GOLDEN_CONV1D_TRANSPOSED_WEIGHT_GRAD_H

#include <stdint.h>

static const float kWgGoldFloat[60] = {
    -39.0f, -35.0f, -18.0f, -18.0f, 25.0f,  -12.0f, -23.0f, 6.0f,  -18.0f, 43.0f,  3.0f,   27.0f,
    3.0f,   25.0f,  -7.0f,  -28.0f, -66.0f, 12.0f,  54.0f,  18.0f, -41.0f, -12.0f, 35.0f,  6.0f,
    45.0f,  3.0f,   -5.0f,  -31.0f, 44.0f,  3.0f,   37.0f,  17.0f, 29.0f,  28.0f,  -11.0f, 32.0f,
    73.0f,  33.0f,  19.0f,  14.0f,  18.0f,  19.0f,  -56.0f, -3.0f, 26.0f,  7.0f,   13.0f,  -14.0f,
    -23.0f, 6.0f,   34.0f,  5.0f,   -46.0f, -6.0f,  -1.0f,  16.0f, 13.0f,  -43.0f, 4.0f,   51.0f};
static const int32_t kWgGoldSym[60] = {
    -4870, 9954,  -14094, -8061,  -1685, -24346, -7219, 4006,   6676,  1562, -7391,  -14872,
    14261, -8300, 1975,   6851,   539,   -770,   453,   -14663, 8505,  -240, 139,    11839,
    14185, 11564, -3881,  -16166, 23422, 5032,   11858, 20729,  18101, 1833, 23863,  7701,
    -4389, -9185, 22912,  -16156, 3373,  15547,  6482,  22242,  4665,  7560, 337,    1540,
    18130, 231,   -9176,  10590,  10851, -32767, 5059,  9757,   14195, 7412, -14949, 139};
static const float kWgGoldSymScale = 0x1.c73cp-6f;
static const uint32_t kWgGoldSymBits = 16u;

#endif // GOLDEN_CONV1D_TRANSPOSED_WEIGHT_GRAD_H
