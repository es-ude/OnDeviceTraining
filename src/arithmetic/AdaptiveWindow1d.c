#define SOURCE_FILE "ODT_ADAPTIVE_WINDOW_1D"

#include "AdaptiveWindow1d.h"

#include "Common.h"

adaptiveWindow1d_t adaptiveWindow1dAt(size_t inputLength, size_t outputLength, size_t outputPos) {
    if (inputLength == 0) {
        PRINT_ERROR("AdaptiveWindow1d: input length must be >= 1");
        exit(1);
    }
    size_t start = (outputPos * inputLength) / outputLength;
    // Integer ceil. Lengths stay far below 2^16, so no overflow even on 32-bit size_t.
    size_t end = ((outputPos + 1) * inputLength + outputLength - 1) / outputLength;
    return (adaptiveWindow1d_t){.start = start, .count = end - start};
}
