#ifndef ODT_ADAPTIVE_WINDOW_1D_H
#define ODT_ADAPTIVE_WINDOW_1D_H

#include <stdlib.h>

/*! Window index range for one output position of an adaptive 1d pooling layer.
 *  Unlike SlidingWindow1d (uniform, kernel/stride/dilation-driven), adaptive
 *  windows are ratio-driven: each output position averages a possibly-different,
 *  possibly-overlapping slice. See AdaptiveAvgPool1d. */
typedef struct adaptiveWindow1d {
    size_t start; /* floor(outputPos * inputLength / outputLength) */
    size_t count; /* ceil((outputPos+1)*inputLength/outputLength) - start; always >= 1 */
} adaptiveWindow1d_t;

/*! Compute the [start, start+count) input slice averaged into output position
 *  outputPos < outputLength (caller guarantees). An empty input exits. The returned window is
 * always within [0, inputLength). */
adaptiveWindow1d_t adaptiveWindow1dAt(size_t inputLength, size_t outputLength, size_t outputPos);

#endif // ODT_ADAPTIVE_WINDOW_1D_H
