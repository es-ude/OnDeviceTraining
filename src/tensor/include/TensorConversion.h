#ifndef TENSOR_CONVERSION_H
#define TENSOR_CONVERSION_H

#include "Tensor.h"

typedef void (*conversionFunction_t)(tensor_t *inputTensor, tensor_t *outputTensor);

void convertTensor(tensor_t *inputTensor, tensor_t *outputTensor);

#define ODT_CONVERSION_CHUNK_ELEMS 256
/* Streams `count` elements of src (FLOAT32/SYM_INT32/SYM/ASYM) starting at
 * element `elemOffset` into out[] as dequantized floats. Contract:
 * count <= ODT_CONVERSION_CHUNK_ELEMS, elemOffset % 8 == 0 (packed-width
 * byte alignment); violations fail fast. */
void dequantChunkToFloat(const tensor_t *src, size_t elemOffset, size_t count, float *out);

/*! @brief SYM_INT32 -> SYM_INT32 requantization with a FRESH dynamic scale.
 *
 * Pass A (reads only): absMax = max_i |mantissa_i * inScale|.
 * scale = (absMax == 0) ? 1.0f : absMax / qMax, with qMax = 2^(qMaxBits-1)-1
 * and qMin = -2^(qMaxBits-1) taken from the OUTPUT tensor's
 * symInt32QConfig_t. Pass B: out_i = roundByMode(clamp((mantissa_i *
 * inScale) / scale, qMin, qMax), outQConfig->roundingMode). Writes the fresh
 * scale to the output qConfig. Never saturates by construction: absMax maps
 * exactly to +-qMax. (Deutel, IEEE TCAD 44(4) 2025, Eqs. 5-7 idiom: observe
 * range -> fresh scale -> requantize.)
 *
 * IN-PLACE CAPABLE: inputTensor == outputTensor is allowed. Pass A only
 * reads; pass B is a same-index read-then-write over int32 storage on both
 * sides. When aliased, the single qConfig carries the input scale on entry
 * and the fresh scale on exit.
 *
 * Elements are processed in flat storage order; orderOfDimensions is ignored
 * (matches every converter in this file). Shape/sparsity are not touched.
 * Wired into conversionMatrix[SYM_INT32][SYM_INT32]; convertTensor's
 * same-type branch short-circuits BEFORE the matrix, so this is reachable
 * only via direct matrix dispatch. */
void requantSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor);
/*! @brief SYM_INT32 -> SYM_INT32 requantization into a PRE-SET target scale.
 *
 * The target scale is the OUTPUT tensor's symInt32QConfig_t->scale and must
 * be set by the caller BEFORE the call; it is never modified. Guard:
 * !(scale > 0.0f) (NaN-robust) -> PRINT_ERROR + exit(1).
 * out_i = roundByMode(clamp((mantissa_i * inScale) / targetScale, qMin,
 * qMax), outQConfig->roundingMode) with qMin/qMax from the output qConfig.
 *
 * SATURATES BY DESIGN: values outside the representable range clamp to
 * qMin/qMax — this is the Deutel (IEEE TCAD 44(4) 2025) Eq. 4 analog
 * (layer-epilogue requant of errors/activations into a known target scale);
 * clamping IS the intended semantics, not an error. Covers the #187-deferred
 * symmetric scratch+convert propLoss case.
 *
 * IN-PLACE CAPABLE via a shared data buffer (two tensor_t views with their
 * own qConfigs): single same-index read-then-write pass over int32 storage.
 * Flat storage order; orderOfDimensions is ignored; shape/sparsity are not
 * touched. NOT wired into conversionMatrix (the dynamic variant owns the
 * diagonal). */
void requantSymInt32TensorToScale(tensor_t *inputTensor, tensor_t *outputTensor);
char *quantTypeToString(qtype_t t);
/*! SYM_INT32 -> SYM with NO rescale: carry the input scale, pack mantissas
 *  verbatim. Exits if any mantissa exceeds the target qBits. The no-rescale
 *  partner of convertSymTensorToSymInt32Tensor (#227). Not a conversionMatrix
 *  cell (the rescale variant owns [SYM_INT32][SYM]); call directly. */
void repackSymInt32ToSymNoRescale(tensor_t *inputTensor, tensor_t *outputTensor);

/* Widens n packed srcBits-wide codes to int32 and sign-extends the
 * two's-complement payload ((v ^ signBit) - signBit). srcStartBit is the BIT
 * position of the first code within src, so DeltaSym-style decoders can
 * sign-extend a segment that starts mid-byte; byte-aligned callers pass 0.
 * Direct-call helper behind every SYM -> * conversionMatrix cell (see
 * docs/conventions/tensor.md, "Sign-extend on unpack"). srcBits must be > 0;
 * srcBits >= 32 emits the low 32 bits unextended (full-width codes). */
void unpackSignExtend(const uint8_t *src, size_t srcBits, size_t srcStartBit, int32_t *dst,
                      size_t n);

/* Grad-accumulate primitives (PR3, #261). Direct-call only — not conversionMatrix
 * cells. FixedGrid = fit-preserving: carries the target's scale (first store after
 * a zero-fill derives it from the increment) and ABORTS on grid overflow (#227
 * discipline, no clamp). Rescale = requant: fresh absmax (SYM) / fresh affine grid
 * (ASYM) on every store. n must equal the target's element count. */
void accumulateFloatIntoSymTensorFixedGrid(tensor_t *target, const float *inc, size_t n);
void accumulateFloatIntoSymTensorRescale(tensor_t *target, const float *inc, size_t n);
void accumulateFloatIntoAsymTensorRescale(tensor_t *target, const float *inc, size_t n);

/* Tensor-typed accumulate entry points (#296 Stage 2) — stream the increment
 * chunk-wise via dequantChunkToFloat; float* variants keep their signatures.
 * accumulateSymInt32IntoSymInt32Rescale reproduces addSymInt32TensorsInplace's
 * Strategy-A semantics (dequant both -> float add -> fresh-absmax requant with
 * the TARGET's roundingMode) in O(chunk); Add.c stays untouched.
 * accumulateTensorIntoSymFixedGrid/accumulateTensorIntoSymRescale/
 * accumulateTensorIntoAsymRescale reject a self-aliased increment (increment
 * and target sharing the same data pointer) with exit(1) — the funnel
 * epilogue always passes a distinct intermediate (release-review, PR #324). */
void accumulateTensorIntoSymFixedGrid(tensor_t *target, const tensor_t *increment);
void accumulateTensorIntoSymRescale(tensor_t *target, const tensor_t *increment);
void accumulateTensorIntoAsymRescale(tensor_t *target, const tensor_t *increment);
void accumulateTensorIntoFloat32Inplace(tensor_t *target, const tensor_t *increment);
void accumulateSymInt32IntoSymInt32Rescale(tensor_t *target, const tensor_t *increment);

extern conversionFunction_t conversionMatrix[7][7];

#endif // TENSOR_CONVERSION_H
