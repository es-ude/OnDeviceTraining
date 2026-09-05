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
/*! @brief BFP -> BFP requantization onto the TARGET's geometry/widths.
 *
 * Two-pass value-domain repack: pass 1 streams the source through
 * dequantChunkToFloat (the source's OWN group scales apply per element) and
 * derives one FRESH stored exponent per TARGET group (absmax -> smallest E
 * with absmax/2^E <= qMax, D6 clamps at the stored-range ends); pass 2
 * re-streams and packs mantissas at the target's mantissaBits with the
 * TARGET config's roundingMode. Source exponents are never copied. This one
 * cell covers the OUT_WRITE width-restore, mantissa/exponent-width changes,
 * and re-blocking (any source geometry -> any target geometry). n == 0
 * writes zero-state exponents (= bias) and no payload.
 *
 * NOT in-place capable (unlike requantSymInt32Tensor): pass 2 re-reads the
 * source under its original exponents. Flat storage order; orderOfDimensions
 * is ignored; shape/sparsity are not touched. Wired into
 * conversionMatrix[BFP][BFP]; convertTensor's same-type branch (verbatim
 * copy between IDENTICAL configs) short-circuits BEFORE the matrix, so this
 * is reachable only via direct matrix dispatch (executeConvert / the
 * OUT_WRITE epilogue). */
void requantBfpTensor(tensor_t *inputTensor, tensor_t *outputTensor);
/* The single BFP exponent authority (frexpf snap-up, D6 clamp both ends).
   A NON-FINITE absMax (inf or NaN -- an overflowed product in a caller's
   pass 1) has no derivable exponent and saturates at the cap, D6's high
   regime taken to its limit: the block's mantissas then clamp to the code
   range under the largest FINITE scale, instead of frexpf's unspecified
   result leaking an arbitrary exponent into the emit pass.
   The funnel's staging quantizer and (since PR2) wire OUT_WRITE epilogues
   derive exponents through this authority; epic PR3 added the grad-accumulate
   engines and the scale arm, and extended OUT_WRITE's reach to the backward's
   dx wire (op-local re-blocking never happens -- the D8 amendment,
   docs/conventions/arithmetic-bfp.md §9). */
void deriveBfpStoredExponent(float absMax, float qMax, int32_t bias, uint8_t maxStored,
                             uint8_t *storedOut);
/* Quantize a float buffer into UNPACKED int32 BFP mantissa codes (no payload
   packing). Two passes per group: absmax -> deriveBfpStoredExponent into
   outQC->exponents[g], then round/clamp codes. outQC supplies
   geometry/widths/roundingMode; its exponents array must have numGroups
   entries and is overwritten. codesOut is caller-owned, n entries. Writing
   into the CALLER's codesOut is exempt from the no-O(n)-internal-scratch
   converter contract by design -- this function allocates nothing.
   Value-domain: saturates (D6), never aborts. */
void quantizeFloatBufferToBfpCodes(const float *values, size_t n, bfpQConfig_t *outQC,
                                   int32_t *codesOut);
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

/* BFP grad-accumulate twins (BFP epic PR3). FixedGrid = fit-preserving:
 * carries the target's per-group exponents (a fresh ALL-ZERO-code accumulator
 * first derives them from the increment, per group) and ABORTS on mantissa
 * overflow (#227 code-domain discipline, no clamp). Rescale = requant:
 * re-derives every group's exponent from the decoded-plus-increment absmax
 * (value-domain, saturates — D6). n must equal the target's element count;
 * the tensor-typed twins stream any dequantChunkToFloat-supported increment
 * and reject a self-aliased one (shared data pointer) with exit(1), like
 * their SYM/ASYM siblings. */
void accumulateFloatIntoBfpTensorFixedGrid(tensor_t *target, const float *inc, size_t n);
void accumulateFloatIntoBfpTensorRescale(tensor_t *target, const float *inc, size_t n);
void accumulateTensorIntoBfpFixedGrid(tensor_t *target, const tensor_t *increment);
void accumulateTensorIntoBfpRescale(tensor_t *target, const tensor_t *increment);

/* In-place value-domain scale of a BFP tensor by an arbitrary float factor
 * (scaleOptimizerGradients's BFP arm: REDUCTION_MEAN mean-scale, clip
 * coefficients). BFP has no O(1) scale fold -- a general factor moves every
 * group's absmax off its 2^E grid -- so this is an honest O(n) two-pass
 * repack: pass 1 re-derives every group's exponent from the scaled absmax
 * (old exponents latched first); pass 2 requantizes with the config's OWN
 * storage roundingMode (storage requantization, not an op -- #282
 * target-owned convention), one roundByMode per element in element order,
 * clamped (value-domain saturation, D6). A power-of-two factor is exact:
 * exponents shift, codes bit-unchanged -- except where the derived exponent
 * saturates at 0 or the cap (D6), where codes shift instead. An all-zero
 * group re-derives the zero state (stored = bias). A group already sitting
 * at the exponent cap has no headroom left, so even a finite factor can
 * overflow its scaled values to +-inf: those saturate to the code range
 * (the clamp runs in the float domain, before the round). An EMPTY tensor
 * (n == 0) is left in the canonical zero state (every group's stored
 * exponent = bias), never with its previous grid. Grouped-capable;
 * direct-call only, not a conversionMatrix cell.
 * `factor` MUST be finite: a non-finite factor fail-fasts (PRINT_ERROR +
 * exit(1)) because a BFP grid has no NaN/inf code -- unlike the FLOAT32/
 * SYM/ASYM scale arms, there is nothing here to propagate it into. */
void scaleBfpTensorInPlace(tensor_t *t, float factor);

extern conversionFunction_t conversionMatrix[7][7];

#endif // TENSOR_CONVERSION_H
