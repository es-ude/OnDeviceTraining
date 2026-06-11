#ifndef TENSOR_CONVERSION_H
#define TENSOR_CONVERSION_H

#include "Tensor.h"

typedef void (*conversionFunction_t)(tensor_t *inputTensor, tensor_t *outputTensor);

void convertTensor(tensor_t *inputTensor, tensor_t *outputTensor);
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

extern conversionFunction_t conversionMatrix[6][6];

#endif // TENSOR_CONVERSION_H
