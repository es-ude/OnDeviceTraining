#define SOURCE_FILE "EXECUTE-OP-UTEST"
#include <stdint.h>
#include <string.h>

#include "Add.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "DeathTest.h"
#include "ExecuteOp.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* ---- helpers -------------------------------------------------------- */

static tensor_t *buildSym(size_t n, const int32_t *mantissas, float scale) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < n; i++) {
        ((int32_t *)t->data)[i] = mantissas[i];
    }
    ((symInt32QConfig_t *)t->quantization->qConfig)->scale = scale;
    return t;
}

static tensor_t *buildFloat(size_t n, const float *values) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)values, n);
    return t;
}

/* SYM_INT32 (HALF_AWAY, default qMaxBits=ODT_SYM_OPERAND_QMAXBITS) tensor
 * seeded from float values via tensorFillFromFloatBuffer ->
 * convertFloatTensorToSymInt32Tensor (absmax -> scale, round-clamp) — the
 * same idiom buildSymInt32TensorND uses in the other funnel-consumer test
 * files (UnitTestLayerNorm.c, UnitTestGroupNorm.c, UnitTestReduce.c). */
static tensor_t *buildSymInt32Tensor1dFromFloats(size_t n, const float *values) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(t, (float *)values, n);
    return t;
}

/* Packed sub-byte SYM tensor (PR3 targets), distinct from buildSym's
 * SYM_INT32 compute format above. Mantissas are packed verbatim via
 * byteConversion (the same seeding idiom UnitTestTensorConversion.c uses for
 * the primitives this task wires up); all-zero mantissas reproduce the
 * post-initTensor / post-optimizerZeroGrad fresh-accumulator state. */
static tensor_t *buildPackedSym(size_t n, const int32_t *mantissas, uint8_t qBits, float scale) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitSym(qBits, HALF_AWAY), NULL);
    byteConversion((uint8_t *)mantissas, 32, t->data, qBits, n);
    ((symQConfig_t *)t->quantization->qConfig)->scales[0] = scale;
    return t;
}

/* Packed ASYM tensor (PR3 targets); codes are non-negative, no sign-extend
 * needed on the seeding side (byteConversion narrows verbatim). zeroPoint is
 * the CODE-domain uint16 (group-quant PR4, D6): dequant = (code - zeroPoint)*scale. */
static tensor_t *buildAsymPacked(size_t n, const int32_t *codes, uint8_t qBits, float scale,
                                 uint16_t zeroPoint) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitAsym(qBits, HALF_AWAY), NULL);
    byteConversion((uint8_t *)codes, 32, t->data, qBits, n);
    asymQConfig_t *qc = t->quantization->qConfig;
    qc->scales[0] = scale;
    qc->zeroPoints[0] = zeroPoint;
    return t;
}

/* Sign-extends packed SYM mantissas for in-test readback (byteConversion
 * zero-fills on widen); mirrors UnitTestTensorConversion.c's helper of the
 * same name. */
static void symTestUnpackSignExtend(const uint8_t *packed, size_t qBits, int32_t *out, size_t n) {
    byteConversion((uint8_t *)packed, qBits, (uint8_t *)out, 32, n);
    const int32_t signBit = (int32_t)1 << (qBits - 1);
    const int32_t mask = (int32_t)(((uint32_t)1 << qBits) - 1u);
    for (size_t i = 0; i < n; i++) {
        int32_t v = out[i] & mask;
        out[i] = (v ^ signBit) - signBit;
    }
}

/* ---- tests ---------------------------------------------------------- */

/* Prologue no-op: dtype matches the arithmetic -> the source is passed
 * through untouched (data AND scale unchanged after the call). */
void testProloguePassesMatchingOperandThroughUntouched(void) {
    tensor_t *in = buildSym(3, (int32_t[]){100, -200, 300}, 0.01f);
    tensor_t *out = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .mode = OUT_WRITE,
        },
        out);

    int32_t srcM0 = ((int32_t *)in->data)[0];
    float srcScale = ((symInt32QConfig_t *)in->quantization->qConfig)->scale;
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT(100, srcM0);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.01f, srcScale);
}

/* Prologue conversion: a FLOAT32 operand entering SYM arithmetic is
 * quantized into transient scratch; the result equals running the same
 * kernel on a pre-converted operand, and the float source is untouched. */
void testPrologueConvertsFloatOperandIntoSymArithmetic(void) {
    tensor_t *in = buildFloat(3, (float[]){0.5f, -1.0f, 1.5f});
    tensor_t *out = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .mode = OUT_WRITE,
        },
        out);

    /* Reference: quantize the float input to int12 SYM directly, then requant
     * to the target exactly as the epilogue's OUT_WRITE diagonal does. */
    tensor_t *ref = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    tensor_t *inSym = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    convertTensor(in, inSym, NULL);
    conversionMatrix[SYM_INT32][SYM_INT32](inSym, ref, NULL);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)out->data)[i];
        want[i] = ((int32_t *)ref->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)out->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)ref->quantization->qConfig)->scale;
    float srcV0 = ((float *)in->data)[0];
    freeTensor(inSym);
    freeTensor(ref);
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, srcV0); /* source untouched */
}

static void probeSum3Kernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                            tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    (void)ctx;
    size_t n = calcNumberOfElementsByTensor(rawOut);
    float *out = (float *)rawOut->data;
    for (size_t i = 0; i < n; i++) {
        out[i] = 0.f;
        for (size_t k = 0; k < nOperands; k++) {
            out[i] += ((const float *)operands[k]->data)[i];
        }
    }
}

/* inputs: FLOAT32 (passthrough), SYM_INT32 (row), SYM_INT32 (row) under
 * ARITH_FLOAT32 -> result = elementwise sum of dequantized values. Pins the
 * prologue's conversion correctness across MULTIPLE rows -- the offset
 * arithmetic of the Task-2 rewrite lives or dies here. `b`/`c` are seeded
 * so every mantissa round-trips losslessly at default qMaxBits=12 (absmax
 * element hits qMax=2047 exactly; the third element is an exact multiple of
 * the resulting quantum) -- so the sums below are exact, not just "close". */
void testExecuteOpConvertsOnlyMismatchedOperands(void) {
    tensor_t *a = buildFloat(3, (float[]){0.5f, 1.0f, -1.5f});
    tensor_t *b = buildSymInt32Tensor1dFromFloats(3, (float[]){2.047f, -2.047f, 0.5f});
    tensor_t *c = buildSymInt32Tensor1dFromFloats(3, (float[]){1.0f, 1.0f, 1.0f});
    tensor_t *out = buildFloat(3, (float[]){0, 0, 0});
    executeOp(
        &(opSpec_t){.kernel = probeSum3Kernel,
                    .inputs = (tensor_t *[]){a, b, c},
                    .nInputs = 3,
                    .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                    .mode = OUT_WRITE,
                    .auxOut = NULL},
        out);
    float r0 = ((float *)out->data)[0], r1 = ((float *)out->data)[1], r2 = ((float *)out->data)[2];
    freeTensor(a);
    freeTensor(b);
    freeTensor(c);
    freeTensor(out);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.547f, r0);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.047f, r1);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, r2);
}

/* OUT_WRITE SYM->SYM must be bit-identical to requantSymInt32Tensor (the
 * conversionMatrix diagonal) — NOT convertTensor's same-type memmove. Uses
 * accumulator-range mantissas so the memmove-vs-requant difference is huge. */
void testOutWriteSymToSymBitEqualsDiagonalRequant(void) {
    tensor_t *in = buildSym(3, (int32_t[]){1500000, -750000, 250000}, 1e-4f);
    tensor_t *out = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .mode = OUT_WRITE,
        },
        out);

    tensor_t *ref = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    conversionMatrix[SYM_INT32][SYM_INT32](in, ref, NULL);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)out->data)[i];
        want[i] = ((int32_t *)ref->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)out->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)ref->quantization->qConfig)->scale;
    freeTensor(ref);
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
}

/* OUT_WRITE SYM->FLOAT32 equals convertTensor (dequantization). */
void testOutWriteSymToFloatEqualsConvertTensor(void) {
    tensor_t *in = buildSym(2, (int32_t[]){12000, -6000}, 0.5f);
    tensor_t *out = buildFloat(2, (float[]){0.f, 0.f});
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .mode = OUT_WRITE,
        },
        out);

    float got0 = ((float *)out->data)[0];
    float got1 = ((float *)out->data)[1];
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_FLOAT(6000.0f, got0); /* 12000 * 0.5 — anti-vacuity */
    TEST_ASSERT_EQUAL_FLOAT(-3000.0f, got1);
}

/* Exact float accumulation: pre-seeded target, accumulate twice, exact sums.
 * Catches overwrite-instead-of-add and wrong-target mutations. */
void testAccDynamicIntoFloatIsExactAndAccumulates(void) {
    tensor_t *inc = buildFloat(3, (float[]){0.25f, -0.5f, 1.0f});
    tensor_t *grad = buildFloat(3, (float[]){1.0f, 1.0f, 1.0f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        grad);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        grad);

    float g0 = ((float *)grad->data)[0];
    float g1 = ((float *)grad->data)[1];
    float g2 = ((float *)grad->data)[2];
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_FLOAT(1.5f, g0); /* 1.0 + 2*0.25 — exact */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, g1);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, g2);
}

/* OUT_ACC_FIXED_SCALE on a FLOAT32 target collapses to the same exact add. */
void testAccFixedIntoFloatCollapsesToExactAdd(void) {
    tensor_t *inc = buildFloat(2, (float[]){0.5f, -0.25f});
    tensor_t *grad = buildFloat(2, (float[]){2.0f, 2.0f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        grad);

    float g0 = ((float *)grad->data)[0];
    float g1 = ((float *)grad->data)[1];
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_FLOAT(2.5f, g0);
    TEST_ASSERT_EQUAL_FLOAT(1.75f, g1);
}

/* Dynamic SYM+SYM must be bit-identical to addSymInt32TensorsInplace. */
void testAccDynamicSymIntoSymBitEqualsAddSymInplace(void) {
    tensor_t *inc = buildSym(3, (int32_t[]){1000, -2000, 500}, 0.001f);
    tensor_t *grad = buildSym(3, (int32_t[]){10, 20, 30}, 0.05f);
    tensor_t *refInc = buildSym(3, (int32_t[]){1000, -2000, 500}, 0.001f);
    tensor_t *refGrad = buildSym(3, (int32_t[]){10, 20, 30}, 0.05f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        grad);
    addSymInt32TensorsInplace(refGrad, refInc, NULL);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)grad->data)[i];
        want[i] = ((int32_t *)refGrad->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)grad->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)refGrad->quantization->qConfig)->scale;
    freeTensor(refGrad);
    freeTensor(refInc);
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
}

/* Dynamic FLOAT32 intermediate + SYM target: the increment is quantized to
 * operand-width SYM (roundingMode from the TARGET) then Strategy-A-added —
 * bit-identical to the LayerNorm.c:446-463 reference emulated here. */
void testAccDynamicFloatIncIntoSymTargetMatchesLayerNormReference(void) {
    float incVals[3] = {0.3f, -0.7f, 0.1f};
    tensor_t *inc = buildFloat(3, incVals);
    tensor_t *grad = buildSym(3, (int32_t[]){40, -80, 120}, 0.02f);
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        grad);

    /* Reference: exact LayerNorm.c:446-463 sequence on a twin. */
    tensor_t *refGrad = buildSym(3, (int32_t[]){40, -80, 120}, 0.02f);
    tensor_t *refIncFloat = buildFloat(3, incVals);
    tensor_t *refIncSym = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f); /* int12 shell, HALF_AWAY */
    convertTensor(refIncFloat, refIncSym, NULL);
    addSymInt32TensorsInplace(refGrad, refIncSym, NULL);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)grad->data)[i];
        want[i] = ((int32_t *)refGrad->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)grad->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)refGrad->quantization->qConfig)->scale;
    freeTensor(refIncSym);
    freeTensor(refIncFloat);
    freeTensor(refGrad);
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
}

/* Fixed-scale SYM+SYM reproduces linearCalcBiasGradsSymInt32's semantics via
 * rescaleIntoAccumulatorScale(interm[i], intermScale, targetScale, HALF_AWAY),
 * NO clamp; the target's roundingMode here is HALF_AWAY (default of buildSym),
 * so this pins the PR1b.2 D4 bit-identical-to-old-behavior case (roundHalfAway ==
 * the pre-migration bare roundf for these inputs). Hand-computed: interm
 * {700, -300} @ 0.01 into target {5, -5} @ 0.02: 700*0.01/0.02 = 350 (exact)
 * -> 355; -300*0.01/0.02 = -150 (exact) -> -155. Target scale must stay
 * EXACTLY 0.02 (never re-derived). */
void testAccFixedSymIntoSymRescalesIntoExistingScale(void) {
    tensor_t *inc = buildSym(2, (int32_t[]){700, -300}, 0.01f);
    tensor_t *grad = buildSym(2, (int32_t[]){5, -5}, 0.02f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        grad);

    int32_t g0 = ((int32_t *)grad->data)[0];
    int32_t g1 = ((int32_t *)grad->data)[1];
    float gScale = ((symInt32QConfig_t *)grad->quantization->qConfig)->scale;
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_INT(355, g0);
    TEST_ASSERT_EQUAL_INT(-155, g1);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.02f, gScale);
}

/* CONTRACT FLIP (PR3 Task 3, #261): this test previously pinned "packed
 * sub-byte (SYM) targets abort under accumulate" -- that restriction is
 * lifted by this task's new `case SYM:` arm in accumulateOut. Flipped into
 * the OUT_ACC_FIXED_SCALE happy path: a fresh (post-initTensor all-zero)
 * SYM@8 target derives its grid from the first increment, then a second
 * accumulate carries that grid verbatim (fit-preserving, packed-grad D1/D2). One of
 * exactly two sanctioned contract-flip test edits in this PR (the other is UnitTestSgd's admission
 * flip — the PR2 ASYM-zero test's config-reset extension is additive, not a flip). Parity oracle: a
 * twin target driven directly through accumulateFloatIntoSymTensorFixedGrid must end up
 * bit-identical to the executeOp-driven target. inc2 deliberately SHRINKS the element that hit the
 * grid boundary on call 1 (index 0, derived to exactly +127) while nudging
 * the others -- carrying the grid keeps scale unchanged, but a mutant that
 * swaps in the DYNAMIC_RESCALE primitive for FIXED_SCALE would re-derive from
 * the new (smaller) absmax and land on a visibly different, smaller scale;
 * this is what makes the "swap FIXED/DYNAMIC primitive calls" mutation
 * (Step 5) observable here (an unperturbed already-maxed element would let a
 * fresh re-derivation reproduce the same tight grid by coincidence -- the
 * Task-2-report lesson on vacuous carried-grid fixtures). */
void testAccFixedIntoPackedSymDerivesThenCarriesGrid(void) {
    size_t n = 3;
    tensor_t *target = buildPackedSym(n, (int32_t[]){0, 0, 0}, 8, 1.0f);
    tensor_t *ref = buildPackedSym(n, (int32_t[]){0, 0, 0}, 8, 1.0f);
    tensor_t *inc1T = buildFloat(n, (float[]){2.0f, -0.9f, 0.37f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc1T},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        target);
    float inc1[] = {2.0f, -0.9f, 0.37f};
    accumulateFloatIntoSymTensorFixedGrid(ref, inc1, n, NULL);
    float scaleAfterCall1 = ((symQConfig_t *)target->quantization->qConfig)->scales[0];

    tensor_t *inc2T = buildFloat(n, (float[]){-1.8f, 0.05f, -0.1f});
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc2T},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        target);
    float inc2[] = {-1.8f, 0.05f, -0.1f};
    accumulateFloatIntoSymTensorFixedGrid(ref, inc2, n, NULL);

    int32_t got[3];
    int32_t want[3];
    symTestUnpackSignExtend(target->data, 8, got, n);
    symTestUnpackSignExtend(ref->data, 8, want, n);
    float gotScale = ((symQConfig_t *)target->quantization->qConfig)->scales[0];
    float wantScale = ((symQConfig_t *)ref->quantization->qConfig)->scales[0];
    freeTensor(inc2T);
    freeTensor(inc1T);
    freeTensor(ref);
    freeTensor(target);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, n);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
    TEST_ASSERT_EQUAL_FLOAT(scaleAfterCall1, gotScale); /* carried, not re-derived */
}

/* The float-bridge closure: a SYM_INT32 intermediate must reach
 * the SAME packed-SYM result as a value-identical FLOAT32 intermediate, for
 * BOTH FIXED_SCALE (tested here) since the epilogue dequantizes either
 * representation before calling the primitive. Fixture chosen so the
 * SYM_INT32 intermediate's dequant (mantissa*scale) is bit-exact against the
 * FLOAT32 sibling's literals (5*0.25=1.25, -2*0.25=-0.5, 8*0.25=2.0 -- all
 * exact binary fractions), so the two targets must end up byte-for-byte and
 * scale-for-scale identical, not merely within tolerance.
 * Mutation guard: dropping the SYM_INT32 branch (e.g. always treating the
 * intermediate as FLOAT32, misreading its raw int32 bits as float) makes
 * targetViaSymInt32 diverge from targetViaFloat -- RED. */
void testAccFixedSymPackedAcceptsSymInt32IntermediateBitIdenticalToFloatBridge(void) {
    size_t n = 3;
    int32_t seedMant[] = {10, -20, 5};
    tensor_t *targetViaFloat = buildPackedSym(n, seedMant, 8, 0.05f);
    tensor_t *targetViaSymInt32 = buildPackedSym(n, seedMant, 8, 0.05f);

    tensor_t *incFloat = buildFloat(n, (float[]){1.25f, -0.5f, 2.0f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){incFloat},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        targetViaFloat);

    tensor_t *incSymInt32 = buildSym(n, (int32_t[]){5, -2, 8}, 0.25f);
    quantization_t symArith;
    symInt32QConfig_t symArithQC;
    initSymInt32QConfig(HALF_AWAY, &symArithQC);
    initSymInt32Quantization(&symArithQC, &symArith);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){incSymInt32},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&symArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        targetViaSymInt32);

    int32_t gotFloat[3];
    int32_t gotSymInt32[3];
    symTestUnpackSignExtend(targetViaFloat->data, 8, gotFloat, n);
    symTestUnpackSignExtend(targetViaSymInt32->data, 8, gotSymInt32, n);
    float scaleFloat = ((symQConfig_t *)targetViaFloat->quantization->qConfig)->scales[0];
    float scaleSymInt32 = ((symQConfig_t *)targetViaSymInt32->quantization->qConfig)->scales[0];
    freeTensor(incSymInt32);
    freeTensor(incFloat);
    freeTensor(targetViaSymInt32);
    freeTensor(targetViaFloat);
    TEST_ASSERT_EQUAL_INT32_ARRAY(gotFloat, gotSymInt32, n);
    TEST_ASSERT_EQUAL_FLOAT(scaleFloat, scaleSymInt32);
}

/* DYNAMIC_RESCALE on a packed SYM target must match
 * accumulateFloatIntoSymTensorRescale exactly and must actually RE-DERIVE the
 * grid from the new absmax (not carry the old one) -- the defining
 * difference from the FIXED_SCALE arm above. Seed scale 0.1 is small; the
 * increment (+-50) dwarfs the seed values, so a correctly re-derived scale
 * must land close to 50.4/127 (~0.397), several times the old 0.1 --
 * anti-vacuity against a mutant that carries the old scale instead. */
void testAccDynamicSymPackedRederivesScaleMatchesRescalePrimitive(void) {
    size_t n = 2;
    int32_t seedMant[] = {4, -2};
    tensor_t *target = buildPackedSym(n, seedMant, 8, 0.1f);
    tensor_t *ref = buildPackedSym(n, seedMant, 8, 0.1f);
    tensor_t *inc = buildFloat(n, (float[]){50.0f, -50.0f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        target);
    float incVals[] = {50.0f, -50.0f};
    accumulateFloatIntoSymTensorRescale(ref, incVals, n, NULL);

    int32_t got[2];
    int32_t want[2];
    symTestUnpackSignExtend(target->data, 8, got, n);
    symTestUnpackSignExtend(ref->data, 8, want, n);
    float gotScale = ((symQConfig_t *)target->quantization->qConfig)->scales[0];
    float wantScale = ((symQConfig_t *)ref->quantization->qConfig)->scales[0];
    freeTensor(inc);
    freeTensor(ref);
    freeTensor(target);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, n);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
    TEST_ASSERT_TRUE(gotScale > 3.0f * 0.1f); /* re-derived (~0.397), not carried at 0.1 */
}

/* DYNAMIC_RESCALE analog of
 * testAccFixedSymPackedAcceptsSymInt32IntermediateBitIdenticalToFloatBridge:
 * a SYM_INT32 intermediate must reach the SAME packed-SYM result as a
 * value-identical FLOAT32 intermediate under OUT_ACC_DYNAMIC_RESCALE too.
 * The FIXED_SCALE bridge test never reaches accumulateTensorIntoSymRescale
 * (the tensor-typed streamed entry point for this mode); this is its only
 * executeOp-level coverage. Fixture identical to the FIXED_SCALE bridge test
 * (5*0.25=1.25, -2*0.25=-0.5, 8*0.25=2.0 -- exact binary fractions).
 * Mutation guard: dropping the SYM_INT32 branch (e.g. always treating the
 * intermediate as FLOAT32, misreading its raw int32 bits as float) makes
 * targetViaSymInt32 diverge from targetViaFloat -- RED. */
void testAccDynamicSymPackedAcceptsSymInt32IntermediateBitIdenticalToFloatBridge(void) {
    size_t n = 3;
    int32_t seedMant[] = {10, -20, 5};
    tensor_t *targetViaFloat = buildPackedSym(n, seedMant, 8, 0.05f);
    tensor_t *targetViaSymInt32 = buildPackedSym(n, seedMant, 8, 0.05f);

    tensor_t *incFloat = buildFloat(n, (float[]){1.25f, -0.5f, 2.0f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){incFloat},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        targetViaFloat);

    tensor_t *incSymInt32 = buildSym(n, (int32_t[]){5, -2, 8}, 0.25f);
    quantization_t symArith;
    symInt32QConfig_t symArithQC;
    initSymInt32QConfig(HALF_AWAY, &symArithQC);
    initSymInt32Quantization(&symArithQC, &symArith);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){incSymInt32},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&symArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        targetViaSymInt32);

    int32_t gotFloat[3];
    int32_t gotSymInt32[3];
    symTestUnpackSignExtend(targetViaFloat->data, 8, gotFloat, n);
    symTestUnpackSignExtend(targetViaSymInt32->data, 8, gotSymInt32, n);
    float scaleFloat = ((symQConfig_t *)targetViaFloat->quantization->qConfig)->scales[0];
    float scaleSymInt32 = ((symQConfig_t *)targetViaSymInt32->quantization->qConfig)->scales[0];
    freeTensor(incSymInt32);
    freeTensor(incFloat);
    freeTensor(targetViaSymInt32);
    freeTensor(targetViaFloat);
    TEST_ASSERT_EQUAL_INT32_ARRAY(gotFloat, gotSymInt32, n);
    TEST_ASSERT_EQUAL_FLOAT(scaleFloat, scaleSymInt32);
}

/* ASYM DYNAMIC_RESCALE happy path (packed-grad D4: the only supported ASYM accumulate
 * mode) must match accumulateFloatIntoAsymTensorRescale exactly (fresh
 * affine grid every store). Fixture matches Task 2's own ASYM-rescale
 * fixture: ASYM@5 codes {12,16,20,24} @
 * scale=0.25/zeroPoint=+4 (code-domain, PR4) dequant to {2,3,4,5}. */
void testAccDynamicAsymPackedMatchesRescalePrimitive(void) {
    size_t n = 4;
    int32_t seedCodes[] = {12, 16, 20, 24};
    tensor_t *target = buildAsymPacked(n, seedCodes, 5, 0.25f, 4);
    tensor_t *ref = buildAsymPacked(n, seedCodes, 5, 0.25f, 4);
    tensor_t *inc = buildFloat(n, (float[]){1.0f, -0.5f, 2.0f, 0.25f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        target);
    float incVals[] = {1.0f, -0.5f, 2.0f, 0.25f};
    accumulateFloatIntoAsymTensorRescale(ref, incVals, n, NULL);

    int32_t got[4];
    int32_t want[4];
    byteConversion(target->data, 5, (uint8_t *)got, 32,
                   n); /* asym codes: non-negative, no sign-extend */
    byteConversion(ref->data, 5, (uint8_t *)want, 32, n);
    float gotScale = ((asymQConfig_t *)target->quantization->qConfig)->scales[0];
    float wantScale = ((asymQConfig_t *)ref->quantization->qConfig)->scales[0];
    uint16_t gotZp = ((asymQConfig_t *)target->quantization->qConfig)->zeroPoints[0];
    uint16_t wantZp = ((asymQConfig_t *)ref->quantization->qConfig)->zeroPoints[0];
    freeTensor(inc);
    freeTensor(ref);
    freeTensor(target);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, n);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
    TEST_ASSERT_EQUAL_INT32(wantZp, gotZp);
}

/* SYM_INT32 intermediate bridging analog of the SYM DYNAMIC_RESCALE bridge
 * test above, for ASYM targets -- exercises accumulateTensorIntoAsymRescale,
 * the tensor-typed streamed entry point (the FLOAT32-intermediate coverage
 * above never reaches this branch). Fixture: incSymInt32 dequantizes
 * bit-exact to the same values as incFloat (4*0.25=1.0, -2*0.25=-0.5,
 * 8*0.25=2.0, 1*0.25=0.25 -- exact binary fractions).
 * Mutation guard: dropping the SYM_INT32 branch makes targetViaSymInt32
 * diverge from targetViaFloat -- RED. */
void testAccDynamicAsymPackedAcceptsSymInt32IntermediateBitIdenticalToFloatBridge(void) {
    size_t n = 4;
    int32_t seedCodes[] = {12, 16, 20, 24};
    tensor_t *targetViaFloat = buildAsymPacked(n, seedCodes, 5, 0.25f, 4);
    tensor_t *targetViaSymInt32 = buildAsymPacked(n, seedCodes, 5, 0.25f, 4);

    tensor_t *incFloat = buildFloat(n, (float[]){1.0f, -0.5f, 2.0f, 0.25f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){incFloat},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        targetViaFloat);

    tensor_t *incSymInt32 = buildSym(n, (int32_t[]){4, -2, 8, 1}, 0.25f);
    quantization_t symArith;
    symInt32QConfig_t symArithQC;
    initSymInt32QConfig(HALF_AWAY, &symArithQC);
    initSymInt32Quantization(&symArithQC, &symArith);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){incSymInt32},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&symArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        targetViaSymInt32);

    int32_t gotFloat[4];
    int32_t gotSymInt32[4];
    byteConversion(targetViaFloat->data, 5, (uint8_t *)gotFloat, 32, n);
    byteConversion(targetViaSymInt32->data, 5, (uint8_t *)gotSymInt32, 32, n);
    float scaleFloat = ((asymQConfig_t *)targetViaFloat->quantization->qConfig)->scales[0];
    float scaleSymInt32 = ((asymQConfig_t *)targetViaSymInt32->quantization->qConfig)->scales[0];
    uint16_t zpFloat = ((asymQConfig_t *)targetViaFloat->quantization->qConfig)->zeroPoints[0];
    uint16_t zpSymInt32 =
        ((asymQConfig_t *)targetViaSymInt32->quantization->qConfig)->zeroPoints[0];
    freeTensor(incSymInt32);
    freeTensor(incFloat);
    freeTensor(targetViaSymInt32);
    freeTensor(targetViaFloat);
    TEST_ASSERT_EQUAL_INT32_ARRAY(gotFloat, gotSymInt32, n);
    TEST_ASSERT_EQUAL_FLOAT(scaleFloat, scaleSymInt32);
    TEST_ASSERT_EQUAL_INT32(zpFloat, zpSymInt32);
}

/* Grad-width contract moved from layerNormValidateSymGrad: SYM targets wider
 * than ODT_SYM_GRAD_QMAXBITS abort. */
void testAccIntoTooWideSymTargetAborts(void) {
    tensor_t *inc = buildFloat(2, (float[]){1.f, 2.f});
    tensor_t *grad = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    ((symInt32QConfig_t *)grad->quantization->qConfig)->qMaxBits = 24; /* > 16 */
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    ASSERT_EXITS_WITH_FAILURE(executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        grad));

    freeTensor(grad);
    freeTensor(inc);
}

/* The same ODT_SYM_GRAD_QMAXBITS(16) contract applies to packed SYM targets
 * (the mirror of the existing SYM_INT32 guard). qBits=17 packs and
 * unpacks fine on its own (packChunkGuarded allows up to 31) -- only the grad
 * contract rejects it. */
void testAccIntoTooWidePackedSymTargetAborts(void) {
    size_t n = 2;
    tensor_t *target = buildPackedSym(n, (int32_t[]){0, 0}, 17, 1.0f); /* 17 > 16 */
    tensor_t *inc = buildFloat(n, (float[]){1.f, 2.f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    ASSERT_EXITS_WITH_FAILURE(executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        target));

    freeTensor(target);
    freeTensor(inc);
}

/* packed-grad D4: no fit-preserving ASYM pack exists, so OUT_ACC_FIXED_SCALE on an ASYM
 * target must abort rather than silently behave like DYNAMIC_RESCALE.
 * Mutation guard: dropping this guard (falling through to the rescale path)
 * lets the child exit 0 -- RED. */
void testAccFixedScaleOnAsymTargetAborts(void) {
    size_t n = 2;
    tensor_t *target = buildAsymPacked(n, (int32_t[]){0, 0}, 8, 1.0f, 0);
    tensor_t *inc = buildFloat(n, (float[]){1.f, -1.f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    ASSERT_EXITS_WITH_FAILURE(executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        target));

    freeTensor(target);
    freeTensor(inc);
}

/* Stack BFP target for the accumulateOut arm (epic PR3): codes packed
 * verbatim at mantissaBits, exponents/geometry field-assigned -- the seeding
 * idiom of the BFP operand tests below and UnitTestTensorConversion.c. */
static void buildBfpTarget(tensor_t *t, uint8_t *data, shape_t *shape, quantization_t *q,
                           bfpQConfig_t *qc, const int32_t *codes, size_t n) {
    initBfpQuantization(qc, q);
    byteConversion((uint8_t *)codes, 32, data, qc->mantissaBits, n);
    setTensorValues(t, data, shape, q, NULL);
}

/* DYNAMIC_RESCALE into a BFP-stored grad target: the epilogue must re-derive
 * every group's exponent from the decoded-plus-increment absmax (the
 * RederivesExponents engine fixture, all steps exact): group0 stored
 * 127 -> 126, group1 125 -> 127 (over a power-of-two boundary), codes
 * {20,-6,16,-4}. Mutation guard: dispatching the FixedGrid twin instead
 * carries the old grid -> group1's value 16 at scale 0.25 is code 64,
 * overflowing 6 bits -> abort, RED by crash. */
void testAccDynamicIntoBfpTargetRederivesExponents(void) {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    uint8_t exponents[2] = {127, 125};
    bfpQConfig_t qc = {.exponents = exponents,
                       .numGroups = 2,
                       .groupSize = 2,
                       .roundingMode = HALF_AWAY,
                       .mantissaBits = 6,
                       .exponentBits = 8};
    quantization_t q;
    uint8_t data[3];
    tensor_t target;
    buildBfpTarget(&target, data, &shape, &q, &qc, (int32_t[]){8, -4, 16, 8}, n);

    tensor_t *inc = buildFloat(n, (float[]){2.f, 1.f, 12.f, -6.f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        &target);

    int32_t got[4];
    symTestUnpackSignExtend(data, 6, got, n);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_UINT8(126, exponents[0]);
    TEST_ASSERT_EQUAL_UINT8(127, exponents[1]);
    int32_t expected[4] = {20, -6, 16, -4};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, n);
}

/* FIXED_SCALE into a zero-state BFP target (the packed-SYM
 * DerivesThenCarriesGrid clone): call 1 derives the per-group grid from the
 * increment ({127,129}), call 2 must CARRY it verbatim. Both calls are
 * checked byte-for-byte against the float primitive. Anti-vacuity of the
 * carried-grid assert: call 2's group1 sums SHRINK to absmax 4, so a
 * mode-swap mutant (Rescale dispatched instead) re-derives stored 127 and
 * codes {4,0} where carry keeps 129 and {1,0} -- rederive does NOT coincide
 * with carry here (a first-draft inc2 where they coincided was vacuous;
 * house throwaway-harness cross-check). */
void testAccFixedIntoBfpTargetCarriesGrid(void) {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    uint8_t exponents[2] = {127, 127};
    bfpQConfig_t qc = {.exponents = exponents,
                       .numGroups = 2,
                       .groupSize = 2,
                       .roundingMode = HALF_AWAY,
                       .mantissaBits = 4,
                       .exponentBits = 8};
    quantization_t q;
    uint8_t data[2];
    tensor_t target;
    buildBfpTarget(&target, data, &shape, &q, &qc, (int32_t[]){0, 0, 0, 0}, n);

    uint8_t refExponents[2] = {127, 127};
    bfpQConfig_t refQC = {.exponents = refExponents,
                          .numGroups = 2,
                          .groupSize = 2,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t refQ;
    uint8_t refData[2];
    tensor_t ref;
    buildBfpTarget(&ref, refData, &shape, &refQ, &refQC, (int32_t[]){0, 0, 0, 0}, n);

    quantization_t floatArith;
    initFloat32Quantization(&floatArith);
    tensor_t *inc1T = buildFloat(n, (float[]){6.f, 1.f, 28.f, -7.f});
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc1T},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        &target);
    float inc1[] = {6.f, 1.f, 28.f, -7.f};
    accumulateFloatIntoBfpTensorFixedGrid(&ref, inc1, n, NULL);
    uint8_t expAfterCall1[2] = {exponents[0], exponents[1]};

    tensor_t *inc2T = buildFloat(n, (float[]){1.f, -1.f, -24.f, 8.f});
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc2T},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        &target);
    float inc2[] = {1.f, -1.f, -24.f, 8.f};
    accumulateFloatIntoBfpTensorFixedGrid(&ref, inc2, n, NULL);

    int32_t got[4];
    symTestUnpackSignExtend(data, 4, got, n);
    freeTensor(inc2T);
    freeTensor(inc1T);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refExponents, exponents, 2);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refData, data, 2);
    TEST_ASSERT_EQUAL_UINT8(expAfterCall1[0], exponents[0]); /* carried, not re-derived */
    TEST_ASSERT_EQUAL_UINT8(expAfterCall1[1], exponents[1]);
    TEST_ASSERT_EQUAL_UINT8(127, exponents[0]);
    TEST_ASSERT_EQUAL_UINT8(129, exponents[1]);
    int32_t expected[4] = {7, 0, 1, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, n);
}

/* The BFP arm's tensor-typed branch: an ARITH_SYM_INT32 raw intermediate
 * (identity kernel propagates mantissas {8,4,48,-24} AND scale 0.25 -- exact
 * dequant image {2,1,12,-6}) must reach the SAME result as the tensor twin
 * primitive on identical data -- the RederivesExponents golds. Mutation
 * guard: a mode-swap in the tensor-typed branch (FixedGrid dispatched under
 * DYNAMIC) carries the old grid -> group1 value 16 at scale 0.25 is code 64
 * -> 6-bit overflow abort, RED by crash. */
void testAccDynamicSymInt32IntermediateIntoBfpTargetMatchesTensorTwin(void) {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    uint8_t exponents[2] = {127, 125};
    bfpQConfig_t qc = {.exponents = exponents,
                       .numGroups = 2,
                       .groupSize = 2,
                       .roundingMode = HALF_AWAY,
                       .mantissaBits = 6,
                       .exponentBits = 8};
    quantization_t q;
    uint8_t data[3];
    tensor_t target;
    buildBfpTarget(&target, data, &shape, &q, &qc, (int32_t[]){8, -4, 16, 8}, n);

    uint8_t refExponents[2] = {127, 125};
    bfpQConfig_t refQC = {.exponents = refExponents,
                          .numGroups = 2,
                          .groupSize = 2,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 6,
                          .exponentBits = 8};
    quantization_t refQ;
    uint8_t refData[3];
    tensor_t ref;
    buildBfpTarget(&ref, refData, &shape, &refQ, &refQC, (int32_t[]){8, -4, 16, 8}, n);

    tensor_t *incSymInt32 = buildSym(n, (int32_t[]){8, 4, 48, -24}, 0.25f);
    quantization_t symArith;
    symInt32QConfig_t symArithQC;
    initSymInt32QConfig(HALF_AWAY, &symArithQC);
    initSymInt32Quantization(&symArithQC, &symArith);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){incSymInt32},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&symArith),
            .mode = OUT_ACC_DYNAMIC_RESCALE,
        },
        &target);
    accumulateTensorIntoBfpRescale(&ref, incSymInt32, NULL);

    int32_t got[4];
    symTestUnpackSignExtend(data, 6, got, n);
    freeTensor(incSymInt32);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refExponents, exponents, 2);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refData, data, 3);
    TEST_ASSERT_EQUAL_UINT8(126, exponents[0]); /* the RederivesExponents golds */
    TEST_ASSERT_EQUAL_UINT8(127, exponents[1]);
    int32_t expected[4] = {20, -6, 16, -4};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, n);
}

/* The 4th cell of the BFP accumulateOut dispatch matrix (TENSOR intermediate
 * x OUT_ACC_FIXED_SCALE), the one the three tests above leave unpinned: an
 * ARITH_SYM_INT32 raw intermediate under FIXED_SCALE must reach the
 * FixedGrid tensor twin. Two calls on a zero-state target (the CarriesGrid
 * shape, mantissas * 0.25 = its float literals exactly): call 1 derives
 * {127,129} from the increment alone, call 2 must CARRY that grid -- its
 * group1 sums SHRINK to absmax 4, so a mode-swap mutant (Rescale dispatched
 * in this cell) re-derives stored 127 and codes {4,0} where carry keeps 129
 * and {1,0} (CarriesGrid's anti-vacuity inc2; call 1 alone cannot see the
 * swap because fresh-derive and rescale coincide on an all-zero target).
 * Byte-for-byte against the tensor twin primitive on identical data. */
void testAccFixedSymInt32IntermediateIntoBfpTargetCarriesGrid(void) {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    uint8_t exponents[2] = {127, 127};
    bfpQConfig_t qc = {.exponents = exponents,
                       .numGroups = 2,
                       .groupSize = 2,
                       .roundingMode = HALF_AWAY,
                       .mantissaBits = 4,
                       .exponentBits = 8};
    quantization_t q;
    uint8_t data[2];
    tensor_t target;
    buildBfpTarget(&target, data, &shape, &q, &qc, (int32_t[]){0, 0, 0, 0}, n);

    uint8_t refExponents[2] = {127, 127};
    bfpQConfig_t refQC = {.exponents = refExponents,
                          .numGroups = 2,
                          .groupSize = 2,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t refQ;
    uint8_t refData[2];
    tensor_t ref;
    buildBfpTarget(&ref, refData, &shape, &refQ, &refQC, (int32_t[]){0, 0, 0, 0}, n);

    quantization_t symArith;
    symInt32QConfig_t symArithQC;
    initSymInt32QConfig(HALF_AWAY, &symArithQC);
    initSymInt32Quantization(&symArithQC, &symArith);

    tensor_t *inc1 = buildSym(n, (int32_t[]){24, 4, 112, -28}, 0.25f);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc1},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&symArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        &target);
    accumulateTensorIntoBfpFixedGrid(&ref, inc1, NULL);
    uint8_t expAfterCall1[2] = {exponents[0], exponents[1]};

    tensor_t *inc2 = buildSym(n, (int32_t[]){4, -4, -96, 32}, 0.25f);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc2},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&symArith),
            .mode = OUT_ACC_FIXED_SCALE,
        },
        &target);
    accumulateTensorIntoBfpFixedGrid(&ref, inc2, NULL);

    int32_t got[4];
    symTestUnpackSignExtend(data, 4, got, n);
    freeTensor(inc2);
    freeTensor(inc1);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refExponents, exponents, 2);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refData, data, 2);
    TEST_ASSERT_EQUAL_UINT8(expAfterCall1[0], exponents[0]); /* carried, not re-derived */
    TEST_ASSERT_EQUAL_UINT8(expAfterCall1[1], exponents[1]);
    TEST_ASSERT_EQUAL_UINT8(127, exponents[0]);
    TEST_ASSERT_EQUAL_UINT8(129, exponents[1]);
    int32_t expected[4] = {7, 0, 1, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, n);
}

/* ---- opSpec_t: ctx / auxOut / FIXED_SCALE roundingMode (PR1b.2 D1+D4) --- */

typedef struct {
    float addend;
} testCtx_t;

/* Adapter proving ctx actually reaches the kernel: adds ctx->addend to every
 * element of operand 0 into rawOut. If the funnel dropped/NULLed ctx this
 * would crash (NULL deref) or - if silently defaulted - would leave the
 * output un-added; the pinned +10 catches either regression. */
static void kernelAddsCtx(tensor_t **ops, size_t nOps, tensor_t *rawOut, tensor_t *auxOut,
                          const void *ctx, rngStream_t *rs) {
    (void)nOps;
    (void)auxOut;
    const testCtx_t *c = (const testCtx_t *)ctx;
    size_t n = calcNumberOfElementsByTensor(ops[0]);
    float *src = (float *)ops[0]->data;
    float *dst = (float *)rawOut->data;
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] + c->addend;
    }
}

void testCtxReachesKernel(void) {
    tensor_t *in = buildFloat(2, (float[]){1.0f, -2.0f});
    tensor_t *out = buildFloat(2, (float[]){0.f, 0.f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);
    testCtx_t ctx = {.addend = 10.0f};

    executeOp(
        &(opSpec_t){
            .kernel = kernelAddsCtx,
            .ctx = &ctx,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_WRITE,
        },
        out);

    float got0 = ((float *)out->data)[0];
    float got1 = ((float *)out->data)[1];
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_FLOAT(11.0f, got0); /* 1.0 + ctx->addend(10) */
    TEST_ASSERT_EQUAL_FLOAT(8.0f, got1);  /* -2.0 + 10 */
}

/* Adapter writing distinct values into rawOut and auxOut, plus a sentinel SYM
 * scale on auxOut that no funnel machinery would ever produce on its own
 * (0.777f). auxOut is a DIFFERENT dtype (SYM_INT32) from the FLOAT32
 * arithmetic/target here specifically so that, if a future edit accidentally
 * routed auxOut through writeOut/accumulateOut (funnel-converting it), the
 * requant would recompute its scale from the data's absmax and destroy the
 * sentinel - this test would then fail. */
static void kernelWritesAuxVerbatim(tensor_t **ops, size_t nOps, tensor_t *rawOut, tensor_t *auxOut,
                                    const void *ctx, rngStream_t *rs) {
    (void)nOps;
    (void)ctx;
    size_t n = calcNumberOfElementsByTensor(ops[0]);
    float *src = (float *)ops[0]->data;
    float *dst = (float *)rawOut->data;
    int32_t *aux = (int32_t *)auxOut->data;
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i];
        aux[i] = (int32_t)(i + 7); /* sentinel indices, unrelated to op values */
    }
    ((symInt32QConfig_t *)auxOut->quantization->qConfig)->scale = 0.777f;
}

void testAuxOutIsKernelWrittenVerbatimAndNeverFunnelConverted(void) {
    tensor_t *in = buildFloat(2, (float[]){3.0f, -4.0f});
    tensor_t *out = buildFloat(2, (float[]){0.f, 0.f});
    tensor_t *aux = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(
        &(opSpec_t){
            .kernel = kernelWritesAuxVerbatim,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&floatArith),
            .mode = OUT_WRITE,
            .auxOut = aux,
        },
        out);

    float outGot0 = ((float *)out->data)[0];
    float outGot1 = ((float *)out->data)[1];
    int32_t auxGot0 = ((int32_t *)aux->data)[0];
    int32_t auxGot1 = ((int32_t *)aux->data)[1];
    float auxScale = ((symInt32QConfig_t *)aux->quantization->qConfig)->scale;
    freeTensor(aux);
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, outGot0); /* target: untouched by auxOut content */
    TEST_ASSERT_EQUAL_FLOAT(-4.0f, outGot1);
    TEST_ASSERT_EQUAL_INT32(7, auxGot0); /* auxOut: kernel's verbatim write survives */
    TEST_ASSERT_EQUAL_INT32(8, auxGot1);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.777f, auxScale); /* sentinel scale untouched */
}

/* OUT_ACC_FIXED_SCALE must consult the TARGET's roundingMode (PR1b.2 D4) via
 * rescaleIntoAccumulatorScale, not a bare roundf (which is HALF_AWAY-only and
 * ignores SR jitter entirely). Values are chosen so the pre-rescale ratio
 * lands exactly on a HALF_AWAY tie (705*0.01/0.02 = 352.5, -705*... = -352.5)
 * so the two rounding modes provably diverge for a jitter draw that lands
 * before the tie boundary.
 *
 * The rescale draws twice (one per element, in element order) from the op
 * key's ACC(0) stream, fold(key, ACC = 11, 0), key = rngKeyFromSeed(31):
 *   uv run python -c "from examples._shared.keyed_rng import (fold, key_from_seed as kfs,
 *   open_stream as o, next_unit_float as u); s = o(fold(kfs(31), 11, 0)); s, j0 = u(s); print(
 *   j0, u(s)[1])"
 * prints j0 = 0.0817152..., j1 = 0.7940227... :
 *   HALF_AWAY:   round(352.5)  = 353 ;  round(-352.5) = -353  (the OLD bare-
 *                roundf behavior this test would still show if the epilogue
 *                ignored the target's roundingMode - anti-vacuity)
 *   SR_HALF_AWAY: round(352.5 + j0 - 0.5) = round(352.08...) = 352
 *                 round(-352.5 + j1 - 0.5) = round(-352.21...) = -352
 * Target scale must stay EXACTLY 0.02 (never re-derived), matching
 * testAccFixedSymIntoSymRescalesIntoExistingScale's HALF_AWAY pin (PR1b.2 D4's
 * "bit-identical for HALF_AWAY" case). */
#define SR_GOLD_KEY_SEED 31u

void testAccFixedScaleHonorsTargetSrRoundingMode(void) {
    tensor_t *inc = buildSym(2, (int32_t[]){705, -705}, 0.01f);
    tensor_t *grad = buildSym(2, (int32_t[]){0, 0}, 0.02f);
    ((symInt32QConfig_t *)grad->quantization->qConfig)->roundingMode = SR_HALF_AWAY;
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC); /* arithmetic's own roundingMode is
                                               * irrelevant to FIXED_SCALE's
                                               * epilogue rescale (target's is
                                               * what counts, per PR1b.2 D4) */
    initSymInt32Quantization(&arithQC, &arith);
    const rngKey_t key = rngKeyFromSeed(SR_GOLD_KEY_SEED);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){inc},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .key = &key,
            .mode = OUT_ACC_FIXED_SCALE,
        },
        grad);

    int32_t g0 = ((int32_t *)grad->data)[0];
    int32_t g1 = ((int32_t *)grad->data)[1];
    float gScale = ((symInt32QConfig_t *)grad->quantization->qConfig)->scale;
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_INT(352, g0);
    TEST_ASSERT_EQUAL_INT(-352, g1);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.02f, gScale);
}

/* ---- #296 Stage 1: writesInPlaceSafe / rawData aliasing (OUT_WRITE only,
 * FLOAT32 arithmetic + FLOAT32 target) ------------------------------------ */

/* probeAddOneKernel is elementwise (reads in[i] before writing out[i]). */
static const void *g_probeRawOutData;
static void probeAddOneKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                              tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)nOperands;
    (void)auxOut;
    (void)ctx;
    g_probeRawOutData = rawOut->data;
    size_t n = calcNumberOfElementsByTensor(rawOut);
    const float *in = (const float *)operands[0]->data;
    float *out = (float *)rawOut->data;
    for (size_t i = 0; i < n; i++) {
        out[i] = in[i] + 1.0f;
    }
}

void testExecuteOpAliasesTargetForMatchingFloatWrite(void) {
    /* FLOAT32 arithmetic, FLOAT32 target, OUT_WRITE, target not an input:
     * the kernel must receive target->data directly (no rawData staging). */
    tensor_t *in = buildFloat(4, (float[]){1, 2, 3, 4});
    tensor_t *out = buildFloat(4, (float[]){0, 0, 0, 0});
    executeOp(
        &(opSpec_t){.kernel = probeAddOneKernel,
                    .ctx = NULL,
                    .inputs = (tensor_t *[]){in},
                    .nInputs = 1,
                    .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                    .mode = OUT_WRITE,
                    .auxOut = NULL},
        out);
    /* CAPTURE -> free -> assert per file convention. */
    const void *rawPtr = g_probeRawOutData;
    float o1 = ((float *)out->data)[1];
    freeTensor(in);
    /* assert BEFORE freeing out so the pointer comparison target is stable */
    TEST_ASSERT_EQUAL_PTR(out->data, rawPtr);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, o1);
    freeTensor(out);
}

void testExecuteOpDoesNotAliasWhenTargetIsAnInput(void) {
    /* target == inputs[0], flag NOT set -> funnel must stage via rawData. */
    tensor_t *t = buildFloat(4, (float[]){1, 2, 3, 4});
    executeOp(
        &(opSpec_t){.kernel = probeAddOneKernel,
                    .inputs = (tensor_t *[]){t},
                    .nInputs = 1,
                    .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                    .mode = OUT_WRITE,
                    .auxOut = NULL},
        t);
    const void *rawPtr = g_probeRawOutData;
    float v2 = ((float *)t->data)[2];
    TEST_ASSERT_TRUE_MESSAGE(rawPtr != t->data, "unsafe self-target must stage through rawData");
    TEST_ASSERT_EQUAL_FLOAT(4.0f, v2);
    freeTensor(t);
}

void testExecuteOpAliasesSelfTargetWithWritesInPlaceSafe(void) {
    /* Same self-target op, flag SET (kernel is elementwise) -> direct write. */
    tensor_t *t = buildFloat(4, (float[]){1, 2, 3, 4});
    executeOp(
        &(opSpec_t){.kernel = probeAddOneKernel,
                    .inputs = (tensor_t *[]){t},
                    .nInputs = 1,
                    .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                    .mode = OUT_WRITE,
                    .auxOut = NULL,
                    .writesInPlaceSafe = true},
        t);
    const void *rawPtr = g_probeRawOutData;
    float v2 = ((float *)t->data)[2];
    TEST_ASSERT_EQUAL_PTR(t->data, rawPtr);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, v2);
    freeTensor(t);
}

void testExecuteOpNeverAliasesSymTarget(void) {
    /* SYM_INT32 target under ARITH_FLOAT32: epilogue must width-restore via
     * the conversionMatrix — aliasing would skip it. Assert staging happens. */
    tensor_t *in = buildFloat(4, (float[]){1, 2, 3, 4});
    tensor_t *out = buildSym(4, (int32_t[]){0, 0, 0, 0}, 1.0f);
    executeOp(
        &(opSpec_t){.kernel = probeAddOneKernel,
                    .inputs = (tensor_t *[]){in},
                    .nInputs = 1,
                    .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                    .mode = OUT_WRITE,
                    .auxOut = NULL,
                    .writesInPlaceSafe = true},
        out);
    const void *rawPtr = g_probeRawOutData;
    freeTensor(in);
    TEST_ASSERT_TRUE_MESSAGE(rawPtr != out->data, "non-FLOAT32 target must never alias");
    freeTensor(out);
}

/* ---- #282: OUT_WRITE epilogue rounding is operation-owned ------------- */

/* The OUT_WRITE requant into a quantized target must round by the OP's
 * arithmetic.roundingMode, NOT the target's storage qConfig (which stays
 * authoritative for bare conversions, ACC grids and serialization — and must
 * be RESTORED after the call). Fixture puts element 0 exactly on a HALF_AWAY
 * tie: floats {352.5, -2047.0} -> absMax 2047 -> scale = 2047/2047 = 1.0f
 * EXACTLY, so the requant ratios are 352.5 (tie) and -2047.0 (integral).
 * The epilogue draws from the op key's OUT(0) stream, fold(key, OUT = 10, 0),
 * key = rngKeyFromSeed(31):
 *   uv run python -c "from examples._shared.keyed_rng import (fold, key_from_seed as kfs,
 *   open_stream as o, next_unit_float as u); s = o(fold(kfs(31), 10, 0)); s, j0 = u(s); print(
 *   j0, u(s)[1])"
 * prints j0 = 0.0749632..., j1 = 0.2810223... :
 *   SR_HALF_AWAY: round(352.5 + j0 - 0.5) = round(352.07...) = 352
 *                 round(-2047.0 + j1 - 0.5) = round(-2047.21...) = -2047
 *   HALF_AWAY:    round(352.5) = 353 (what the target's config would give —
 *                 anti-vacuity: the two modes provably diverge on element 0) */
void testOutWriteEpilogueRoundsByArithmeticNotTargetQConfig(void) {
    tensor_t *in = buildFloat(2, (float[]){352.5f, -2047.0f});
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f); /* storage: HALF_AWAY */
    const rngKey_t key = rngKeyFromSeed(SR_GOLD_KEY_SEED);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY},
            .key = &key,
            .mode = OUT_WRITE,
        },
        out);

    int32_t got0 = ((int32_t *)out->data)[0];
    int32_t got1 = ((int32_t *)out->data)[1];
    float gotScale = ((symInt32QConfig_t *)out->quantization->qConfig)->scale;
    roundingMode_t storageModeAfter =
        ((symInt32QConfig_t *)out->quantization->qConfig)->roundingMode;
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT_MESSAGE(352, got0,
                                  "#282: OUT_WRITE must round by the op's SR mode, "
                                  "not the target's HALF_AWAY storage config");
    TEST_ASSERT_EQUAL_INT(-2047, got1);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, gotScale);
    TEST_ASSERT_EQUAL_INT_MESSAGE(HALF_AWAY, storageModeAfter,
                                  "#282: the target's storage roundingMode must be restored "
                                  "(it is serialized into checkpoints)");
}

/* Inverse direction: a deterministic op must stay deterministic even when the
 * target's storage config says SR — the storage mode never leaks into the op
 * (the #279 both-directions pin, now at the funnel level). Same fixture; with
 * the target's SR config in charge (the OLD semantics) element 0 would round
 * stochastically, and without a key it would stop with the named error: the
 * key-less op reaching 353 is the proof that it never draws. */
void testOutWriteDetArithmeticOverridesSrTargetQConfig(void) {
    tensor_t *in = buildFloat(2, (float[]){352.5f, -2047.0f});
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    ((symInt32QConfig_t *)out->quantization->qConfig)->roundingMode = SR_HALF_AWAY;

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
        },
        out);

    int32_t got0 = ((int32_t *)out->data)[0];
    int32_t got1 = ((int32_t *)out->data)[1];
    roundingMode_t storageModeAfter =
        ((symInt32QConfig_t *)out->quantization->qConfig)->roundingMode;
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT_MESSAGE(353, got0,
                                  "#282: a HALF_AWAY op must not pick up the "
                                  "target's SR storage config");
    TEST_ASSERT_EQUAL_INT(-2047, got1);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, storageModeAfter); /* restored */
}

/* Same contract through writeOut's OTHER branch, the SYM->SYM conversionMatrix
 * diagonal (requantSymInt32Tensor): mantissas {705, 4094} @ 0.5 dequantize to
 * {352.5, 2047.0} -> absMax 2047 -> scale 1.0f exactly, requant ratio
 * 705*(0.5/1.0) = 352.5 lands on the tie for element 0. The same OUT(0)
 * stream as testOutWriteEpilogueRoundsByArithmeticNotTargetQConfig (j0 =
 * 0.0749632...): SR 352 vs the target config's HALF_AWAY 353. */
void testOutWriteSymDiagonalRoundsByArithmetic(void) {
    tensor_t *in = buildSym(2, (int32_t[]){705, 4094}, 0.5f);
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f); /* storage: HALF_AWAY */
    const rngKey_t key = rngKeyFromSeed(SR_GOLD_KEY_SEED);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY},
            .key = &key,
            .mode = OUT_WRITE,
        },
        out);

    int32_t got0 = ((int32_t *)out->data)[0];
    int32_t got1 = ((int32_t *)out->data)[1];
    float gotScale = ((symInt32QConfig_t *)out->quantization->qConfig)->scale;
    roundingMode_t storageModeAfter =
        ((symInt32QConfig_t *)out->quantization->qConfig)->roundingMode;
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT_MESSAGE(352, got0,
                                  "#282: the SYM->SYM diagonal requant must round "
                                  "by the op's SR mode too");
    TEST_ASSERT_EQUAL_INT(2047, got1);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, gotScale);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, storageModeAfter); /* restored */
}

/* BFP epic PR1 (Task 8): the #282 swap must also reach a BFP target's
 * storage slot, or the optimizer's writeBackRounding (and any other
 * OUT_WRITE op rounding) silently drops onto the target's OWN storage
 * roundingMode instead. Fixture: intermediate floats {3.5, 7.0} into a
 * per-tensor BFP target (m=4, e=8). Group absMax = 7.0 (element 1) -> qMax=7
 * (m=4) -> derived E=0, scale=1.0f EXACTLY, so element 0's quotient 3.5/1.0
 * is a TRUE HALF_AWAY tie: the op's HALF_AWAY rounding must give mantissa 4
 * deterministically. The target's OWN storage config is SR_HALF_AWAY: had it
 * leaked through (storageRoundingSlot returning NULL for BFP, op rounding
 * ignored), this key-less op would stop with the named SR error instead. */
void testOutWriteBfpTargetUsesOpRoundingAndRestoresSlot(void) {
    tensor_t *in = buildFloat(2, (float[]){3.5f, 7.0f});

    uint8_t exponents[1] = {9}; /* sentinel, must be overwritten */
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = SR_HALF_AWAY, /* storage mode */
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t outQ;
    initBfpQuantization(&outQC, &outQ);
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    uint8_t outData[calcNumberOfBytesForData(&outQ, 2)];
    tensor_t out;
    setTensorValues(&out, outData, &shape, &outQ, NULL);

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
        },
        &out);

    int32_t mant[2];
    unpackSignExtend(out.data, 4, 0, mant, 2);
    roundingMode_t storageModeAfter = outQC.roundingMode;
    uint8_t exponentAfter = outQC.exponents[0];
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT_MESSAGE(4, mant[0],
                                  "BFP OUT_WRITE must round by the op's HALF_AWAY mode, "
                                  "not the target's SR_HALF_AWAY storage config");
    TEST_ASSERT_EQUAL_INT(7, mant[1]);
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(127, exponentAfter,
                                    "exponent derived fresh from absMax 7 -> qMax 7 -> E=0");
    TEST_ASSERT_EQUAL_INT_MESSAGE(SR_HALF_AWAY, storageModeAfter,
                                  "#282: the target's storage roundingMode must be restored "
                                  "(it is serialized into checkpoints)");
}

/* Counter-pin: executeConvert is a BARE storage-to-storage conversion — a
 * conversion node's rounding IS a storage encode, so the TARGET's qConfig
 * roundingMode applies there (unlike the OUT_WRITE epilogue above). Same
 * fixture, SR on the storage config. A bare conversion draws from the stream
 * it is handed, here rngOpen(rngKeyFromSeed(31)):
 *   uv run python -c "from examples._shared.keyed_rng import (key_from_seed as kfs,
 *   open_stream as o, next_unit_float as u); s = o(kfs(31)); s, j0 = u(s); print(j0, u(s)[1])"
 * prints j0 = 0.1786224..., j1 = 0.0632895...: round(352.5 + j0 - 0.5) = 352
 * (HALF_AWAY: 353), round(-2047.0 + j1 - 0.5) = -2047. */
void testExecuteConvertKeepsTargetStorageRounding(void) {
    tensor_t *in = buildFloat(2, (float[]){352.5f, -2047.0f});
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    ((symInt32QConfig_t *)out->quantization->qConfig)->roundingMode = SR_HALF_AWAY;
    rngStream_t rs = rngOpen(rngKeyFromSeed(SR_GOLD_KEY_SEED));

    executeConvert(in, out, &rs);

    int32_t got0 = ((int32_t *)out->data)[0];
    int32_t got1 = ((int32_t *)out->data)[1];
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT_MESSAGE(352, got0,
                                  "#282: bare conversion keeps storage-owned "
                                  "rounding (the target's SR mode applies)");
    TEST_ASSERT_EQUAL_INT(-2047, got1);
}

/* executeConvert SYM->SYM must requant through the conversionMatrix diagonal
 * (matches writeOut's OUT_WRITE trap) — raw accumulator-range mantissas come
 * out width-restored, bit-identical to conversionMatrix[SYM_INT32][SYM_INT32],
 * and the source is left untouched. */
void testExecuteConvertSymToSymRequantsThroughDiagonal(void) {
    tensor_t *in = buildSym(4, (int32_t[]){100000, -250000, 4095, 0}, 0.5f);
    tensor_t *out = buildSym(4, (int32_t[]){0, 0, 0, 0}, 1.0f);
    tensor_t *gold = buildSym(4, (int32_t[]){0, 0, 0, 0}, 1.0f);

    conversionMatrix[SYM_INT32][SYM_INT32](in, gold, NULL);
    executeConvert(in, out, NULL);

    int32_t got[4];
    int32_t want[4];
    for (size_t i = 0; i < 4; i++) {
        got[i] = ((int32_t *)out->data)[i];
        want[i] = ((int32_t *)gold->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)out->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)gold->quantization->qConfig)->scale;
    int32_t srcM0 = ((int32_t *)in->data)[0];
    freeTensor(gold);
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 4);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
    TEST_ASSERT_EQUAL_INT32(100000, srcM0); /* source untouched */
}

/* executeConvert FLOAT32->SYM must match convertTensor exactly (the funnel's
 * storage-to-storage conversion is not a different code path from the
 * existing quantizer). */
void testExecuteConvertFloatToSymMatchesConvertTensor(void) {
    tensor_t *in = buildFloat(3, (float[]){1.5f, -3.25f, 0.75f});
    tensor_t *out = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    tensor_t *gold = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);

    convertTensor(in, gold, NULL);
    executeConvert(in, out, NULL);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)out->data)[i];
        want[i] = ((int32_t *)gold->data)[i];
    }
    freeTensor(gold);
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
}

/* executeConvert BFP->BFP must route through the conversionMatrix[BFP][BFP]
 * diagonal, exactly parallel to the SYM_INT32 test above. The fixture makes
 * the two candidate paths unambiguous: src m=8 per-tensor vs dst m=4 grouped
 * {2,4} -- convertTensor's same-type branch would DIE on the width guard
 * (8 != 4 bits), and a stale-exponent copy could only yield the source's
 * stored 126; the diagonal instead derives BOTH target exponents FRESH from
 * the source VALUES on the TARGET's geometry (testRequantBfpReblocksToTarget-
 * Geometry's hand-derived gold: {127, 129}, mantissas {6,1,-2,0,7,-2,1,4}).
 * Stack fixtures (Task 2-4 idiom), NOT because a heap tensor would leak --
 * freeQuantization's BFP arm (TensorApi.c) already frees exponents[] cleanly
 * through freeTensor/freeQuantization, same as SYM. The actual reason:
 * executeConvert operates on tensor_t values directly, no layer/wire
 * allocation is involved here at all, so the stack-fixture idiom is simply
 * the lightest-weight way to build the two operand tensors -- it carries no
 * BFP-specific limitation. */
void testExecuteConvertBfpToBfpRoutesDiagonal(void) {
    size_t n = 8;
    size_t dims[] = {8};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[8] = {12, 2, -4, 0, 56, -14, 6, 28};
    uint8_t inExponents[1] = {126}; /* E=-1, scale 0.5 -> values {6,1,-2,0,28,-7,3,14} */
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = SR_HALF_AWAY, /* never read: dequant is rounding-free */
                         .mantissaBits = 8,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, inData, 8, n);
    tensor_t src;
    setTensorValues(&src, inData, &shape, &inQ, NULL);

    uint8_t outExponents[2] = {9, 9}; /* sentinels != expected {127, 129} */
    bfpQConfig_t outQC = {.exponents = outExponents,
                          .numGroups = 2,
                          .groupSize = 4,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t outQ;
    initBfpQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t dst;
    setTensorValues(&dst, outData, &shape, &outQ, NULL);

    executeConvert(&src, &dst, NULL);

    TEST_ASSERT_EQUAL_UINT8_MESSAGE(127, outQC.exponents[0],
                                    "diagonal must derive the target exponent FRESH "
                                    "(a same-type copy would die or carry stored 126)");
    TEST_ASSERT_EQUAL_UINT8(129, outQC.exponents[1]);
    int32_t mant[8];
    unpackSignExtend(dst.data, 4, 0, mant, 8);
    int32_t expected[8] = {6, 1, -2, 0, 7, -2, 1, 4};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, mant, 8);
}

/* ---- Group-quant PR2 (Task 3): funnel grouped-operand gate ------------- */

/* Packed grouped SYM tensor: mantissas packed verbatim (byteConversion),
 * per-group scales written directly (Task 1's quantizationInitSymGrouped
 * allocates scales[numGroups], initialized to 1.f each — overwritten here). */
static tensor_t *buildPackedSymGrouped(size_t n, const int32_t *mantissas, uint8_t qBits,
                                       size_t numGroups, size_t groupSize, const float *scales) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t =
        initTensor(shape, quantizationInitSymGrouped(qBits, HALF_AWAY, numGroups, groupSize), NULL);
    byteConversion((uint8_t *)mantissas, 32, t->data, qBits, n);
    symQConfig_t *qc = t->quantization->qConfig;
    for (size_t g = 0; g < numGroups; g++) {
        qc->scales[g] = scales[g];
    }
    return t;
}

/* Default-deny death test: a grouped SYM input (numGroups=2) reaching an
 * ARITH_SYM_INT32 op WITHOUT a matching groupedSymOperandPos must fail-fast,
 * not silently misinterpret the operand via the SYM->SYM_INT32
 * conversionMatrix cell (which itself fail-fasts on grouped sources, Task 2)
 * or any other path. */
void testExecuteOpRejectsGroupedSymOperandByDefault(void) {
    ASSERT_EXITS_WITH_FAILURE({
        tensor_t *in = buildPackedSymGrouped(8, (int32_t[]){1, -2, 3, -4, 5, -6, 7, -8}, 6, 2, 4,
                                             (float[]){0.1f, 0.2f});
        tensor_t *out = buildSym(8, (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0}, 1.0f);
        quantization_t arith;
        symInt32QConfig_t arithQC;
        initSymInt32QConfig(HALF_AWAY, &arithQC);
        initSymInt32Quantization(&arithQC, &arith);

        executeOp(
            &(opSpec_t){
                .kernel = executeOpIdentityKernel,
                .inputs = (tensor_t *[]){in},
                .nInputs = 1,
                .arithmetic = arithmeticFromQuantization(&arith),
                .mode = OUT_WRITE,
                /* groupedSymOperandPos intentionally NOT set (zero-init = deny) */
            },
            out);
    });
}

/* A grouped SYM input at a position OTHER than the one
 * declared must still fail-fast — an opt-in for SOME position must not be
 * read as a blanket opt-in for grouped operands anywhere in the operand
 * list. groupedSymOperandPos=2 declares inputs[1] as the only allowed
 * grouped position; the (single) actual input here is inputs[0] (position
 * 1), so this must deny exactly like the zero-init case above. */
void testExecuteOpRejectsGroupedSymOperandAtWrongPosition(void) {
    ASSERT_EXITS_WITH_FAILURE({
        tensor_t *in = buildPackedSymGrouped(8, (int32_t[]){1, -2, 3, -4, 5, -6, 7, -8}, 6, 2, 4,
                                             (float[]){0.1f, 0.2f});
        tensor_t *out = buildSym(8, (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0}, 1.0f);
        quantization_t arith;
        symInt32QConfig_t arithQC;
        initSymInt32QConfig(HALF_AWAY, &arithQC);
        initSymInt32Quantization(&arithQC, &arith);

        executeOp(
            &(opSpec_t){
                .kernel = executeOpIdentityKernel,
                .inputs = (tensor_t *[]){in},
                .nInputs = 1,
                .arithmetic = arithmeticFromQuantization(&arith),
                .mode = OUT_WRITE,
                .groupedSymOperandPos = 2, /* declares inputs[1] (nonexistent here) —
                                            * inputs[0] being grouped must still deny */
            },
            out);
    });
}

static int32_t g_capturedGroupedMantissas[8];
static float g_capturedGroupedScale;
static uint8_t g_capturedGroupedQMaxBits;

static void captureGroupedOperandKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                                        tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)nOperands;
    (void)auxOut;
    (void)ctx;
    size_t n = calcNumberOfElementsByTensor(operands[0]);
    for (size_t i = 0; i < n; i++) {
        g_capturedGroupedMantissas[i] = ((int32_t *)operands[0]->data)[i];
    }
    symInt32QConfig_t *qc = operands[0]->quantization->qConfig;
    g_capturedGroupedScale = qc->scale;
    g_capturedGroupedQMaxBits = qc->qMaxBits;
    /* rawOut written deterministically (zero) -- this test only inspects the
     * PROLOGUE's scratch, not the epilogue. */
    size_t outN = calcNumberOfElementsByTensor(rawOut);
    for (size_t i = 0; i < outN; i++) {
        ((int32_t *)rawOut->data)[i] = 0;
    }
}

/* Allowed path: groupedSymOperandPos=1 (declares inputs[0]) unpacks the
 * grouped SYM operand into scratch — mantissas equal the packed codes
 * (sign-extended), scratch scale is poisoned to 1.0f (never a real scale),
 * qMaxBits carries the source's qBits. */
void testExecuteOpUnpacksGroupedSymWhenAllowed(void) {
    int32_t mantissas[] = {1, -2, 3, -4, 5, -6, 7, -8};
    tensor_t *in = buildPackedSymGrouped(8, mantissas, 6, 2, 4, (float[]){0.1f, 0.2f});
    tensor_t *out = buildSym(8, (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0}, 1.0f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(
        &(opSpec_t){
            .kernel = captureGroupedOperandKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .mode = OUT_WRITE,
            .groupedSymOperandPos = 1,
        },
        out);

    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT32_ARRAY(mantissas, g_capturedGroupedMantissas, 8);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, g_capturedGroupedScale);
    TEST_ASSERT_EQUAL_UINT8(6, g_capturedGroupedQMaxBits);
}

/* The grouped-operand deny gate must apply under
 * ARITH_FLOAT32 too, not just ARITH_SYM_INT32 — before this fix, a grouped
 * SYM operand reaching an ARITH_FLOAT32 op would silently succeed (the
 * FLOAT32 prologue's convertTensor call dispatches to the group-aware
 * convertSymTensorToFloat32Tensor cell regardless of any gate), letting
 * FLOAT32-math ops (e.g. an optimizer update or a FLOAT32-math backward dx)
 * quietly dequantize a grouped weight/param that PR2's contract says must be
 * denied outside the two declared gather-forward call sites. No position
 * declared here (zero-init deny), so this must fail-fast exactly like the
 * SYM_INT32 default-deny sibling above. */
void testExecuteOpRejectsGroupedSymOperandUnderFloat32ByDefault(void) {
    ASSERT_EXITS_WITH_FAILURE({
        tensor_t *in = buildPackedSymGrouped(8, (int32_t[]){1, -2, 3, -4, 5, -6, 7, -8}, 6, 2, 4,
                                             (float[]){0.1f, 0.2f});
        tensor_t *out = buildFloat(8, (float[]){0, 0, 0, 0, 0, 0, 0, 0});

        executeOp(
            &(opSpec_t){
                .kernel = executeOpIdentityKernel,
                .inputs = (tensor_t *[]){in},
                .nInputs = 1,
                .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                .mode = OUT_WRITE,
                /* groupedSymOperandPos intentionally NOT set (zero-init = deny) */
            },
            out);
    });
}

/* ---- Group-quant PR4 (Task 3): funnel grouped-ASYM operand gate --------- */

/* Packed grouped ASYM tensor: codes packed verbatim (byteConversion,
 * non-negative, no sign bit), per-group scales AND code-domain zeroPoints
 * written directly (Task 1's quantizationInitAsymGrouped allocates both
 * arrays). Mirrors buildPackedSymGrouped above. */
static tensor_t *buildPackedAsymGrouped(size_t n, const int32_t *codes, uint8_t qBits,
                                        size_t numGroups, size_t groupSize, const float *scales,
                                        const uint16_t *zeroPoints) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(
        shape, quantizationInitAsymGrouped(qBits, HALF_AWAY, numGroups, groupSize), NULL);
    byteConversion((uint8_t *)codes, 32, t->data, qBits, n);
    asymQConfig_t *qc = t->quantization->qConfig;
    for (size_t g = 0; g < numGroups; g++) {
        qc->scales[g] = scales[g];
        qc->zeroPoints[g] = zeroPoints[g];
    }
    return t;
}

/* Shared grouped-ASYM gate fixture: 2 groups of 4, qBits=6 (codes in [0, 63]),
 * PAIRWISE-DISTINCT zeroPoints (a zp[g] -> zp[0] shift bug must change the
 * mantissas) chosen so the shifted mantissas carry BOTH signs in BOTH groups. */
static const int32_t kAsymGateCodes[8] = {21, 5, 60, 12, 63, 0, 50, 30};
static const uint16_t kAsymGateZps[2] = {20, 41};
static const float kAsymGateScales[2] = {0.1f, 0.2f};
/* mantissas = code - zp[i/4]: {1, -15, 40, -8} (zp 20) ++ {22, -41, 9, -11}
 * (zp 41). */
static const int32_t kAsymGateMantissas[8] = {1, -15, 40, -8, 22, -41, 9, -11};

/* Default-deny death test, ASYM twin of
 * testExecuteOpRejectsGroupedSymOperandByDefault: a grouped ASYM input
 * reaching an ARITH_SYM_INT32 op WITHOUT a matching groupedSymOperandPos must
 * fail-fast AT THE GATE. RED evidence (pre-Task-3): this test already "dies",
 * but in the WRONG place -- Task 1's requirePerTensorAsym inside the
 * ASYM->SYM_INT32 conversionMatrix cell ("grouped ASYM ... has no compute
 * image in this cell"), not the funnel's carrier-contract deny ("reached an
 * op without a matching groupedSymOperandPos declaration"); the exit-code
 * harness cannot tell them apart, so the message difference is pinned in the
 * task report's RED transcript instead. */
void testExecuteOpRejectsGroupedAsymOperandByDefault(void) {
    ASSERT_EXITS_WITH_FAILURE({
        tensor_t *in =
            buildPackedAsymGrouped(8, kAsymGateCodes, 6, 2, 4, kAsymGateScales, kAsymGateZps);
        tensor_t *out = buildSym(8, (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0}, 1.0f);
        quantization_t arith;
        symInt32QConfig_t arithQC;
        initSymInt32QConfig(HALF_AWAY, &arithQC);
        initSymInt32Quantization(&arithQC, &arith);

        executeOp(
            &(opSpec_t){
                .kernel = executeOpIdentityKernel,
                .inputs = (tensor_t *[]){in},
                .nInputs = 1,
                .arithmetic = arithmeticFromQuantization(&arith),
                .mode = OUT_WRITE,
                /* groupedSymOperandPos intentionally NOT set (zero-init = deny) */
            },
            out);
    });
}

/* Wrong-position deny, ASYM twin of
 * testExecuteOpRejectsGroupedSymOperandAtWrongPosition: declaring SOME
 * position must not read as a blanket ASYM opt-in either. */
void testExecuteOpRejectsGroupedAsymOperandAtWrongPosition(void) {
    ASSERT_EXITS_WITH_FAILURE({
        tensor_t *in =
            buildPackedAsymGrouped(8, kAsymGateCodes, 6, 2, 4, kAsymGateScales, kAsymGateZps);
        tensor_t *out = buildSym(8, (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0}, 1.0f);
        quantization_t arith;
        symInt32QConfig_t arithQC;
        initSymInt32QConfig(HALF_AWAY, &arithQC);
        initSymInt32Quantization(&arithQC, &arith);

        executeOp(
            &(opSpec_t){
                .kernel = executeOpIdentityKernel,
                .inputs = (tensor_t *[]){in},
                .nInputs = 1,
                .arithmetic = arithmeticFromQuantization(&arith),
                .mode = OUT_WRITE,
                .groupedSymOperandPos = 2, /* declares inputs[1] (nonexistent here) --
                                            * inputs[0] being grouped must still deny */
            },
            out);
    });
}

/* THE Task-3 vulnerability pin (RED = this test FAILS pre-gate): under
 * ARITH_FLOAT32 a grouped ASYM operand at a NON-declared position passes
 * SILENTLY today -- Task 2 made the ASYM->FLOAT32 conversionMatrix cell
 * group-aware, and the funnel's grouped-operand gate keys on type == SYM
 * only, so the FLOAT32 prologue dequantizes a grouped ASYM tensor at ANY
 * position with no carrier-contract check at all (unlike grouped SYM, which
 * the PR2 gate denies). Same deny semantics as the SYM sibling
 * testExecuteOpRejectsGroupedSymOperandUnderFloat32ByDefault. */
void testExecuteOpRejectsGroupedAsymOperandUnderFloat32ByDefault(void) {
    ASSERT_EXITS_WITH_FAILURE({
        tensor_t *in =
            buildPackedAsymGrouped(8, kAsymGateCodes, 6, 2, 4, kAsymGateScales, kAsymGateZps);
        tensor_t *out = buildFloat(8, (float[]){0, 0, 0, 0, 0, 0, 0, 0});

        executeOp(
            &(opSpec_t){
                .kernel = executeOpIdentityKernel,
                .inputs = (tensor_t *[]){in},
                .nInputs = 1,
                .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                .mode = OUT_WRITE,
                /* groupedSymOperandPos intentionally NOT set (zero-init = deny) */
            },
            out);
    });
}

/* Allowed path, ASYM twin of testExecuteOpUnpacksGroupedSymWhenAllowed: the
 * declared grouped ASYM operand is zero-extended and SHIFTED into the
 * signed-mantissa image (mantissa[i] == code[i] - zp[i / groupSize] -- the
 * pairwise-distinct fixture zps make a zp[0]-everywhere shift visibly wrong
 * in group 1), scratch scale is poisoned to 1.0f, qMaxBits carries the
 * source's qBits (the same contract as the SYM arm). */
void testExecuteOpUnpacksGroupedAsymWhenAllowed(void) {
    tensor_t *in =
        buildPackedAsymGrouped(8, kAsymGateCodes, 6, 2, 4, kAsymGateScales, kAsymGateZps);
    tensor_t *out = buildSym(8, (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0}, 1.0f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(
        &(opSpec_t){
            .kernel = captureGroupedOperandKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = arithmeticFromQuantization(&arith),
            .mode = OUT_WRITE,
            .groupedSymOperandPos = 1,
        },
        out);

    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kAsymGateMantissas, g_capturedGroupedMantissas, 8);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, g_capturedGroupedScale);
    TEST_ASSERT_EQUAL_UINT8(6, g_capturedGroupedQMaxBits);
}

/* ---- BFP epic PR2 (Task 6): ARITH_BFP prologue/raw arms + bfpStage ------ */

/* Shared capture globals for the ARITH_BFP probe kernels (assert-after-free
 * discipline: heap targets are freed before Unity's longjmp-prone asserts). */
static const uint8_t *g_bfpCapExponentsPtr;
static uint8_t g_bfpCapExponents[2];
static int32_t g_bfpCapCodes[8];
static roundingMode_t g_bfpCapRounding;
static int g_bfpCapQType;
static size_t g_bfpCapNumGroups;
static size_t g_bfpCapGroupSize;
static uint8_t g_bfpCapMantissaBits;
static int g_bfpCapRawQType;

static void captureBfpOperandKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                                    tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)nOperands;
    (void)auxOut;
    (void)ctx;
    g_bfpCapQType = (int)operands[0]->quantization->type;
    bfpQConfig_t *qc = operands[0]->quantization->qConfig;
    g_bfpCapExponentsPtr = qc->exponents;
    for (size_t g = 0; g < qc->numGroups && g < 2; g++) {
        g_bfpCapExponents[g] = qc->exponents[g];
    }
    g_bfpCapRounding = qc->roundingMode;
    g_bfpCapNumGroups = qc->numGroups;
    g_bfpCapGroupSize = qc->groupSize;
    g_bfpCapMantissaBits = qc->mantissaBits;
    size_t n = calcNumberOfElementsByTensor(operands[0]);
    for (size_t i = 0; i < n && i < 8; i++) {
        g_bfpCapCodes[i] = ((int32_t *)operands[0]->data)[i];
    }
    g_bfpCapRawQType = (int)rawOut->quantization->type;
    /* raw is FLOAT32 under ARITH_BFP (BFP D7); write it fully so the epilogue
     * never reads uninitialized stack. */
    size_t outN = calcNumberOfElementsByTensor(rawOut);
    for (size_t i = 0; i < outN; i++) {
        ((float *)rawOut->data)[i] = 0.f;
    }
}

/* (a) A BFP-STORED operand reaches the kernel in unpacked-BFP scratch form:
 * ->data is int32 SIGN-EXTENDED mantissas (negative codes pin sign-extend
 * over zero-extend), ->quantization is BFP whose exponents array is BORROWED
 * from the source config (pointer equality — zero-copy, prologue never
 * copies or rewrites the exponents). Grouped {2,4} fixture WITHOUT any
 * groupedSymOperandPos declaration: BFP blocking is per-operand-legal under
 * ARITH_BFP and must bypass the grouped-SYM/ASYM carrier gate entirely. */
void testExecuteOpBfpStoredOperandUnpacksAndBorrows(void) {
    size_t n = 8;
    size_t dims[] = {8};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[8] = {6, 1, -2, 0, 7, -2, 1, 4}; /* the :5363 requant fixture */
    uint8_t srcExponents[2] = {127, 129};
    bfpQConfig_t srcQC = {.exponents = srcExponents,
                          .numGroups = 2,
                          .groupSize = 4,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t srcQ;
    initBfpQuantization(&srcQC, &srcQ);
    uint8_t srcData[calcNumberOfBytesForData(&srcQ, n)];
    byteConversion((uint8_t *)goldMant, 32, srcData, 4, n);
    tensor_t src;
    setTensorValues(&src, srcData, &shape, &srcQ, NULL);

    tensor_t *out = buildFloat(8, (float[]){0, 0, 0, 0, 0, 0, 0, 0});

    executeOp(
        &(opSpec_t){
            .kernel = captureBfpOperandKernel,
            .inputs = (tensor_t *[]){&src},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
            /* no groupedSymOperandPos, no bfpStage — grouped BFP needs neither */
        },
        out);

    freeTensor(out);
    TEST_ASSERT_EQUAL_INT(BFP, g_bfpCapQType);
    TEST_ASSERT_EQUAL_PTR(srcExponents, g_bfpCapExponentsPtr); /* borrow, not copy */
    TEST_ASSERT_EQUAL_UINT8(127, g_bfpCapExponents[0]);
    TEST_ASSERT_EQUAL_UINT8(129, g_bfpCapExponents[1]);
    TEST_ASSERT_EQUAL_INT32_ARRAY(goldMant, g_bfpCapCodes, 8);
    TEST_ASSERT_EQUAL_size_t(2, g_bfpCapNumGroups);
    TEST_ASSERT_EQUAL_size_t(4, g_bfpCapGroupSize);
    TEST_ASSERT_EQUAL_UINT8(4, g_bfpCapMantissaBits);
}

/* (b) A FLOAT32-STORED operand is staged fresh through the bfpStage geometry
 * template: Task 2's {100,-50,25,0}/m=4 gold — stored exponent 131 (E=+4,
 * scale 16), codes {6,-3,2,0}. The template contributes GEOMETRY+WIDTHS
 * only: its exponents pointer is ignored (funnel-owned scratch backing, so
 * the captured pointer differs and the template's sentinel survives) and its
 * roundingMode is ignored in favor of the OP's arithmetic.roundingMode
 * (template says SR, captured mode must be the op's HALF_AWAY — asserted on
 * the MODE, not the codes, because 100/16, -50/16, 25/16 are all fractional
 * and would make a codes-only assertion stochastic under the bug). */
void testExecuteOpFloat32OperandStagedToBfpCodes(void) {
    tensor_t *in = buildFloat(4, (float[]){100.f, -50.f, 25.f, 0.f});
    tensor_t *out = buildFloat(4, (float[]){0, 0, 0, 0});
    uint8_t templateSentinel[1] = {9};
    bfpQConfig_t stage = {.exponents = templateSentinel,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = SR_HALF_AWAY, /* must be ignored */
                          .mantissaBits = 4,
                          .exponentBits = 8};

    executeOp(
        &(opSpec_t){
            .kernel = captureBfpOperandKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
            .bfpStage = {&stage},
        },
        out);

    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT(BFP, g_bfpCapQType);
    TEST_ASSERT_EQUAL_UINT8(131, g_bfpCapExponents[0]);
    int32_t expected[4] = {6, -3, 2, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, g_bfpCapCodes, 4);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, (int)g_bfpCapRounding);    /* op's, not template's */
    TEST_ASSERT_TRUE(g_bfpCapExponentsPtr != templateSentinel); /* funnel-owned */
    TEST_ASSERT_EQUAL_UINT8(9, templateSentinel[0]);            /* never written */
    TEST_ASSERT_EQUAL_size_t(1, g_bfpCapNumGroups);
    TEST_ASSERT_EQUAL_size_t(0, g_bfpCapGroupSize);
    TEST_ASSERT_EQUAL_UINT8(4, g_bfpCapMantissaBits);
}

static const uint8_t *g_bfpCap2ExponentsPtr[2];
static uint8_t g_bfpCap2Exponents[2][2];
static int32_t g_bfpCap2Codes[2][8];
static size_t g_bfpCap2NumGroups[2];

static void captureTwoBfpOperandsKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                                        tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    (void)ctx;
    for (size_t k = 0; k < nOperands && k < 2; k++) {
        bfpQConfig_t *qc = operands[k]->quantization->qConfig;
        g_bfpCap2ExponentsPtr[k] = qc->exponents;
        g_bfpCap2NumGroups[k] = qc->numGroups;
        for (size_t g = 0; g < qc->numGroups && g < 2; g++) {
            g_bfpCap2Exponents[k][g] = qc->exponents[g];
        }
        size_t n = calcNumberOfElementsByTensor(operands[k]);
        for (size_t i = 0; i < n && i < 8; i++) {
            g_bfpCap2Codes[k][i] = ((int32_t *)operands[k]->data)[i];
        }
    }
    size_t outN = calcNumberOfElementsByTensor(rawOut);
    for (size_t i = 0; i < outN; i++) {
        ((float *)rawOut->data)[i] = 0.f;
    }
}

/* Grouped stage template under real staging: TWO FLOAT32-stored
 * operands in ONE call — operand 0 under a GROUPED {2,4} template, operand 1
 * under {1,0} — so the stage-exponent VLA sizing (sum of the templates'
 * numGroups) and the per-operand slice advance are both load-bearing.
 * Hand-derived golds (m=4 -> qMax 7, e=8 -> bias 127, HALF_AWAY):
 *   op0 group0 {6,1,-2,0}: absMax 6, 6/7 in (1/2, 1] -> E=0 -> stored 127,
 *     scale 1, codes {6,1,-2,0} exact;
 *   op0 group1 {28,-7,3,14}: absMax 28, 28/7 = 4 = 2^2 (snap-up boundary)
 *     -> E=+2 -> stored 129, scale 4, codes {7, -1.75->-2, 0.75->1, 3.5->4};
 *   op1 {100,-50,25,0}: absMax 100, 100/7 ~ 14.29 in (2^3, 2^4] -> E=+4 ->
 *     stored 131, scale 16, codes {6.25->6, -3.125->-3, 1.5625->2, 0}.
 * All three stored exponents are pairwise distinct, so any slice mix-up is
 * visible in the values; the distinct-address assert additionally kills a
 * dropped stageOffset advance (both operands would alias the same funnel
 * slice AND op1's staging would overwrite op0's exponent). An undersized
 * VLA (totalStageExponents miscounted, e.g. counting a grouped template as
 * 1) is a stack OOB WRITE past the VLA end — locally that scribble can go
 * unnoticed by these asserts; the CI ASan job (unit_test_asan) is the
 * intended catcher for that mutation class. */
void testExecuteOpStagesGroupedAndPerTensorOperandsDistinctSlices(void) {
    tensor_t *a = buildFloat(8, (float[]){6.f, 1.f, -2.f, 0.f, 28.f, -7.f, 3.f, 14.f});
    tensor_t *b = buildFloat(4, (float[]){100.f, -50.f, 25.f, 0.f});
    tensor_t *out = buildFloat(8, (float[]){0, 0, 0, 0, 0, 0, 0, 0});
    uint8_t sentinelsA[2] = {9, 9};
    bfpQConfig_t stageA = {.exponents = sentinelsA,
                           .numGroups = 2,
                           .groupSize = 4,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 4,
                           .exponentBits = 8};
    uint8_t sentinelB[1] = {9};
    bfpQConfig_t stageB = {.exponents = sentinelB,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 4,
                           .exponentBits = 8};

    executeOp(
        &(opSpec_t){
            .kernel = captureTwoBfpOperandsKernel,
            .inputs = (tensor_t *[]){a, b},
            .nInputs = 2,
            .arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
            .bfpStage = {&stageA, &stageB},
        },
        out);

    freeTensor(out);
    freeTensor(b);
    freeTensor(a);
    TEST_ASSERT_EQUAL_size_t(2, g_bfpCap2NumGroups[0]);
    TEST_ASSERT_EQUAL_size_t(1, g_bfpCap2NumGroups[1]);
    TEST_ASSERT_EQUAL_UINT8(127, g_bfpCap2Exponents[0][0]);
    TEST_ASSERT_EQUAL_UINT8(129, g_bfpCap2Exponents[0][1]);
    int32_t expectedA[8] = {6, 1, -2, 0, 7, -2, 1, 4};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedA, g_bfpCap2Codes[0], 8);
    TEST_ASSERT_EQUAL_UINT8(131, g_bfpCap2Exponents[1][0]);
    int32_t expectedB[4] = {6, -3, 2, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedB, g_bfpCap2Codes[1], 4);
    /* distinct funnel slices — a dropped stageOffset advance aliases these */
    TEST_ASSERT_TRUE(g_bfpCap2ExponentsPtr[0] != g_bfpCap2ExponentsPtr[1]);
}

/* (c) death: FLOAT32-stored operand under ARITH_BFP with no stage template. */
void testExecuteOpBfpMissingStageDies(void) {
    ASSERT_EXITS_WITH_FAILURE({
        tensor_t *in = buildFloat(4, (float[]){1.f, 2.f, 3.f, 4.f});
        tensor_t *out = buildFloat(4, (float[]){0, 0, 0, 0});
        executeOp(
            &(opSpec_t){
                .kernel = captureBfpOperandKernel,
                .inputs = (tensor_t *[]){in},
                .nInputs = 1,
                .arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY},
                .mode = OUT_WRITE,
                /* .bfpStage intentionally not set (zero-init = NULL) */
            },
            out);
    });
}

/* (d) death: SYM_INT32-stored operand under ARITH_BFP (v1 accepts FLOAT32 or
 * BFP storage only; anything else must convert explicitly first). */
void testExecuteOpBfpRejectsSymStoredOperand(void) {
    ASSERT_EXITS_WITH_FAILURE({
        tensor_t *in = buildSym(4, (int32_t[]){10, -20, 30, -40}, 0.1f);
        tensor_t *out = buildFloat(4, (float[]){0, 0, 0, 0});
        executeOp(
            &(opSpec_t){
                .kernel = captureBfpOperandKernel,
                .inputs = (tensor_t *[]){in},
                .nInputs = 1,
                .arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY},
                .mode = OUT_WRITE,
            },
            out);
    });
}

/* Body extracted to a function: the death-test macro takes ONE argument, so
 * statement-level braced initializers (comma at brace depth, paren depth 0)
 * would split the macro argument list. */
static void bfpStageGeometryMismatchBody(void) {
    tensor_t *in = buildFloat(4, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *out = buildFloat(4, (float[]){0, 0, 0, 0});
    uint8_t sentinels[2] = {9, 9};
    bfpQConfig_t stage = {.exponents = sentinels,
                          .numGroups = 2,
                          .groupSize = 3, /* 2*3 != 4 */
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    executeOp(
        &(opSpec_t){
            .kernel = captureBfpOperandKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
            .bfpStage = {&stage},
        },
        out);
}

/* (e) death: grouped stage template whose numGroups*groupSize != n. */
void testExecuteOpBfpStageGeometryMismatchDies(void) {
    ASSERT_EXITS_WITH_FAILURE(bfpStageGeometryMismatchBody());
}

/* (f) SR staging draws from the op key's STAGE(0) stream, fold(key, STAGE = 8,
 * 0): the quotients v / 2 at the derived scale 2 (stored exponent 128, the
 * derivation itself is rounding-free) round to these codes for key
 * rngKeyFromSeed(31), by the C float arithmetic (round half away of
 * (q + j) - 0.5 in float, then the [-8, 7] clamp):
 *   uv run python -c "import struct, math, itertools; from examples._shared.keyed_rng import (
 *   fold, key_from_seed as kfs, open_stream as o, next_unit_float as u); f = lambda v: (
 *   struct.unpack('f', struct.pack('f', v))[0]); r = lambda x: int(math.copysign(
 *   math.floor(abs(x) + 0.5), x)); st = list(itertools.accumulate(range(8),
 *   lambda s, _: u(s)[0], initial=o(fold(kfs(31), 8, 0))))[1:]; print([max(-8, min(7,
 *   r(f(f(q + (s >> 8) / 2**24) - 0.5)))) for q, s in zip((7, 1.75, -0.875, 1.3125, 3.5,
 *   -5.25, 0.4375, 2.625), st)])"
 * prints [7, 2, -1, 1, 3, -6, 0, 3]; HALF_AWAY gives 7 2 -1 1 4 -5 0 3. The same key
 * twice gives the same codes, a different key (32) different ones. Kills a
 * hardcoded-HALF_AWAY staging mutation. */
void testExecuteOpBfpStagingSrDrawsFromTheStageStream(void) {
    float vals[8] = {14.f, 3.5f, -1.75f, 2.625f, 7.f, -10.5f, 0.875f, 5.25f};
    tensor_t *in = buildFloat(8, vals);
    tensor_t *out = buildFloat(8, (float[]){0, 0, 0, 0, 0, 0, 0, 0});
    uint8_t templateSentinel[1] = {0};
    bfpQConfig_t stage = {.exponents = templateSentinel,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY, /* ignored either way */
                          .mantissaBits = 4,
                          .exponentBits = 8};
    const rngKey_t key = rngKeyFromSeed(SR_GOLD_KEY_SEED);
    const rngKey_t otherKey = rngKeyFromSeed(SR_GOLD_KEY_SEED + 1u);
    opSpec_t srSpec = {
        .kernel = captureBfpOperandKernel,
        .inputs = (tensor_t *[]){in},
        .nInputs = 1,
        .arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = SR_HALF_AWAY},
        .key = &key,
        .mode = OUT_WRITE,
        .bfpStage = {&stage},
    };

    executeOp(&srSpec, out);
    int32_t sr1[8];
    memcpy(sr1, g_bfpCapCodes, sizeof(sr1));
    uint8_t srExponent = g_bfpCapExponents[0];

    executeOp(&srSpec, out);
    int32_t sr2[8];
    memcpy(sr2, g_bfpCapCodes, sizeof(sr2));

    opSpec_t otherSpec = srSpec;
    otherSpec.key = &otherKey;
    executeOp(&otherSpec, out);
    int32_t other[8];
    memcpy(other, g_bfpCapCodes, sizeof(other));

    opSpec_t refSpec = srSpec;
    refSpec.arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    executeOp(&refSpec, out);
    int32_t ref[8];
    memcpy(ref, g_bfpCapCodes, sizeof(ref));

    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_UINT8(128, srExponent);
    TEST_ASSERT_EQUAL_INT32_ARRAY(((int32_t[]){7, 2, -1, 1, 3, -6, 0, 3}), sr1, 8);
    TEST_ASSERT_EQUAL_INT32_ARRAY(sr1, sr2, 8);
    TEST_ASSERT_FALSE(memcmp(sr1, other, sizeof sr1) == 0);
    TEST_ASSERT_EQUAL_INT32_ARRAY(((int32_t[]){7, 2, -1, 1, 4, -5, 0, 3}), ref, 8);
}

static void writeKnownFloatsKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                                   tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)operands;
    (void)nOperands;
    (void)auxOut;
    (void)ctx;
    g_bfpCapRawQType = (int)rawOut->quantization->type;
    float vals[4] = {100.f, -50.f, 25.f, 0.f};
    for (size_t i = 0; i < 4; i++) {
        ((float *)rawOut->data)[i] = vals[i];
    }
}

/* (g) The raw intermediate under ARITH_BFP is FLOAT32 (BFP D7) and the OUT_WRITE
 * epilogue packs a BFP target through the [FLOAT32][BFP] cell: the probe
 * writes Task 2's {100,-50,25,0} into rawOut, the m=4 target must come out
 * as stored exponent 131 + packed codes {6,-3,2,0} (composition test — the
 * cell itself is PR1-covered). */
void testExecuteOpBfpRawIsFloat32AndPacksTarget(void) {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t srcMant[4] = {1, 1, 1, 1}; /* content irrelevant: probe ignores operands */
    uint8_t srcExponents[1] = {127};
    bfpQConfig_t srcQC = {.exponents = srcExponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t srcQ;
    initBfpQuantization(&srcQC, &srcQ);
    uint8_t srcData[calcNumberOfBytesForData(&srcQ, n)];
    byteConversion((uint8_t *)srcMant, 32, srcData, 4, n);
    tensor_t src;
    setTensorValues(&src, srcData, &shape, &srcQ, NULL);

    uint8_t outExponents[1] = {9}; /* sentinel != expected 131 */
    bfpQConfig_t outQC = {.exponents = outExponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t outQ;
    initBfpQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t dst;
    setTensorValues(&dst, outData, &shape, &outQ, NULL);

    executeOp(
        &(opSpec_t){
            .kernel = writeKnownFloatsKernel,
            .inputs = (tensor_t *[]){&src},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
        },
        &dst);

    TEST_ASSERT_EQUAL_INT(FLOAT32, g_bfpCapRawQType);
    TEST_ASSERT_EQUAL_UINT8(131, outExponents[0]);
    int32_t mant[4];
    unpackSignExtend(dst.data, 4, 0, mant, 4);
    int32_t expected[4] = {6, -3, 2, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, mant, 4);
}

/* Body extracted for the same one-macro-argument reason as
 * bfpStageGeometryMismatchBody above. */
static void symArithmeticBfpStoredOperandBody(void) {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    int32_t srcMant[4] = {6, -3, 2, 0};
    uint8_t srcExponents[1] = {131};
    bfpQConfig_t srcQC = {.exponents = srcExponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t srcQ;
    initBfpQuantization(&srcQC, &srcQ);
    uint8_t srcData[calcNumberOfBytesForData(&srcQ, n)];
    byteConversion((uint8_t *)srcMant, 32, srcData, 4, n);
    tensor_t src;
    setTensorValues(&src, srcData, &shape, &srcQ, NULL);

    tensor_t *out = buildSym(4, (int32_t[]){0, 0, 0, 0}, 1.0f);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){&src},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
        },
        out);
}

/* (h) PR2-Decision 11, deny direction: a BFP-STORED operand under ARITH_SYM_INT32
 * must fail-fast — the [BFP][SYM_INT32] cell would silently collapse the
 * operand's group structure to a single scalar grid. */
void testExecuteOpSymArithmeticRejectsBfpStoredOperand(void) {
    ASSERT_EXITS_WITH_FAILURE(symArithmeticBfpStoredOperandBody());
}

/* (h) PR2-Decision 11, keep direction: a GROUPED BFP-stored operand under
 * ARITH_FLOAT32 keeps working — the PR1 fake-quant path (float-bridge
 * staging via the group-aware BFP->FLOAT32 dequant cell) is what every BFP
 * layer runs on until the Task 9 derivation flip; regression guard. */
void testExecuteOpFloatArithmeticStillDequantsGroupedBfp(void) {
    size_t n = 8;
    size_t dims[] = {8};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[8] = {6, 1, -2, 0, 7, -2, 1, 4};
    uint8_t srcExponents[2] = {127, 129}; /* group scales 1 and 4 */
    bfpQConfig_t srcQC = {.exponents = srcExponents,
                          .numGroups = 2,
                          .groupSize = 4,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t srcQ;
    initBfpQuantization(&srcQC, &srcQ);
    uint8_t srcData[calcNumberOfBytesForData(&srcQ, n)];
    byteConversion((uint8_t *)goldMant, 32, srcData, 4, n);
    tensor_t src;
    setTensorValues(&src, srcData, &shape, &srcQ, NULL);

    tensor_t *out = buildFloat(8, (float[]){0, 0, 0, 0, 0, 0, 0, 0});

    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){&src},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
            /* no groupedSymOperandPos: BFP never uses the SYM/ASYM carrier gate */
        },
        out);

    float got[8];
    memcpy(got, out->data, sizeof(got));
    freeTensor(out);
    float expected[8] = {6.f, 1.f, -2.f, 0.f, 28.f, -8.f, 4.f, 16.f};
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_EQUAL_FLOAT(expected[i], got[i]);
    }
}

/* ---- the funnel's keyed sub-streams ---- */

/* The expected stream states come from the Python mirror, not from the C key
 * chain, with the tag values spelled as numbers (STAGE 8, KERNEL 9, OUT 10,
 * ACC 11). From the repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (fold, key_from_seed,
 *   open_stream); k = key_from_seed(2024); print([hex(open_stream(fold(k, t, i)))
 *   for (t, i) in ((8, 1), (9, 0), (10, 0), (11, 0), (11, 1), (8, 0))])"
 * Each check also compares with the explicit rngFold chain, so a wrong tag
 * value and a wrong wiring fail differently. */
#define FUNNEL_TEST_SEED 2024u
#define FUNNEL_STAGE1_STATE 0x606fec86u
#define FUNNEL_KERNEL0_STATE 0x780e2145u
#define FUNNEL_OUT0_STATE 0xcc59e4b1u
#define FUNNEL_ACC0_STATE 0x3ecc80e6u
#define FUNNEL_ACC1_STATE 0xac44fd15u
#define FUNNEL_STAGE0_STATE 0x72195ec4u
#define FUNNEL_MAX_RECORDS 4

typedef struct {
    bool hasStream;
    uint32_t state;
} funnelRecord_t;

static funnelRecord_t funnelRecords[FUNNEL_MAX_RECORDS];
static size_t funnelRecordCount;
static conversionFunction_t funnelSavedCells[7][7];

static void funnelRecord(const rngStream_t *rs) {
    if (funnelRecordCount < FUNNEL_MAX_RECORDS) {
        funnelRecords[funnelRecordCount].hasStream = rs != NULL;
        funnelRecords[funnelRecordCount].state = (rs != NULL) ? rs->state : 0u;
    }
    funnelRecordCount++;
}

/* Records the stream, then runs the real cell, so the op completes normally. */
static void funnelRecordingCell(tensor_t *inputTensor, tensor_t *outputTensor, rngStream_t *rs) {
    funnelRecord(rs);
    funnelSavedCells[inputTensor->quantization->type][outputTensor->quantization->type](
        inputTensor, outputTensor, rs);
}

/* FLOAT32 copy of operand 0 into rawOut; records the kernel's stream. */
static void funnelRecordingKernel(tensor_t **ops, size_t nOps, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx, rngStream_t *rs) {
    (void)nOps;
    (void)auxOut;
    (void)ctx;
    size_t n = calcNumberOfElementsByTensor(rawOut);
    for (size_t i = 0; i < n; i++) {
        ((float *)rawOut->data)[i] = ((float *)ops[0]->data)[i];
    }
    funnelRecord(rs);
}

static void funnelInstallCell(qtype_t from, qtype_t to) {
    funnelSavedCells[from][to] = conversionMatrix[from][to];
    conversionMatrix[from][to] = funnelRecordingCell;
}

static void funnelRestoreCell(qtype_t from, qtype_t to) {
    conversionMatrix[from][to] = funnelSavedCells[from][to];
}

static void assertFunnelStream(size_t record, uint32_t pythonState, uint32_t tag, uint32_t index) {
    rngKey_t key = rngKeyFromSeed(FUNNEL_TEST_SEED);
    TEST_ASSERT_TRUE(funnelRecords[record].hasStream);
    TEST_ASSERT_EQUAL_HEX32(rngOpen(rngFold(key, tag, index)).state, funnelRecords[record].state);
    TEST_ASSERT_EQUAL_HEX32(pythonState, funnelRecords[record].state);
}

static tensor_t *buildInt32(size_t n, const int32_t *values) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitInt32(), NULL);
    for (size_t i = 0; i < n; i++) {
        ((int32_t *)t->data)[i] = values[i];
    }
    return t;
}

static const arithmetic_t funnelFloatArith = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};

/* An INT32 operand under FLOAT32 arithmetic is staged through [INT32][FLOAT32]
 * with STAGE(i) of its operand index (1 here), then the kernel runs with
 * KERNEL(0); the FLOAT32 target is written in place, so no epilogue runs. */
void testExecuteOpOpensTheStageAndKernelStreamsFromTheOpKey(void) {
    tensor_t *a = buildFloat(2, (float[]){1.0f, 2.0f});
    tensor_t *b = buildInt32(2, (int32_t[]){3, 4});
    tensor_t *out = buildFloat(2, (float[]){0.f, 0.f});
    rngKey_t key = rngKeyFromSeed(FUNNEL_TEST_SEED);
    funnelRecordCount = 0;
    funnelInstallCell(INT32, FLOAT32);

    executeOp(&(opSpec_t){.kernel = funnelRecordingKernel,
                          .inputs = (tensor_t *[]){a, b},
                          .nInputs = 2,
                          .arithmetic = funnelFloatArith,
                          .key = &key,
                          .mode = OUT_WRITE},
              out);

    funnelRestoreCell(INT32, FLOAT32);
    freeTensor(out);
    freeTensor(b);
    freeTensor(a);
    TEST_ASSERT_EQUAL_UINT32(2u, (uint32_t)funnelRecordCount);
    assertFunnelStream(0, FUNNEL_STAGE1_STATE, RNG_TAG_STAGE, 1u);
    assertFunnelStream(1, FUNNEL_KERNEL0_STATE, RNG_TAG_KERNEL, 0u);
}

/* A FLOAT32 intermediate written into a SYM_INT32 target goes through
 * [FLOAT32][SYM_INT32] with OUT(0), after the kernel ran with KERNEL(0). */
void testExecuteOpOpensTheOutStreamFromTheOpKey(void) {
    tensor_t *a = buildFloat(2, (float[]){1.0f, -2.0f});
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    rngKey_t key = rngKeyFromSeed(FUNNEL_TEST_SEED);
    funnelRecordCount = 0;
    funnelInstallCell(FLOAT32, SYM_INT32);

    executeOp(&(opSpec_t){.kernel = funnelRecordingKernel,
                          .inputs = (tensor_t *[]){a},
                          .nInputs = 1,
                          .arithmetic = funnelFloatArith,
                          .key = &key,
                          .mode = OUT_WRITE},
              out);

    funnelRestoreCell(FLOAT32, SYM_INT32);
    freeTensor(out);
    freeTensor(a);
    TEST_ASSERT_EQUAL_UINT32(2u, (uint32_t)funnelRecordCount);
    assertFunnelStream(0, FUNNEL_KERNEL0_STATE, RNG_TAG_KERNEL, 0u);
    assertFunnelStream(1, FUNNEL_OUT0_STATE, RNG_TAG_OUT, 0u);
}

/* A FLOAT32 operand under SYM_INT32 arithmetic is staged through
 * [FLOAT32][SYM_INT32] with STAGE(i) of its operand index (0 here). The
 * identity kernel carries the staged codes into the raw, and the FLOAT32
 * target takes them back through [SYM_INT32][FLOAT32], which is not recorded:
 * the staging is the only call the recorded cell sees. */
void testExecuteOpOpensTheSymInt32StageStreamFromTheOpKey(void) {
    tensor_t *a = buildFloat(2, (float[]){1.0f, -2.0f});
    tensor_t *out = buildFloat(2, (float[]){0.f, 0.f});
    rngKey_t key = rngKeyFromSeed(FUNNEL_TEST_SEED);
    funnelRecordCount = 0;
    funnelInstallCell(FLOAT32, SYM_INT32);

    executeOp(&(opSpec_t){.kernel = executeOpIdentityKernel,
                          .inputs = (tensor_t *[]){a},
                          .nInputs = 1,
                          .arithmetic =
                              (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY},
                          .key = &key,
                          .mode = OUT_WRITE},
              out);

    funnelRestoreCell(FLOAT32, SYM_INT32);
    freeTensor(out);
    freeTensor(a);
    TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)funnelRecordCount);
    assertFunnelStream(0, FUNNEL_STAGE0_STATE, RNG_TAG_STAGE, 0u);
}

/* The dynamic-rescale ACC of a FLOAT32 intermediate into a SYM_INT32 target
 * makes two rounding passes, each with its own stream: the increment's
 * conversion with ACC(0), then addSymInt32TensorsInplace's requantization
 * with ACC(1). Both go through [FLOAT32][SYM_INT32]. */
void testExecuteOpOpensOneAccStreamPerRoundingPass(void) {
    tensor_t *a = buildFloat(2, (float[]){0.5f, -1.0f});
    tensor_t *acc = buildSym(2, (int32_t[]){4, -2}, 0.25f);
    rngKey_t key = rngKeyFromSeed(FUNNEL_TEST_SEED);
    funnelRecordCount = 0;
    funnelInstallCell(FLOAT32, SYM_INT32);

    executeOp(&(opSpec_t){.kernel = funnelRecordingKernel,
                          .inputs = (tensor_t *[]){a},
                          .nInputs = 1,
                          .arithmetic = funnelFloatArith,
                          .key = &key,
                          .mode = OUT_ACC_DYNAMIC_RESCALE},
              acc);

    funnelRestoreCell(FLOAT32, SYM_INT32);
    freeTensor(acc);
    freeTensor(a);
    TEST_ASSERT_EQUAL_UINT32(3u, (uint32_t)funnelRecordCount);
    assertFunnelStream(1, FUNNEL_ACC0_STATE, RNG_TAG_ACC, 0u);
    assertFunnelStream(2, FUNNEL_ACC1_STATE, RNG_TAG_ACC, 1u);
}

/* Without an op key every phase gets a NULL stream: the staging, the kernel
 * and the OUT_WRITE epilogue. */
void testExecuteOpWithoutAKeyHandsEveryPhaseANullStream(void) {
    tensor_t *a = buildFloat(2, (float[]){1.0f, 2.0f});
    tensor_t *b = buildInt32(2, (int32_t[]){3, 4});
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    funnelRecordCount = 0;
    funnelInstallCell(INT32, FLOAT32);
    funnelInstallCell(FLOAT32, SYM_INT32);

    executeOp(&(opSpec_t){.kernel = funnelRecordingKernel,
                          .inputs = (tensor_t *[]){a, b},
                          .nInputs = 2,
                          .arithmetic = funnelFloatArith,
                          .mode = OUT_WRITE},
              out);

    funnelRestoreCell(FLOAT32, SYM_INT32);
    funnelRestoreCell(INT32, FLOAT32);
    freeTensor(out);
    freeTensor(b);
    freeTensor(a);
    TEST_ASSERT_EQUAL_UINT32(3u, (uint32_t)funnelRecordCount);
    for (size_t r = 0; r < 3; r++) {
        TEST_ASSERT_FALSE(funnelRecords[r].hasStream);
    }
}

/* ---- executeConvert hands its stream to the matrix cell on every route ---- */

static rngStream_t *routeCellStream;
static uint32_t routeCellCalls;

static void routeRecordingCell(tensor_t *inputTensor, tensor_t *outputTensor, rngStream_t *rs) {
    (void)inputTensor;
    (void)outputTensor;
    routeCellStream = rs;
    routeCellCalls++;
}

/* executeConvert reaches a matrix cell on three routes: the SYM_INT32 and BFP
 * diagonals (called directly, so a same-type pair requantizes instead of
 * copying) and convertTensor for every other pair. Each route must hand the
 * caller's stream through unchanged. */
void testExecuteConvertHandsTheStreamToTheCellOnEveryRoute(void) {
    static const char *const routes[3] = {"SYM_INT32 diagonal", "BFP diagonal", "convertTensor"};
    int32_t inData[1] = {0};
    int32_t outData[1] = {0};
    size_t dims[] = {1};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    symInt32QConfig_t symInQC;
    symInt32QConfig_t symOutQC;
    initSymInt32QConfig(HALF_AWAY, &symInQC);
    initSymInt32QConfig(HALF_AWAY, &symOutQC);
    bfpQConfig_t bfpInQC = {0};
    bfpQConfig_t bfpOutQC = {0};
    quantization_t inQ[3];
    quantization_t outQ[3];
    initSymInt32Quantization(&symInQC, &inQ[0]);
    initSymInt32Quantization(&symOutQC, &outQ[0]);
    initBfpQuantization(&bfpInQC, &inQ[1]);
    initBfpQuantization(&bfpOutQC, &outQ[1]);
    initInt32Quantization(&inQ[2]);
    initFloat32Quantization(&outQ[2]);
    rngStream_t stream = rngOpen(rngKeyFromSeed(5u));

    for (size_t r = 0; r < 3; r++) {
        tensor_t in;
        tensor_t out;
        setTensorValues(&in, (uint8_t *)inData, &shape, &inQ[r], NULL);
        setTensorValues(&out, (uint8_t *)outData, &shape, &outQ[r], NULL);
        conversionFunction_t *cell = &conversionMatrix[inQ[r].type][outQ[r].type];
        conversionFunction_t saved = *cell;
        *cell = routeRecordingCell;
        routeCellStream = NULL;
        routeCellCalls = 0;

        executeConvert(&in, &out, &stream);

        *cell = saved;
        TEST_ASSERT_EQUAL_UINT32_MESSAGE(1u, routeCellCalls, routes[r]);
        TEST_ASSERT_EQUAL_PTR_MESSAGE(&stream, routeCellStream, routes[r]);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testProloguePassesMatchingOperandThroughUntouched);
    RUN_TEST(testPrologueConvertsFloatOperandIntoSymArithmetic);
    RUN_TEST(testExecuteOpConvertsOnlyMismatchedOperands);
    RUN_TEST(testOutWriteSymToSymBitEqualsDiagonalRequant);
    RUN_TEST(testOutWriteSymToFloatEqualsConvertTensor);
    RUN_TEST(testAccDynamicIntoFloatIsExactAndAccumulates);
    RUN_TEST(testAccFixedIntoFloatCollapsesToExactAdd);
    RUN_TEST(testAccDynamicSymIntoSymBitEqualsAddSymInplace);
    RUN_TEST(testAccDynamicFloatIncIntoSymTargetMatchesLayerNormReference);
    RUN_TEST(testAccFixedSymIntoSymRescalesIntoExistingScale);
    RUN_TEST(testAccFixedIntoPackedSymDerivesThenCarriesGrid);
    RUN_TEST(testAccFixedSymPackedAcceptsSymInt32IntermediateBitIdenticalToFloatBridge);
    RUN_TEST(testAccDynamicSymPackedRederivesScaleMatchesRescalePrimitive);
    RUN_TEST(testAccDynamicSymPackedAcceptsSymInt32IntermediateBitIdenticalToFloatBridge);
    RUN_TEST(testAccDynamicAsymPackedMatchesRescalePrimitive);
    RUN_TEST(testAccDynamicAsymPackedAcceptsSymInt32IntermediateBitIdenticalToFloatBridge);
    RUN_TEST(testAccIntoTooWideSymTargetAborts);
    RUN_TEST(testAccIntoTooWidePackedSymTargetAborts);
    RUN_TEST(testAccFixedScaleOnAsymTargetAborts);
    RUN_TEST(testAccDynamicIntoBfpTargetRederivesExponents);
    RUN_TEST(testAccFixedIntoBfpTargetCarriesGrid);
    RUN_TEST(testAccDynamicSymInt32IntermediateIntoBfpTargetMatchesTensorTwin);
    RUN_TEST(testAccFixedSymInt32IntermediateIntoBfpTargetCarriesGrid);
    RUN_TEST(testCtxReachesKernel);
    RUN_TEST(testAuxOutIsKernelWrittenVerbatimAndNeverFunnelConverted);
    RUN_TEST(testAccFixedScaleHonorsTargetSrRoundingMode);
    RUN_TEST(testOutWriteEpilogueRoundsByArithmeticNotTargetQConfig);
    RUN_TEST(testOutWriteDetArithmeticOverridesSrTargetQConfig);
    RUN_TEST(testOutWriteSymDiagonalRoundsByArithmetic);
    RUN_TEST(testOutWriteBfpTargetUsesOpRoundingAndRestoresSlot);
    RUN_TEST(testExecuteConvertKeepsTargetStorageRounding);
    RUN_TEST(testExecuteConvertSymToSymRequantsThroughDiagonal);
    RUN_TEST(testExecuteConvertFloatToSymMatchesConvertTensor);
    RUN_TEST(testExecuteConvertBfpToBfpRoutesDiagonal);
    RUN_TEST(testExecuteOpAliasesTargetForMatchingFloatWrite);
    RUN_TEST(testExecuteOpDoesNotAliasWhenTargetIsAnInput);
    RUN_TEST(testExecuteOpAliasesSelfTargetWithWritesInPlaceSafe);
    RUN_TEST(testExecuteOpNeverAliasesSymTarget);
    RUN_TEST(testExecuteOpRejectsGroupedSymOperandByDefault);
    RUN_TEST(testExecuteOpRejectsGroupedSymOperandAtWrongPosition);
    RUN_TEST(testExecuteOpUnpacksGroupedSymWhenAllowed);
    RUN_TEST(testExecuteOpRejectsGroupedSymOperandUnderFloat32ByDefault);
    RUN_TEST(testExecuteOpRejectsGroupedAsymOperandByDefault);
    RUN_TEST(testExecuteOpRejectsGroupedAsymOperandAtWrongPosition);
    RUN_TEST(testExecuteOpRejectsGroupedAsymOperandUnderFloat32ByDefault);
    RUN_TEST(testExecuteOpUnpacksGroupedAsymWhenAllowed);
    RUN_TEST(testExecuteOpSymArithmeticRejectsBfpStoredOperand);
    RUN_TEST(testExecuteOpFloatArithmeticStillDequantsGroupedBfp);
    RUN_TEST(testExecuteOpBfpMissingStageDies);
    RUN_TEST(testExecuteOpBfpRejectsSymStoredOperand);
    RUN_TEST(testExecuteOpBfpStageGeometryMismatchDies);
    RUN_TEST(testExecuteOpBfpStoredOperandUnpacksAndBorrows);
    RUN_TEST(testExecuteOpFloat32OperandStagedToBfpCodes);
    RUN_TEST(testExecuteOpStagesGroupedAndPerTensorOperandsDistinctSlices);
    RUN_TEST(testExecuteOpBfpStagingSrDrawsFromTheStageStream);
    RUN_TEST(testExecuteOpBfpRawIsFloat32AndPacksTarget);
    RUN_TEST(testExecuteConvertHandsTheStreamToTheCellOnEveryRoute);
    RUN_TEST(testExecuteOpOpensTheStageAndKernelStreamsFromTheOpKey);
    RUN_TEST(testExecuteOpOpensTheOutStreamFromTheOpKey);
    RUN_TEST(testExecuteOpOpensTheSymInt32StageStreamFromTheOpKey);
    RUN_TEST(testExecuteOpOpensOneAccStreamPerRoundingPass);
    RUN_TEST(testExecuteOpWithoutAKeyHandsEveryPhaseANullStream);
    return UNITY_END();
}
