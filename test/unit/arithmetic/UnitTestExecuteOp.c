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
 * the CODE-domain uint16 (PR4, D6): dequant = (code - zeroPoint)*scale. */
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
    convertTensor(in, inSym);
    conversionMatrix[SYM_INT32][SYM_INT32](inSym, ref);

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
                            tensor_t *auxOut, const void *ctx) {
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
    conversionMatrix[SYM_INT32][SYM_INT32](in, ref);

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
    addSymInt32TensorsInplace(refGrad, refInc);

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
    convertTensor(refIncFloat, refIncSym);
    addSymInt32TensorsInplace(refGrad, refIncSym);

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
 * so this pins the D4 bit-identical-to-old-behavior case (roundHalfAway ==
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
 * accumulate carries that grid verbatim (fit-preserving, spec D1/D2). One of
 * exactly two sanctioned contract-flip test edits in this PR (spec §4.1/§9;
 * the other is UnitTestSgd's admission flip — the PR2 ASYM-zero test's
 * config-reset extension is additive, not a flip). Parity oracle: a twin
 * target driven directly through
 * accumulateFloatIntoSymTensorFixedGrid must end up bit-identical to the
 * executeOp-driven target. inc2 deliberately SHRINKS the element that hit the
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
    accumulateFloatIntoSymTensorFixedGrid(ref, inc1, n);
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
    accumulateFloatIntoSymTensorFixedGrid(ref, inc2, n);

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

/* The float-bridge closure (spec §4.1): a SYM_INT32 intermediate must reach
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
    accumulateFloatIntoSymTensorRescale(ref, incVals, n);

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

/* ASYM DYNAMIC_RESCALE happy path (D4: the only supported ASYM accumulate
 * mode) must match accumulateFloatIntoAsymTensorRescale exactly (fresh
 * affine grid every store). Fixture matches Task 2's own ASYM-rescale
 * fixture (recon-pack precedent): ASYM@5 codes {12,16,20,24} @
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
    accumulateFloatIntoAsymTensorRescale(ref, incVals, n);

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
 * (spec §4.1: "mirror of the existing SYM_INT32 guard"). qBits=17 packs and
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

/* D4: no fit-preserving ASYM pack exists, so OUT_ACC_FIXED_SCALE on an ASYM
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

/* ---- opSpec_t: ctx / auxOut / FIXED_SCALE roundingMode (spec D1+D4) --- */

typedef struct {
    float addend;
} testCtx_t;

/* Adapter proving ctx actually reaches the kernel: adds ctx->addend to every
 * element of operand 0 into rawOut. If the funnel dropped/NULLed ctx this
 * would crash (NULL deref) or - if silently defaulted - would leave the
 * output un-added; the pinned +10 catches either regression. */
static void kernelAddsCtx(tensor_t **ops, size_t nOps, tensor_t *rawOut, tensor_t *auxOut,
                          const void *ctx) {
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
                                    const void *ctx) {
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

/* OUT_ACC_FIXED_SCALE must consult the TARGET's roundingMode (spec D4) via
 * rescaleIntoAccumulatorScale, not a bare roundf (which is HALF_AWAY-only and
 * ignores SR jitter entirely). Values are chosen so the pre-rescale ratio
 * lands exactly on a HALF_AWAY tie (705*0.01/0.02 = 352.5, -705*... = -352.5)
 * so the two rounding modes provably diverge for ANY jitter draw that lands
 * before the tie boundary.
 *
 * Derivation (verified by compiling Rounding.c+RNG.c standalone against
 * rngSetSeed(99) — see task-1-report.md): with the module-global RNG seeded
 * to 99, the two draws consumed by roundSRHalfAway (one per element, in
 * element order) are jitter0=0.005865..., jitter1=0.558617... :
 *   HALF_AWAY:   round(352.5)  = 353 ;  round(-352.5) = -353  (the OLD bare-
 *                roundf behavior this test would still show if the epilogue
 *                ignored the target's roundingMode - anti-vacuity)
 *   SR_HALF_AWAY: round(352.5 + jitter0 - 0.5) = round(352.005...) = 352
 *                 round(-352.5 + jitter1 - 0.5) = round(-352.44...) = -352
 * Target scale must stay EXACTLY 0.02 (never re-derived), matching
 * testAccFixedSymIntoSymRescalesIntoExistingScale's HALF_AWAY pin (D4's
 * "bit-identical for HALF_AWAY" case). */
void testAccFixedScaleHonorsTargetSrRoundingMode(void) {
    tensor_t *inc = buildSym(2, (int32_t[]){705, -705}, 0.01f);
    tensor_t *grad = buildSym(2, (int32_t[]){0, 0}, 0.02f);
    ((symInt32QConfig_t *)grad->quantization->qConfig)->roundingMode = SR_HALF_AWAY;
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC); /* arithmetic's own roundingMode is
                                               * irrelevant to FIXED_SCALE's
                                               * epilogue rescale (target's is
                                               * what counts, per D4) */
    initSymInt32Quantization(&arithQC, &arith);

    rngSetSeed(99);
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
    TEST_ASSERT_EQUAL_INT(352, g0);
    TEST_ASSERT_EQUAL_INT(-352, g1);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.02f, gScale);
}

/* ---- #296 Stage 1: writesInPlaceSafe / rawData aliasing (OUT_WRITE only,
 * FLOAT32 arithmetic + FLOAT32 target) ------------------------------------ */

/* probeAddOneKernel is elementwise (reads in[i] before writing out[i]). */
static const void *g_probeRawOutData;
static void probeAddOneKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                              tensor_t *auxOut, const void *ctx) {
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
 * Seed 99 draws (verified against RNG.c's xorshift32, and matching the
 * documented draws in testAccFixedScaleHonorsTargetSrRoundingMode):
 * j0=0.005865..., j1=0.558617... :
 *   SR_HALF_AWAY: round(352.5 + j0 - 0.5) = round(352.006) = 352
 *                 round(-2047.0 + j1 - 0.5) = round(-2046.94) = -2047
 *   HALF_AWAY:    round(352.5) = 353 (what the target's config would give —
 *                 anti-vacuity: the two modes provably diverge on element 0) */
void testOutWriteEpilogueRoundsByArithmeticNotTargetQConfig(void) {
    tensor_t *in = buildFloat(2, (float[]){352.5f, -2047.0f});
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f); /* storage: HALF_AWAY */

    rngSetSeed(99);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY},
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
 * the target's SR config in charge (the OLD semantics) seed 99 would yield
 * 352 on element 0 — the pinned 353 diverges from that. */
void testOutWriteDetArithmeticOverridesSrTargetQConfig(void) {
    tensor_t *in = buildFloat(2, (float[]){352.5f, -2047.0f});
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    ((symInt32QConfig_t *)out->quantization->qConfig)->roundingMode = SR_HALF_AWAY;

    rngSetSeed(99);
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
 * 705*(0.5/1.0) = 352.5 lands on the tie for element 0. Seed 99:
 * SR 352 vs the target config's HALF_AWAY 353. */
void testOutWriteSymDiagonalRoundsByArithmetic(void) {
    tensor_t *in = buildSym(2, (int32_t[]){705, 4094}, 0.5f);
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f); /* storage: HALF_AWAY */

    rngSetSeed(99);
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){in},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY},
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
 * deterministically. The target's OWN storage config is SR_HALF_AWAY (seed
 * 99, first draw 0.005865...): 3.5 + 0.005865 - 0.5 = 3.005865 -> rounds to
 * 3 -- the value storageRoundingSlot returning NULL for BFP would silently
 * substitute instead (op rounding ignored, storage mode leaks through). */
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

    rngSetSeed(99);
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
 * fixture, SR on the storage config: seed 99 must yield the SR result. */
void testExecuteConvertKeepsTargetStorageRounding(void) {
    tensor_t *in = buildFloat(2, (float[]){352.5f, -2047.0f});
    tensor_t *out = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    ((symInt32QConfig_t *)out->quantization->qConfig)->roundingMode = SR_HALF_AWAY;

    rngSetSeed(99);
    executeConvert(in, out);

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

    conversionMatrix[SYM_INT32][SYM_INT32](in, gold);
    executeConvert(in, out);

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

    convertTensor(in, gold);
    executeConvert(in, out);

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
 * Stack fixtures (Task 2-4 idiom): initTensor-built BFP tensors would leak
 * exponents[] through freeTensor until the owner-chain task of this epic PR
 * adds the freeQuantization BFP arm. */
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

    executeConvert(&src, &dst);

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

/* Final-review Fix 2: a grouped SYM input at a position OTHER than the one
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
                                        tensor_t *auxOut, const void *ctx) {
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

/* Final-review Fix 3(a): the grouped-operand deny gate must apply under
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
    return UNITY_END();
}
