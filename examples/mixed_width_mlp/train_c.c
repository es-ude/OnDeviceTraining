#define SOURCE_FILE "mixed_width_mlp_train_c"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "NPYLoaderApi.h"
#include "Optimizer.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "Rounding.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TraceApi.h"
#include "TrainingLoopApi.h"

/* Acceptance example for the arithmetic-type-split (spec §7): mixed SYM_INT32
 * widths + per-op arithmetic divergence, all on the mnist_mlp topology.
 * Offline, fixed seed, no PyTorch twin (memory-over-accuracy: SYM divergence
 * by design) — this does NOT join the bit-parity job. */

#define NUM_CLASSES 10
#define HIDDEN 64
/* Training budget (spec §7): first 256 train samples, batch 1, 1 epoch. */
#define TRAIN_SUBSET 256
#define LR 0.01f
#define MOMENTUM 0.9f

/* Flatten -> Linear0 -> Relu -> Linear1 -> Quant1 -> Softmax = 6 layers.
 * No Quant0 (spec D3): Linear0's forward funnel already restores width to
 * SYM_INT32@12 at the wire (Linear0.outputQ), so a follow-up Quant node with
 * the IDENTICAL @12 config would just repeat that restore for free noise
 * (the double-requant anti-pattern, docs/conventions/arithmetic-sym.md).
 * Quant1 stays: it's a genuine SYM_INT32@12 -> FLOAT32 dtype change, the
 * legitimate single-requant case D3 carves out. */
#define MODEL_SIZE 6
#define FLATTEN_IDX 0
#define LINEAR0_IDX 1
#define RELU_IDX 2
#define LINEAR1_IDX 3
#define QUANT1_IDX 4
#define SOFTMAX_IDX 5

/* Pinned widths (spec §7; grad storage flipped to packed SYM by PR3, spec §8:
 * docs/superpowers/specs/2026-07-03-pr3-packed-grad-storage-design.md). */
#define WEIGHT_QMAXBITS 8
#define BIAS_QMAXBITS 16
#define GRAD_QBITS 8     /* packed SYM (was GRAD_QMAXBITS 16 SYM_INT32 pre-PR3) */
#define WIRE_QMAXBITS 12 /* matches ODT_SYM_OPERAND_QMAXBITS (Quantization.h) */

/* Byte gate expected total (spec §8, derived): Linear0 weight 784*64=50176
 * elements + bias 64 elements, Linear1 weight 64*10=640 elements + bias 10
 * elements, all packed SYM@GRAD_QBITS — calcBytesPerTensor ceils
 * qBits*N/8 per tensor => 50176 + 64 + 640 + 10 = 50890 bytes at qBits=8
 * (vs 203,560 B at FLOAT32/SYM_INT32 pre-PR3 — recon-pack §5). */
#define EXPECTED_GRAD_BYTES 50890

static dataset_t g_trainDataset;

static tensorArray_t *buildOneHotLabels(tensorArray_t *intLabels) {
    tensorArray_t *out = reserveMemory(sizeof(tensorArray_t));
    tensor_t **arr = reserveMemory(intLabels->size * sizeof(tensor_t *));
    out->array = arr;
    out->size = intLabels->size;

    for (size_t i = 0; i < intLabels->size; ++i) {
        size_t *dims = reserveMemory(1 * sizeof(size_t));
        size_t *order = reserveMemory(1 * sizeof(size_t));
        dims[0] = NUM_CLASSES;
        order[0] = 0;
        shape_t *shape = reserveMemory(sizeof(shape_t));
        shape->dimensions = dims;
        shape->orderOfDimensions = order;
        shape->numberOfDimensions = 1;

        quantization_t *q = quantizationInitFloat();
        tensor_t *t = initTensor(shape, q, NULL);

        int32_t cls = ((int32_t *)intLabels->array[i]->data)[0];
        float *data = (float *)t->data;
        for (size_t c = 0; c < NUM_CLASSES; ++c) {
            data[c] = (c == (size_t)cls) ? 1.0f : 0.0f;
        }
        arr[i] = t;
    }
    return out;
}

static void initDataSets(void) {
    tensorArray_t *trainItems = npyLoad("examples/mnist_mlp/data/train_x.npy");
    tensorArray_t *trainLabelsRaw = npyLoad("examples/mnist_mlp/data/train_y.npy");
    g_trainDataset.items = trainItems;
    g_trainDataset.labels = buildOneHotLabels(trainLabelsRaw);

    /* Only the first TRAIN_SUBSET samples are ever touched (training budget,
     * spec §7): quantize just those to SYM_INT32@12 (the operand-contract
     * width — Quantization.h's ODT_SYM_OPERAND_QMAXBITS). Linear0's forward
     * (ARITH_SYM_INT32) is the raw/unmigrated kernel (matmulValidateSymOperand,
     * Matmul.c): it fails fast unless its input tensor is natively SYM_INT32,
     * so the dataset's native FLOAT32 samples must be pre-quantized here
     * rather than relying on Flatten's dtype-passthrough. */
    quantization_t *inputQ = quantizationInitSymInt32WithBits(HALF_AWAY, WIRE_QMAXBITS);
    for (size_t i = 0; i < TRAIN_SUBSET; i++) {
        requantizeTensorInPlace(g_trainDataset.items->array[i], inputQ);
    }
}

static sample_t *getTrainSample(size_t id) {
    return npyGetSample(&g_trainDataset, id);
}
static size_t getTrainSubsetSize(void) {
    return TRAIN_SUBSET;
}

typedef struct mixedWidthQuant {
    quantization_t *floatInitQ; /* FLOAT32 — temporary weight/bias storage, see buildModel() */
    quantization_t *weightQ;    /* SYM_INT32 @8  — weightStorage (post-construction fixup) */
    quantization_t *biasQ;      /* SYM_INT32 @16 — biasStorage   (post-construction fixup) */
    quantization_t *gradQ;      /* SYM @8 (packed) — weightGradStorage/biasGradStorage knob,
                                   PR3 grad-storage flip (was SYM_INT32 @16) */
    quantization_t *operandQ; /* SYM_INT32 @12 — Linear0/Linear1 outputQ (producer-restored wire) */
    quantization_t *wireQ;    /* SYM_INT32 @12 — Relu outputQ + every layer's propLossQ */
    quantization_t *floatQ;   /* FLOAT32 — Quant1/Softmax outputQ */
} mixedWidthQuant_t;

/* Flatten [1,28,28] -> [1,784] (the channel-1 acts as batch, mnist_mlp convention) -> Linear0
 * -> Relu -> Linear1 -> Quant1 -> Softmax, CE loss (spec §7 topology). */
static void buildModel(layer_t **model, mixedWidthQuant_t *mq) {
    model[FLATTEN_IDX] = flattenLayerInit();

    /* Linear0/Linear1 share the identical pinned profile (spec §7): forward
     * ARITH_SYM_INT32 (native); weightGrad/propLoss ARITH_FLOAT32 — routed
     * through the executeOp funnel (Task 1), which dequantizes SYM operands
     * on the fly and re-quantizes the FLOAT32 intermediate into the packed
     * SYM grad target (weightGradAccMode/biasGradAccMode below), unlike the
     * raw/unmigrated forward kernel below.
     *
     * weightGradAccMode/biasGradAccMode are both set explicitly to
     * OUT_ACC_FIXED_SCALE: lqLinear is a hand-written literal (not built via
     * layerQuantInitUniform), so these fields have no implicit default —
     * leaving them at the zero-init OUT_WRITE would trip the PR3 hazard
     * guard (executeOpValidateAccMode) at the first grad accumulate. Setting
     * both to FIXED_SCALE (the fit-preserving carried-grid scheme, spec
     * §4.1) rather than the Linear.c hardcode split
     * (weight=DYNAMIC_RESCALE/bias=FIXED_SCALE) is this acceptance run's
     * deliberate choice: it exercises the new packed-target FIXED_SCALE path
     * on both weight and bias grads at once.
     *
     * biasGradMath stays ARITH_SYM_INT32 — a kept choice now, not a forced
     * one. Pre-PR3, executeOp's OUT_ACC_FIXED_SCALE epilogue had no
     * FLOAT32-intermediate-into-SYM-target bridge, so biasGradStorage being
     * SYM_INT32@16 forced biasGradMath to match. PR3 closes that gap for
     * packed SYM targets (spec §4.1: the SYM epilogue's FIXED_SCALE arm now
     * accepts BOTH FLOAT32 and SYM_INT32 intermediates), so biasGradMath
     * could switch to FLOAT32 like weightGradMath/propLossMath — SYM_INT32
     * is retained here as the minimal diff, not because the funnel still
     * demands it. weightGradMath and propLossMath already used a FLOAT32
     * intermediate before PR3 (both bridges pre-date this change) and stay
     * FLOAT32 per the pinned config.
     *
     * weightStorage/biasStorage are mq->floatInitQ: the factory's
     * PyTorch-parity init (initWeightTensor/initBiasTensor, LayerCommon.c
     * requireFloat32) requires FLOAT32-native param storage by design
     * (#270) — FLOAT32 init followed by requantizeTensorInPlace() is the
     * documented public-API path to SYM_INT32-native params. Construction
     * below uses FLOAT32 storage so the Kaiming/PyTorch init succeeds with
     * real values; main() then requantizes weights/bias into the pinned
     * SYM_INT32 widths before the first forward pass. Grad
     * tensors are unaffected: gradInit derives their dtype from
     * weightGradStorage/biasGradStorage (mq->gradQ) independent of the
     * param's own storage, so they are already correctly packed SYM@GRAD_QBITS
     * straight out of the factory. */
    layerQuant_t lqLinear = {
        .forwardMath = (arithmetic_t){ARITH_SYM_INT32, HALF_AWAY},
        .weightGradMath = (arithmetic_t){ARITH_FLOAT32, HALF_AWAY},
        .biasGradMath = (arithmetic_t){ARITH_SYM_INT32, HALF_AWAY},
        .propLossMath = (arithmetic_t){ARITH_FLOAT32, HALF_AWAY},
        .outputQ = mq->operandQ,
        .propLossQ = mq->wireQ,
        .weightStorage = mq->floatInitQ,
        .biasStorage = mq->floatInitQ,
        .weightGradStorage = mq->gradQ,
        .biasGradStorage = mq->gradQ,
        .weightGradAccMode = OUT_ACC_FIXED_SCALE,
        .biasGradAccMode = OUT_ACC_FIXED_SCALE,
    };
    model[LINEAR0_IDX] =
        linearLayerInit(&(linearInit_t){.inFeatures = 28 * 28, .outFeatures = HIDDEN}, &lqLinear);
    model[LINEAR1_IDX] = linearLayerInit(
        &(linearInit_t){.inFeatures = HIDDEN, .outFeatures = NUM_CLASSES}, &lqLinear);

    layerQuant_t lqRelu = {
        .forwardMath = (arithmetic_t){ARITH_SYM_INT32, HALF_AWAY},
        .propLossMath = (arithmetic_t){ARITH_SYM_INT32, HALF_AWAY}, /* native on SYM wires */
        .outputQ = mq->wireQ,
        .propLossQ = mq->wireQ,
    };
    model[RELU_IDX] = reluLayerInit(&lqRelu);

    /* Quant1: outputQ FLOAT32 — CE backward has FLOAT32/ASYM arms only, no
     * SYM_INT32; also showcases heterogeneous wire dtypes across the model.
     * propLossQ SYM_INT32@12 executeConverts the float loss-grad seed. */
    layerQuant_t lqQuant1 = {
        .outputQ = mq->floatQ,
        .propLossQ = mq->wireQ,
    };
    model[QUANT1_IDX] = quantLayerInit(&lqQuant1);

    layerQuant_t lqSoftmax = {
        .forwardMath = (arithmetic_t){ARITH_FLOAT32, HALF_AWAY},
        /* propLossMath/propLossQ are required non-NULL by the factory but
         * never exercised: CalculateGradsSequential's CE+Softmax shortcut
         * skips Softmax's own backward (combined gradient). */
        .propLossMath = (arithmetic_t){ARITH_FLOAT32, HALF_AWAY},
        .outputQ = mq->floatQ,
        .propLossQ = mq->wireQ,
    };
    model[SOFTMAX_IDX] = softmaxLayerInit(&lqSoftmax);
}

/* ---- Hard gates (spec §7 acceptance) --------------------------------------
 * PRINT_ERROR + exit(1) on failure — platform-independent, in-binary. */

typedef struct wireGateCtx {
    bool checked;
} wireGateCtx_t;

/* Fired via tracedGrads for the first training sample: asserts Linear0's own
 * forward wire is already SYM_INT32@12 (spec D3 — the funnel restores width
 * at the producer directly; there is no separate Quant0 wire to probe
 * anymore). */
static void linear0WireGateSink(void *ctxVoid, size_t layerIdx, layerType_t layerType,
                                const char *phase, tensor_t *tensor) {
    (void)layerType;
    if (layerIdx != LINEAR0_IDX || strcmp(phase, "fwd") != 0) {
        return;
    }
    wireGateCtx_t *ctx = ctxVoid;
    if (tensor->quantization->type != SYM_INT32) {
        PRINT_ERROR("gate: Linear0 forward wire expected SYM_INT32, got qtype %d",
                    (int)tensor->quantization->type);
        exit(1);
    }
    symInt32QConfig_t *qc = tensor->quantization->qConfig;
    if (qc->qMaxBits != WIRE_QMAXBITS) {
        PRINT_ERROR("gate: Linear0 forward wire expected qMaxBits %d, got %u", WIRE_QMAXBITS,
                    (unsigned)qc->qMaxBits);
        exit(1);
    }
    ctx->checked = true;
}

typedef struct paramGateCtx {
    bool isGrad; /* false: traceModelWeights (PARAM); true: traceModelGrads (GRAD) */
    int count;
} paramGateCtx_t;

/* Fired via traceModelWeights/traceModelGrads: asserts every LINEAR layer's
 * weight/bias PARAM tensor is SYM_INT32@8 / SYM_INT32@16 respectively, and
 * both its GRAD tensors are packed SYM@GRAD_QBITS (the grad knob, PR3 spec
 * §8 — grads are no longer SYM_INT32, so they need their own dtype/width
 * arm instead of sharing the PARAM branch below). */
static void paramGateSink(void *ctxVoid, size_t layerIdx, layerType_t layerType, const char *phase,
                          tensor_t *tensor) {
    if (layerType != LINEAR) {
        return;
    }
    paramGateCtx_t *ctx = ctxVoid;

    if (ctx->isGrad) {
        if (tensor->quantization->type != SYM) {
            PRINT_ERROR("gate: layer %zu %s expected SYM, got qtype %d", layerIdx, phase,
                        (int)tensor->quantization->type);
            exit(1);
        }
        symQConfig_t *qc = tensor->quantization->qConfig;
        if (qc->qBits != GRAD_QBITS) {
            PRINT_ERROR("gate: layer %zu %s expected qBits %u, got %u", layerIdx, phase,
                        (unsigned)GRAD_QBITS, (unsigned)qc->qBits);
            exit(1);
        }
        ctx->count++;
        return;
    }

    bool isWeight = strstr(phase, ".weight") != NULL;
    uint8_t expectedBits = isWeight ? WEIGHT_QMAXBITS : BIAS_QMAXBITS;

    if (tensor->quantization->type != SYM_INT32) {
        PRINT_ERROR("gate: layer %zu %s expected SYM_INT32, got qtype %d", layerIdx, phase,
                    (int)tensor->quantization->type);
        exit(1);
    }
    symInt32QConfig_t *qc = tensor->quantization->qConfig;
    if (qc->qMaxBits != expectedBits) {
        PRINT_ERROR("gate: layer %zu %s expected qMaxBits %u, got %u", layerIdx, phase,
                    (unsigned)expectedBits, (unsigned)qc->qMaxBits);
        exit(1);
    }
    ctx->count++;
}

typedef struct gradBytesGateCtx {
    size_t totalBytes;
} gradBytesGateCtx_t;

/* Fired via traceModelGrads: sums calcBytesPerTensor across every LINEAR
 * layer's grad tensors and asserts the packed-SYM total against
 * EXPECTED_GRAD_BYTES (spec §8) — the memory-shrink claim measured in-binary
 * rather than left as a comment. */
static void gradBytesGateSink(void *ctxVoid, size_t layerIdx, layerType_t layerType,
                              const char *phase, tensor_t *tensor) {
    (void)layerIdx;
    (void)phase;
    if (layerType != LINEAR) {
        return;
    }
    gradBytesGateCtx_t *ctx = ctxVoid;
    ctx->totalBytes += calcBytesPerTensor(tensor);
}

int main(void) {
    initDataSets();

    mixedWidthQuant_t mq = {
        .floatInitQ = quantizationInitFloat(),
        .weightQ = quantizationInitSymInt32WithBits(HALF_AWAY, WEIGHT_QMAXBITS),
        .biasQ = quantizationInitSymInt32WithBits(HALF_AWAY, BIAS_QMAXBITS),
        .gradQ = quantizationInitSym(GRAD_QBITS, HALF_AWAY),
        .operandQ = quantizationInitSymInt32WithBits(HALF_AWAY, WIRE_QMAXBITS),
        .wireQ = quantizationInitSymInt32WithBits(HALF_AWAY, WIRE_QMAXBITS),
        .floatQ = quantizationInitFloat(),
    };

    layer_t *model[MODEL_SIZE];
    rngSetSeed(1); /* fixed seed, sed-able literal: the 10-seed sweep harness
                    * patches this integer per run (spec §8) */
    buildModel(model, &mq);

    /* Requantize Linear0/Linear1's weight/bias PARAM tensors to the pinned
     * SYM_INT32 widths now that PyTorch-parity init has run against the
     * FLOAT32 init storage — the documented path to SYM_INT32-native params
     * (#270, see buildModel's comment). */
    const size_t linearIdx[2] = {LINEAR0_IDX, LINEAR1_IDX};
    for (size_t k = 0; k < 2; k++) {
        linearConfig_t *cfg = model[linearIdx[k]]->config->linear;
        requantizeTensorInPlace(cfg->weights->param, mq.weightQ);
        requantizeTensorInPlace(cfg->bias->param, mq.biasQ);
    }

    /* Factory default (#279): param write-backs run under seeded SR_HALF_AWAY
     * (dead-zone escape) -- deterministic given the fixed rngSetSeed above.
     * The gates below are dtype + loss-decrease checks, not gold values, so
     * this example deliberately trains under the framework default. */
    optimizer_t *sgd = sgdMCreateOptim(
        LR, MOMENTUM, /*weightDecay*/ 0.0f, model, MODEL_SIZE, quantizationInitFloat(),
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerFunctions_t optimFns = optimizerFunctions[sgd->type];

    lossConfig_t lossCfg = defaultLossConfig(CROSS_ENTROPY);

    /* Same 256-sample subset for training and for the before/after loss
     * sanity check (index-based batches, so re-running is safe/repeatable). */
    dataLoader_t *evalLoader = dataLoaderInit(getTrainSample, getTrainSubsetSize, /*batchSize*/ 32,
                                              NULL, NULL, /*shuffle*/ false, /*shuffleSeed*/ 0,
                                              /*dropLast*/ true);

    epochStats_t initialStats = evaluationEpochWithMetrics(
        model, MODEL_SIZE, CROSS_ENTROPY, evalLoader, inferenceWithLoss, REDUCTION_MEAN);
    fprintf(stdout, "initial_loss=%.6f\n", (double)initialStats.loss);

    wireGateCtx_t wireCtx = {.checked = false};

    for (size_t i = 0; i < TRAIN_SUBSET; i++) {
        sample_t *smp = getTrainSample(i);
        tensor_t *label = g_trainDataset.labels->array[i];

        trainingStats_t *stats;
        if (i == 0) {
            /* One traced step: also probes Linear0's own forward wire. */
            stats = tracedGrads(model, MODEL_SIZE, lossCfg, REDUCTION_MEAN, smp->item, label,
                                linear0WireGateSink, &wireCtx);
        } else {
            stats = calculateGradsSequential(model, MODEL_SIZE, lossCfg, REDUCTION_MEAN, smp->item,
                                             label);
        }
        freeTrainingStats(stats);
        freeSample(smp);

        /* batch=1 => computeMeanScaleCE(1, ...) == 1.0 (identity): the mean-
         * reduction macro-batch scale trainingEpochDefault would apply here
         * is a no-op, so a bare step/zero per sample is exact (CLAUDE.md
         * loss/training_loop microbatch contract). */
        optimFns.step(sgd);
        optimFns.zero(sgd);

        if (i == 0) {
            /* Hard gates (spec §7): after one training step, storage/wire
             * dtypes must already match the pinned config. */
            if (!wireCtx.checked) {
                PRINT_ERROR("gate: Linear0 forward-wire probe never fired (layer index wrong?)");
                exit(1);
            }

            paramGateCtx_t weightsCtx = {.isGrad = false, .count = 0};
            traceModelWeights(model, MODEL_SIZE, "gate", paramGateSink, &weightsCtx);
            if (weightsCtx.count != 4) {
                PRINT_ERROR("gate: expected 4 weight/bias param checks (2 Linear layers), saw %d",
                            weightsCtx.count);
                exit(1);
            }

            paramGateCtx_t gradsCtx = {.isGrad = true, .count = 0};
            traceModelGrads(model, MODEL_SIZE, "gate", paramGateSink, &gradsCtx);
            if (gradsCtx.count != 4) {
                PRINT_ERROR("gate: expected 4 grad param checks (2 Linear layers), saw %d",
                            gradsCtx.count);
                exit(1);
            }

            gradBytesGateCtx_t bytesCtx = {.totalBytes = 0};
            traceModelGrads(model, MODEL_SIZE, "gate", gradBytesGateSink, &bytesCtx);
            if (bytesCtx.totalBytes != EXPECTED_GRAD_BYTES) {
                PRINT_ERROR("gate: expected %d packed grad bytes total, saw %zu",
                            EXPECTED_GRAD_BYTES, bytesCtx.totalBytes);
                exit(1);
            }

            fprintf(stdout,
                    "GATES PASS: weight=SYM_INT32@%d bias=SYM_INT32@%d grads=SYM@%d "
                    "Linear0-wire=SYM_INT32@%d grad_bytes=%zu\n",
                    WEIGHT_QMAXBITS, BIAS_QMAXBITS, GRAD_QBITS, WIRE_QMAXBITS, bytesCtx.totalBytes);
        }
    }

    epochStats_t finalStats = evaluationEpochWithMetrics(
        model, MODEL_SIZE, CROSS_ENTROPY, evalLoader, inferenceWithLoss, REDUCTION_MEAN);
    fprintf(stdout, "final_loss=%.6f\n", (double)finalStats.loss);

    /* Sanity gate — no pinned float values (libm differs across platforms). */
    if (!(finalStats.loss < initialStats.loss)) {
        PRINT_ERROR("gate: final loss %.6f is not < initial loss %.6f", (double)finalStats.loss,
                    (double)initialStats.loss);
        exit(1);
    }
    fprintf(stdout, "GATE PASS: final loss (%.6f) < initial loss (%.6f)\n", (double)finalStats.loss,
            (double)initialStats.loss);

    return 0;
}
