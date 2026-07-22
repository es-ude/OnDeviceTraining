#define SOURCE_FILE "har_classifier_train_c_finetune"

/* #380 PR3: two-stage pretrain -> freeze -> finetune demo. Same model/data as
 * train_c.c. Stage 1 trains the full model, serializes it to an ODTS
 * checkpoint. Stage 2 rebuilds the SAME topology with the three Conv1d
 * factories set `.trainable = TRAINABLE_FALSE`, deserializes the stage-1
 * checkpoint into it (Task 1's tolerant grad-presence load: a fully-trainable
 * file loads cleanly into a frozen-backbone skeleton, grads skipped), builds
 * an optimizer over it (PR1: frozen layers are optimizer-invisible -- it
 * collects ONLY the head Linear's weight+bias), and fine-tunes just the head.
 *
 * The report lines below are the point of the example: PR2's backward
 * truncation means stage 2 never computes or allocates a dx wire below the
 * head, collapsing the transient dx ping-pong that dominates stage 1's
 * backward memory. */

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "Deserialize.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "NPYLoaderApi.h"
#include "OptimizerApi.h"
#include "Pool1dApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "Serialize.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingLoopApi.h"

#include "mem_instrument.h"
#include "npy_writer.h"

#define BATCH 64 /* macro-batch: loader groups 64 samples per optimizer step */
/* Micro-batch = concurrent samples per forward/backward. The training loop
 * streams the macro-batch one sample at a time (loss.md B=1), so peak activation
 * memory is ONE sample's worth — this is what the analytic footprint must use. */
#define MICRO_BATCH 1
#define NUM_CLASSES 6

#define IN_CHANNELS 9
#define LEN_INPUT 128

#define C1_OUT 16
#define C1_K 7
#define C2_OUT 32
#define C2_K 5
#define C3_OUT 64
#define C3_K 3

/* 3 x (Conv1d + ReLU + Pool) + Flatten + Linear + Softmax = 12 layers.
 * model[10] (head Linear) is the sole trainable layer once conv1/conv2/conv3
 * freeze -- see the dxPeakBytesFrozenHead comment below. */
#define MODEL_SIZE 12

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

/* Runtime config (env-overridable); defaults mirror train_c.c's single EPOCHS. */
static int g_stage1Epochs = 20;
static int g_stage2Epochs = 20;
static float g_lr = 0.01f;
static float g_momentum = 0.9f;
static unsigned g_seed = 42;
static unsigned g_shuffleSeed = 42;

static float envFloat(const char *name, float dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? strtof(v, NULL) : dflt;
}
static int envInt(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? (int)strtol(v, NULL, 10) : dflt;
}

static void reshapeItemsAddBatchDim(tensorArray_t *items) {
    for (size_t i = 0; i < items->size; ++i) {
        tensor_t *t = items->array[i];
        size_t oldRank = t->shape->numberOfDimensions;
        size_t newRank = oldRank + 1;

        size_t *newDims = reserveMemory(newRank * sizeof(size_t));
        size_t *newOrder = reserveMemory(newRank * sizeof(size_t));
        newDims[0] = 1;
        for (size_t d = 0; d < oldRank; ++d) {
            newDims[d + 1] = t->shape->dimensions[d];
        }
        for (size_t d = 0; d < newRank; ++d) {
            newOrder[d] = d;
        }

        freeReservedMemory(t->shape->dimensions);
        freeReservedMemory(t->shape->orderOfDimensions);
        t->shape->dimensions = newDims;
        t->shape->orderOfDimensions = newOrder;
        t->shape->numberOfDimensions = newRank;
    }
}

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
    /* Data path: reuse legacy directory; v2 doesn't duplicate the data. */
    tensorArray_t *trainItems = npyLoad("examples/har_classifier/data/train_x.npy");
    tensorArray_t *trainLabelsRaw = npyLoad("examples/har_classifier/data/train_y.npy");
    reshapeItemsAddBatchDim(trainItems);
    g_trainDataset.items = trainItems;
    g_trainDataset.labels = buildOneHotLabels(trainLabelsRaw);

    tensorArray_t *valItems = npyLoad("examples/har_classifier/data/val_x.npy");
    tensorArray_t *valLabelsRaw = npyLoad("examples/har_classifier/data/val_y.npy");
    reshapeItemsAddBatchDim(valItems);
    g_valDataset.items = valItems;
    g_valDataset.labels = buildOneHotLabels(valLabelsRaw);

    tensorArray_t *testItems = npyLoad("examples/har_classifier/data/test_x.npy");
    tensorArray_t *testLabelsRaw = npyLoad("examples/har_classifier/data/test_y.npy");
    reshapeItemsAddBatchDim(testItems);
    g_testDataset.items = testItems;
    g_testDataset.labels = buildOneHotLabels(testLabelsRaw);
}

static sample_t *getTrainSample(size_t id) {
    return npyGetSample(&g_trainDataset, id);
}
static sample_t *getValSample(size_t id) {
    return npyGetSample(&g_valDataset, id);
}
static sample_t *getTestSample(size_t id) {
    return npyGetSample(&g_testDataset, id);
}
static size_t getTrainSize(void) {
    return g_trainDataset.items->size;
}
static size_t getValSize(void) {
    return g_valDataset.items->size;
}
static size_t getTestSize(void) {
    return g_testDataset.items->size;
}

/* Same 12-layer topology as train_c.c. `freezeConv` sets `.trainable =
 * TRAINABLE_FALSE` on the three Conv1d factories (model[0]/[3]/[6]) for stage
 * 2 -- the head Linear (model[10]) always stays at the trainable default. */
static void buildModel(layer_t **model, layerQuant_t *lq, bool freezeConv) {
    trainable_t convTrainable = freezeConv ? TRAINABLE_FALSE : TRAINABLE_DEFAULT;

    /* Block 1: Conv1d(9->16, K=7, padding=SAME), ReLU, MaxPool(K=2, S=2). */
    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = IN_CHANNELS,
                                               .outChannels = C1_OUT,
                                               .kernelSize = C1_K,
                                               .padding = SAME,
                                               .trainable = convTrainable},
                               lq);
    model[1] = reluLayerInit(lq);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = C1_OUT, .inputLength = LEN_INPUT},
        lq);

    /* Block 2 */
    model[3] = conv1dLayerInit(&(conv1dInit_t){.inChannels = C1_OUT,
                                               .outChannels = C2_OUT,
                                               .kernelSize = C2_K,
                                               .padding = SAME,
                                               .trainable = convTrainable},
                               lq);
    model[4] = reluLayerInit(lq);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = C2_OUT, .inputLength = LEN_INPUT / 2},
        lq);

    /* Block 3 */
    model[6] = conv1dLayerInit(&(conv1dInit_t){.inChannels = C2_OUT,
                                               .outChannels = C3_OUT,
                                               .kernelSize = C3_K,
                                               .padding = SAME,
                                               .trainable = convTrainable},
                               lq);
    model[7] = reluLayerInit(lq);
    model[8] = avgPool1dLayerInit(
        &(avgPool1dInit_t){.kernelSize = LEN_INPUT / 4, .stride = LEN_INPUT / 4}, lq);

    /* Head — always trainable, even in stage 2. */
    model[9] = flattenLayerInit();
    model[10] =
        linearLayerInit(&(linearInit_t){.inFeatures = C3_OUT, .outFeatures = NUM_CLASSES}, lq);
    model[11] = softmaxLayerInit(lq);
}

static FILE *g_log_file = NULL;
static int g_first_epoch = 1;
static int g_currentStage = 1;
static struct timespec g_epoch_t0;

static void epochCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall_s =
        (double)(t1.tv_sec - g_epoch_t0.tv_sec) + (double)(t1.tv_nsec - g_epoch_t0.tv_nsec) * 1e-9;

    if (!g_first_epoch) {
        fprintf(g_log_file, ",\n");
    }
    fprintf(g_log_file,
            "    {\"stage\": %d, \"epoch\": %zu, \"step_losses\": [], \"train_loss\": %.6f, "
            "\"val_loss\": %.6f, \"val_acc\": %.6f, \"wall_s\": %.4f}",
            g_currentStage, epoch, (double)trainLoss, (double)evalStats.loss,
            (double)evalStats.accuracy, wall_s);
    fflush(g_log_file);
    g_first_epoch = 0;

    fprintf(stdout, "stage%d epoch %zu: train_loss=%.4f val_loss=%.4f val_acc=%.4f wall_s=%.2f\n",
            g_currentStage, epoch, (double)trainLoss, (double)evalStats.loss,
            (double)evalStats.accuracy, wall_s);
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
}

static int ensureDir(const char *p) {
    if (mkdir(p, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
        return 0;
    }
    if (errno == EEXIST) {
        return 0;
    }
    fprintf(stderr, "ERROR: cannot create %s: %s\n", p, strerror(errno));
    return 1;
}

/* Sink for traceModelWeights: accumulates calcBytesPerTensor over every
 * weight/bias PARAM tensor in the model, frozen or not. Unlike
 * memInstrumentParamBytes(optim) (which walks only the optimizer's COLLECTED
 * -- i.e. trainable -- parameters), this is the TRUE resident params_b for a
 * model with frozen layers: a frozen Conv1d's weights are still live on
 * device, just not tracked by the optimizer or trained. Every prior HAR
 * binary had 100% trainable layers, where the two sums coincide; this is the
 * first one where they diverge. */
static void sumParamBytesSink(void *ctx, size_t layerIdx, layerType_t layerType, const char *phase,
                              tensor_t *tensor) {
    (void)layerIdx;
    (void)layerType;
    (void)phase;
    size_t *total = (size_t *)ctx;
    *total += calcBytesPerTensor(tensor);
}

static size_t modelResidentParamBytes(layer_t **model, size_t modelSize) {
    size_t total = 0;
    traceModelWeights(model, modelSize, "w", sumParamBytesSink, &total);
    return total;
}

/* #380 PR3: stage 2 freezes conv1/conv2/conv3; the head Linear (model[10])
 * is the ONLY trainable layer, and it is ALSO
 * CalculateGradsSequential's backwardIndex (MODEL_SIZE-1, less 1 for the
 * CE+Softmax combined-gradient shortcut). deepestTrainableIndex() therefore
 * equals backwardIndex: the backward loop's first (only) iteration lands
 * directly on the `i == deepest` branch, which calls layer backward with
 * gradCurr == NULL -- no second dx buffer is EVER allocated (see
 * CalculateGradsSequential.c). There is no ping-pong pair in stage 2 at all;
 * the sole live gradient buffer is the combined CE+Softmax lossGrad seed,
 * shape [NUM_CLASSES]. Reusing memInstrumentHarDxPeakBytes's "2x largest
 * forward wire" formula here would misreport this collapse -- that formula
 * prices in a ping-pong that, post-truncation, never happens -- so this is
 * computed directly instead of through the shared (full-model) helper. */
static size_t dxPeakBytesFrozenHead(size_t microBatch) {
    return (size_t)NUM_CLASSES * microBatch * sizeof(float);
}

int main(void) {
    if (ensureDir("examples/har_classifier/logs") != 0) {
        return 1;
    }
    if (ensureDir("examples/har_classifier/outputs") != 0) {
        return 1;
    }

    g_stage1Epochs = envInt("STAGE1_EPOCHS", g_stage1Epochs);
    g_stage2Epochs = envInt("STAGE2_EPOCHS", g_stage2Epochs);
    g_lr = envFloat("LR", g_lr);
    g_momentum = envFloat("MOMENTUM", g_momentum);
    g_seed = (unsigned)envInt("SEED", (int)g_seed);
    g_shuffleSeed = (unsigned)envInt("SHUFFLE_SEED", (int)g_shuffleSeed);
    const char *logPath = getenv("LOG_PATH");
    const char *checkpointPath = "examples/har_classifier/outputs/har_pretrained.odts";

#ifdef ODT_MEM_PROFILE
    /* Reset the heap counter before the first reserveMemory so dataset_b starts
     * from a clean baseline. */
    memProfileReset();
#endif

    initDataSets();

#ifdef ODT_MEM_PROFILE
    size_t markDataset = memProfileMark(); /* dataset_b */
#endif

    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, /*shuffleSeed*/ 0,
                                              /*dropLast*/ true);
    dataLoader_t *trainLoader =
        dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                       /*shuffle*/ true, /*shuffleSeed*/ g_shuffleSeed, /*dropLast*/ true);
    dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                             /*shuffle*/ false, /*shuffleSeed*/ 0,
                                             /*dropLast*/ true);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    rngSetSeed(g_seed);

    const char *outLog = (logPath != NULL && logPath[0] != '\0')
                             ? logPath
                             : "examples/har_classifier/logs/c_finetune.json";
    g_log_file = fopen(outLog, "w");
    if (!g_log_file) {
        fprintf(stderr, "ERROR: cannot open log file for writing\n");
        return 1;
    }
    fprintf(g_log_file,
            "{\n"
            "  \"impl\": \"c-finetune\",\n"
            "  \"example\": \"har_classifier\",\n"
            "  \"config\": {\"stage1_epochs\": %d, \"stage2_epochs\": %d, \"batch\": %d, "
            "\"lr\": %.6f, \"momentum\": %.6f, \"seed\": %u, \"shuffle_seed\": %u},\n"
            "  \"epochs\": [\n",
            g_stage1Epochs, g_stage2Epochs, BATCH, (double)g_lr, (double)g_momentum, g_seed,
            g_shuffleSeed);
    fflush(g_log_file);

    /* ---- Stage 1: full trainable model ------------------------------------ */

    layer_t *model[MODEL_SIZE];
    buildModel(model, &lq, /*freezeConv*/ false);

    quantization_t *momentumQ1 = quantizationInitFloat();
    optimizer_t *sgd1 =
        sgdMCreateOptim(g_lr, g_momentum, /*weightDecay*/ 0.0f, model, MODEL_SIZE, momentumQ1,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
    g_currentStage = 1;
    trainingRunResult_t stage1Result = trainingRun(
        model, MODEL_SIZE,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        trainLoader, valLoader, sgd1, NULL, g_stage1Epochs, calculateGradsSequential,
        inferenceWithLoss, epochCallback);
    (void)stage1Result;

    epochStats_t stage1TestStats = evaluationEpochWithMetrics(
        model, MODEL_SIZE, CROSS_ENTROPY, testLoader, inferenceWithLoss, REDUCTION_MEAN);
    fprintf(stdout, "STAGE1 test_loss=%.4f test_acc=%.4f\n", (double)stage1TestStats.loss,
            (double)stage1TestStats.accuracy);

    /* Analytic full-model figures, captured BEFORE freeing stage 1. */
    size_t gradsFullB = memInstrumentGradBytes(sgd1);
    size_t optstateFullB = memInstrumentOptStateBytes(sgd1);
    size_t dxPeakStage1B = memInstrumentHarDxPeakBytes(MICRO_BATCH);
    size_t paramsFullB = modelResidentParamBytes(model, MODEL_SIZE);

    FILE *fOut = fopen(checkpointPath, "wb");
    if (!fOut) {
        fprintf(stderr, "ERROR: cannot open %s for writing\n", checkpointPath);
        return 1;
    }
    serializeModel(model, MODEL_SIZE, fOut);
    fclose(fOut);

    /* Teardown: freeOptim frees the tensor data/grad it collected (every
     * layer is trainable in stage 1, so this is all four param layers'
     * weight+bias buffers -- the bulk of stage 1's footprint). The layer
     * shells (layer_t/config structs) are left unfreed, matching every other
     * HAR example binary (none tear down the model; the process exit
     * reclaims it) -- they are a few pointers each, negligible next to the
     * tensor data freeOptim just released. */
    freeOptim(sgd1);

    /* ---- Stage 2: frozen backbone, head-only finetune --------------------- */

#ifdef ODT_MEM_PROFILE
    size_t markBeforeModel2 = memProfileMark();
#endif
    layer_t *model2[MODEL_SIZE];
    buildModel(model2, &lq, /*freezeConv*/ true);
#ifdef ODT_MEM_PROFILE
    size_t markAfterModel2 = memProfileMark(); /* params_grads_b = delta */
    size_t markBeforeOpt2 = 0, markAfterOpt2 = 0;
#endif

    FILE *fIn = fopen(checkpointPath, "rb");
    if (!fIn) {
        fprintf(stderr, "ERROR: cannot open %s for reading\n", checkpointPath);
        return 1;
    }
    deserializeModel(model2, MODEL_SIZE, fIn);
    fclose(fIn);

#ifdef ODT_MEM_PROFILE
    markBeforeOpt2 = memProfileMark();
#endif
    quantization_t *momentumQ2 = quantizationInitFloat();
    optimizer_t *sgd2 =
        sgdMCreateOptim(g_lr, g_momentum, /*weightDecay*/ 0.0f, model2, MODEL_SIZE, momentumQ2,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
#ifdef ODT_MEM_PROFILE
    markAfterOpt2 = memProfileMark(); /* optstate_b = delta */
#endif
    if (sgd2->sizeStates != 2) {
        fprintf(stderr,
                "ERROR: expected 2 trainable states (head weight+bias) after freezing the "
                "conv backbone, got %zu\n",
                sgd2->sizeStates);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
    g_currentStage = 2;
    trainingRunResult_t stage2Result = trainingRun(
        model2, MODEL_SIZE,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        trainLoader, valLoader, sgd2, NULL, g_stage2Epochs, calculateGradsSequential,
        inferenceWithLoss, epochCallback);
    (void)stage2Result;

    epochStats_t stage2TestStats = evaluationEpochWithMetrics(
        model2, MODEL_SIZE, CROSS_ENTROPY, testLoader, inferenceWithLoss, REDUCTION_MEAN);
    fprintf(stdout, "STAGE2 test_loss=%.4f test_acc=%.4f\n", (double)stage2TestStats.loss,
            (double)stage2TestStats.accuracy);

    fprintf(g_log_file,
            "\n  ],\n"
            "  \"final\": {\"stage1_test_loss\": %.6f, \"stage1_test_acc\": %.6f, "
            "\"stage2_test_loss\": %.6f, \"stage2_test_acc\": %.6f, \"test_auc\": null}",
            (double)stage1TestStats.loss, (double)stage1TestStats.accuracy,
            (double)stage2TestStats.loss, (double)stage2TestStats.accuracy);
    fflush(g_log_file);

    /* Analytic frozen-model figures, plus the headline dx-peak collapse. */
    size_t gradsFrozenB = memInstrumentGradBytes(sgd2);
    size_t optstateFrozenB = memInstrumentOptStateBytes(sgd2);
    size_t dxPeakStage2B = dxPeakBytesFrozenHead(MICRO_BATCH);
    size_t paramsFrozenB = modelResidentParamBytes(model2, MODEL_SIZE);

    fprintf(stdout,
            "FREEZE params_full_b=%zu params_frozen_b=%zu (unchanged: freezing "
            "doesn't evict resident weights)\n",
            paramsFullB, paramsFrozenB);
    fprintf(stdout, "FREEZE optstate_analytic_full_b=%zu optstate_analytic_frozen_b=%zu\n",
            optstateFullB, optstateFrozenB);
    fprintf(stdout, "FREEZE grads_full_b=%zu grads_frozen_b=%zu\n", gradsFullB, gradsFrozenB);
    fprintf(stdout, "FREEZE dx_peak_stage1_b=%zu dx_peak_stage2_b=%zu\n", dxPeakStage1B,
            dxPeakStage2B);
    fflush(stdout);

    /* Predictions on the fine-tuned (stage 2) model. A distinct filename from
     * train_c.c/train_c_adamw.c's c_predictions.npy -- this is a different
     * model, and clobbering their comparison artifact would silently break
     * compare.py for anyone running the binaries back-to-back. */
    size_t numTest = getTestSize();
    int32_t *predictions = malloc(numTest * sizeof(int32_t));
    if (!predictions) {
        fprintf(stderr, "OOM allocating predictions\n");
        return 1;
    }

    for (size_t i = 0; i < numTest; ++i) {
        sample_t *s = getTestSample(i);
        tensor_t *out = inference(model2, MODEL_SIZE, s->item);
        float *probs = (float *)out->data;
        size_t argmax = 0;
        float best = probs[0];
        for (size_t c = 1; c < NUM_CLASSES; ++c) {
            if (probs[c] > best) {
                best = probs[c];
                argmax = c;
            }
        }
        predictions[i] = (int32_t)argmax;
        freeTensor(out);
        freeSample(s);
    }

    size_t outShape[] = {numTest};
    int status = 0;
    int rc = npyWriteInt32("examples/har_classifier/outputs/c_predictions_finetune.npy",
                           predictions, outShape, 1);
    if (rc != 0) {
        fprintf(stderr, "ERROR: npyWriteInt32 failed (rc=%d)\n", rc);
        status = 1;
    }
    free(predictions);

    /* Leave the JSON object OPEN (no closing brace): the "memory" block, if
     * profiling is enabled, is appended after predictions are written — the stack probe
     * runs one REAL step and mutates the model, so it must follow the inference
     * loop above — then close the object. */
#ifdef ODT_MEM_PROFILE
    memReport_t report = {0};
    report.sym_bits = -1; /* float binary: no SYM width */
    report.dataset_b = markDataset;
    /* Stage 2's build/optimizer heap deltas (bracketed above, around model2's
     * buildModel/sgdMCreateOptim calls) — stage 1's already-freed footprint is
     * excluded by construction, since these are DELTAS taken after stage 1
     * finished. */
    report.params_grads_b = markAfterModel2 - markBeforeModel2;
    report.optstate_b = markAfterOpt2 - markBeforeOpt2;
    report.params_b = paramsFrozenB;
    report.grads_b = gradsFrozenB;
    report.optstate_analytic_b = optstateFrozenB;
    report.activations_b = memInstrumentHarActivationBytes(MICRO_BATCH);
    report.io_b = memInstrumentHarIoBytes(MICRO_BATCH);
    report.pool_backward_b = memInstrumentPoolBackwardBytes(model2, MODEL_SIZE);
    report.dx_peak_b = dxPeakStage2B;

    sample_t *stepSample = getTrainSample(0);
    memStepCtx_t stepCtx = {
        .model = model2,
        .modelSize = MODEL_SIZE,
        .lossConfig = (lossConfig_t){.funcType = CROSS_ENTROPY,
                                     .backwardReduction = REDUCTION_MEAN,
                                     .classWeights = NULL},
        .input = stepSample->item,
        .label = stepSample->label,
        .optim = sgd2,
    };
    report.stack_peak_b = memInstrumentStackPeakBytes(&stepCtx, 1u << 20);
    freeSample(stepSample);

    report.heap_peak_b = memProfilePeakBytes();
    report.rss_peak_kb = memProfileRssPeakKb();
    memInstrumentFinalize(&report);
    memInstrumentPrintReconciliation(&report);

    fprintf(g_log_file, ",\n  \"memory\": ");
    memInstrumentEmitJson(g_log_file, &report);
#endif
    fprintf(g_log_file, "\n}\n");
    fclose(g_log_file);

    return status;
}
