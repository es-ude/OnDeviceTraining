#define SOURCE_FILE "kws_raw_trace_c"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "AdaptivePool1dApi.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "NPYLoaderApi.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "Pool1dApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

#include "TraceApi.h"
#include "npy_dump_sink.h"
#include "probe_manifest.h"

#define EPOCHS 50
#define BATCH 32
#define LR 0.005f
#define MOMENTUM 0.9f
#define SEED 42
#define SHUFFLE_SEED 42
#define NUM_CLASSES_DEFAULT 6

#define IN_CHANNELS 1
#define LEN_INPUT 16000
#define DS_K 16     /* front AvgPool downsample: 16 kHz -> 1 kHz */
#define LEN_DS 1000 /* LEN_INPUT / DS_K */
#define C1_OUT 16
#define C1_K 3
#define C2_OUT 32
#define C2_K 3
#define C3_OUT 64
#define C3_K 3

/* AvgPool(ds) + 3x(Conv1d+LayerNorm+ReLU+MaxPool) + AdaptiveAvgPool + Flatten + Linear + Softmax
 * = 17 layers */
#define MODEL_SIZE 17

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

static size_t g_numClasses = NUM_CLASSES_DEFAULT;

static size_t readNumClasses(void) {
    const char *env = getenv("KWS_CLASSES");
    if (env == NULL || env[0] == '\0') {
        return NUM_CLASSES_DEFAULT;
    }
    long v = strtol(env, NULL, 10);
    if (v != 6 && v != 35) {
        fprintf(stderr, "KWS_CLASSES must be 6 or 35 (got '%s'); using %d\n", env,
                NUM_CLASSES_DEFAULT);
        return NUM_CLASSES_DEFAULT;
    }
    return (size_t)v;
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
        dims[0] = g_numClasses;
        order[0] = 0;
        shape_t *shape = reserveMemory(sizeof(shape_t));
        shape->dimensions = dims;
        shape->orderOfDimensions = order;
        shape->numberOfDimensions = 1;

        quantization_t *q = quantizationInitFloat();
        tensor_t *t = initTensor(shape, q, NULL);

        int32_t cls = ((int32_t *)intLabels->array[i]->data)[0];
        float *data = (float *)t->data;
        for (size_t c = 0; c < g_numClasses; ++c) {
            data[c] = (c == (size_t)cls) ? 1.0f : 0.0f;
        }
        arr[i] = t;
    }
    return out;
}

static void initDataSets(const char *dataDir) {
    char path[300];
    snprintf(path, sizeof(path), "%s/train_x.npy", dataDir);
    tensorArray_t *trainItems = npyLoad(path);
    snprintf(path, sizeof(path), "%s/train_y.npy", dataDir);
    tensorArray_t *trainLabelsRaw = npyLoad(path);
    reshapeItemsAddBatchDim(trainItems);
    g_trainDataset.items = trainItems;
    g_trainDataset.labels = buildOneHotLabels(trainLabelsRaw);

    snprintf(path, sizeof(path), "%s/val_x.npy", dataDir);
    tensorArray_t *valItems = npyLoad(path);
    snprintf(path, sizeof(path), "%s/val_y.npy", dataDir);
    tensorArray_t *valLabelsRaw = npyLoad(path);
    reshapeItemsAddBatchDim(valItems);
    g_valDataset.items = valItems;
    g_valDataset.labels = buildOneHotLabels(valLabelsRaw);

    snprintf(path, sizeof(path), "%s/test_x.npy", dataDir);
    tensorArray_t *testItems = npyLoad(path);
    snprintf(path, sizeof(path), "%s/test_y.npy", dataDir);
    tensorArray_t *testLabelsRaw = npyLoad(path);
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

static void buildModel(layer_t **model, layerQuant_t *lq) {
    /* Input reshaped to [1, 1, 16000]. */
    /* Front downsample: AvgPool1d(K=16,S=16) -> length 1000 (16 kHz -> 1 kHz). */
    model[0] = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = DS_K, .stride = DS_K}, lq);

    /* 3x [Conv1d -> LayerNorm([C,L]) -> ReLU -> MaxPool(4)]. Per-conv LayerNorm over the full
     * feature map (mirrors PyTorch nn.LayerNorm([C,L]), eps 1e-5) is what gives the raw model
     * stable convergence: a 10-seed sweep showed end-feature LayerNorm collapses on ~40% of
     * seeds while per-conv converges 10/10. normalizedShape is L-coupled like the MaxPool
     * inputLengths, so it tracks the downsample rate. */
    model[1] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = IN_CHANNELS, .outChannels = C1_OUT, .kernelSize = C1_K, .padding = SAME},
        lq);
    model[2] = layerNormLayerInit(&(layerNormInit_t){.normalizedShape = (size_t[]){C1_OUT, LEN_DS},
                                                     .numNormDims = 2,
                                                     .eps = 1e-5f},
                                  lq);
    model[3] = reluLayerInit(lq);
    model[4] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 4, .stride = 4, .inputChannels = C1_OUT, .inputLength = LEN_DS},
        lq);

    model[5] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = C1_OUT, .outChannels = C2_OUT, .kernelSize = C2_K, .padding = SAME},
        lq);
    model[6] = layerNormLayerInit(
        &(layerNormInit_t){
            .normalizedShape = (size_t[]){C2_OUT, LEN_DS / 4}, .numNormDims = 2, .eps = 1e-5f},
        lq);
    model[7] = reluLayerInit(lq);
    model[8] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 4, .stride = 4, .inputChannels = C2_OUT, .inputLength = LEN_DS / 4},
        lq);

    model[9] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = C2_OUT, .outChannels = C3_OUT, .kernelSize = C3_K, .padding = SAME},
        lq);
    model[10] = layerNormLayerInit(
        &(layerNormInit_t){
            .normalizedShape = (size_t[]){C3_OUT, LEN_DS / 16}, .numNormDims = 2, .eps = 1e-5f},
        lq);
    model[11] = reluLayerInit(lq);
    model[12] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 4, .stride = 4, .inputChannels = C3_OUT, .inputLength = LEN_DS / 16},
        lq);

    /* Rate-agnostic head: AdaptiveAvgPool1d(1) -> Flatten -> Linear -> Softmax. */
    model[13] = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, lq);
    model[14] = flattenLayerInit();
    model[15] =
        linearLayerInit(&(linearInit_t){.inFeatures = C3_OUT, .outFeatures = g_numClasses}, lq);
    model[16] = softmaxLayerInit(lq);
}

/* Load PyTorch state_dict from per-layer .npy files written by
 * examples/kws_raw/train_pytorch.py --save-weights.
 *
 * Returns 0 on success, non-zero on first missing file. */
static int loadStateDictFromDir(layer_t **model, const char *weightsDir) {
    char wPath[300], bPath[300];
    /* Param layers in order: conv1=model[1], ln1=model[2], conv2=model[5], ln2=model[6],
     * conv3=model[9], ln3=model[10], fc=model[15]. 7 entries (each ln = gamma/beta). */
    const char *names[7] = {"conv1", "ln1", "conv2", "ln2", "conv3", "ln3", "fc"};
    tensor_t *w[7] = {0};
    tensor_t *b[7] = {0};

    for (int i = 0; i < 7; i++) {
        snprintf(wPath, sizeof(wPath), "%s/%s.weight.npy", weightsDir, names[i]);
        snprintf(bPath, sizeof(bPath), "%s/%s.bias.npy", weightsDir, names[i]);
        w[i] = npyLoadFlat(wPath);
        b[i] = npyLoadFlat(bPath);
        if (w[i] == NULL || b[i] == NULL) {
            fprintf(stderr, "loadStateDictFromDir: missing %s or %s\n", wPath, bPath);
            return 1;
        }
    }

    modelLoadStateDict(
        model, MODEL_SIZE,
        (stateDictEntry_t[]){
            {.name = names[0], .weightData = (float *)w[0]->data, .biasData = (float *)b[0]->data},
            {.name = names[1], .weightData = (float *)w[1]->data, .biasData = (float *)b[1]->data},
            {.name = names[2], .weightData = (float *)w[2]->data, .biasData = (float *)b[2]->data},
            {.name = names[3], .weightData = (float *)w[3]->data, .biasData = (float *)b[3]->data},
            {.name = names[4], .weightData = (float *)w[4]->data, .biasData = (float *)b[4]->data},
            {.name = names[5], .weightData = (float *)w[5]->data, .biasData = (float *)b[5]->data},
            {.name = names[6], .weightData = (float *)w[6]->data, .biasData = (float *)b[6]->data},
        },
        7);

    for (int i = 0; i < 7; i++) {
        freeTensor(w[i]);
        freeTensor(b[i]);
    }
    return 0;
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

/* CLI: --sample-start N (first test sample of the batch, default 0)
 *      --batch B        (samples per step, default 32)
 *      --act-samples K  (samples that dump activations/act-grads, default 4)
 *      --steps S        (re-feed the same batch S times, default 1) */
static size_t g_sampleStart = 0;
static size_t g_batch = 32;
static size_t g_actSamples = 4;
static size_t g_steps = 1;
static void parseArgs(int argc, char **argv) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--sample-start") == 0) {
            g_sampleStart = (size_t)strtoul(argv[++i], 0, 10);
        } else if (strcmp(argv[i], "--batch") == 0) {
            g_batch = (size_t)strtoul(argv[++i], 0, 10);
        } else if (strcmp(argv[i], "--act-samples") == 0) {
            g_actSamples = (size_t)strtoul(argv[++i], 0, 10);
        } else if (strcmp(argv[i], "--steps") == 0) {
            g_steps = (size_t)strtoul(argv[++i], 0, 10);
        }
    }
}

int main(int argc, char **argv) {
    parseArgs(argc, argv);
    g_numClasses = readNumClasses();

    char dataDir[256], weightsDir[256];
    snprintf(dataDir, sizeof(dataDir), "examples/kws_raw/data/%zuclass", g_numClasses);
    snprintf(weightsDir, sizeof(weightsDir), "examples/kws_raw/weights/%zuclass", g_numClasses);
    initDataSets(dataDir);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    layer_t *model[MODEL_SIZE];
    buildModel(model, &lq);

    /* Identical start: load the exported PyTorch state_dict (same as BIT_PARITY). */
    if (loadStateDictFromDir(model, weightsDir) != 0) {
        fprintf(stderr, "trace_c: state_dict load failed\n");
        return 1;
    }

    optimizer_t *sgd = sgdMCreateOptim(
        LR, MOMENTUM, /*weightDecay*/ 0.0f, model, MODEL_SIZE, quantizationInitFloat(),
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerFunctions_t optimFns = optimizerFunctions[sgd->type];

    lossConfig_t lossCfg = {
        .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL};

    /* Effective batch: support ANY --batch, clamped to the samples available from
     * --sample-start. effB is used for the loop, the mean-scale (1/effB) and the
     * mean_loss print, so the C-vs-PyTorch scaling stays consistent for any B. */
    size_t testSize = getTestSize();
    if (g_sampleStart >= testSize) {
        fprintf(stderr, "trace_c: --sample-start %zu >= test size %zu\n", g_sampleStart, testSize);
        return 1;
    }
    size_t effB = g_batch;
    if (g_sampleStart + effB > testSize) {
        effB = testSize - g_sampleStart;
        fprintf(stderr, "trace_c: batch clamped to %zu (requested %zu, only %zu from start %zu)\n",
                effB, g_batch, effB, g_sampleStart);
    }

    /* mean over effB samples; same vtable entry TrainingEpochDefault.c:35 uses (== 1/effB for
     * CE). */
    tensor_t *firstLabel = g_testDataset.labels->array[g_sampleStart];
    float meanScale = lossFunctions[lossCfg.funcType].computeMeanScale(effB, firstLabel);

    ensureDir("examples/kws_raw/dump_c");
    for (size_t step = 0; step < g_steps; step++) {
        char dir[256];
        snprintf(dir, sizeof(dir), "examples/kws_raw/dump_c/step%03zu", step);
        ensureDir(dir);
        npyDumpCtx_t ctx = {.dir = dir,
                            .probeNames = KWS_RAW_PROBES,
                            .numProbes = MODEL_SIZE,
                            .sampleIdx = NPY_DUMP_NO_SAMPLE};

        /* tier 4a: weights before the step (unchanged during accumulation). */
        traceModelWeights(model, MODEL_SIZE, "w_before", npyDumpSink, &ctx);

        /* tiers 1 & 2 per sample (first g_actSamples); grads accumulate over ALL B samples.
         * No zero-grad between samples => param->grad ends up the SUM over the batch.
         * (Grads start at zero: calloc-backed after sgdMCreateOptim / optimFns.zero below.) */
        double sumLoss = 0.0;
        for (size_t s = 0; s < effB; s++) {
            size_t idx = g_sampleStart + s;
            sample_t *smp = getTestSample(idx);
            tensor_t *label = g_testDataset.labels->array[idx];
            bool dumpActs = (s < g_actSamples);
            ctx.sampleIdx = dumpActs ? s : NPY_DUMP_NO_SAMPLE;
            trainingStats_t *stats =
                tracedGrads(model, MODEL_SIZE, lossCfg, REDUCTION_MEAN, smp->item, label,
                            dumpActs ? npyDumpSink : NULL, dumpActs ? &ctx : NULL);
            sumLoss += (double)stats->loss;
            freeTrainingStats(stats);
            freeSample(smp);
        }
        ctx.sampleIdx = NPY_DUMP_NO_SAMPLE;

        /* tier 3a: raw accumulated grads (SUM over the batch, pre-scale). */
        traceModelGrads(model, MODEL_SIZE, "grad_raw", npyDumpSink, &ctx);

        /* mean-reduction scaling, exactly as TrainingEpochDefault does it. */
        scaleOptimizerGradients(sgd, meanScale);

        /* tier 3b: scaled grads (MEAN, pre-step). */
        traceModelGrads(model, MODEL_SIZE, "grad_scaled", npyDumpSink, &ctx);

        /* the update (through optimizerStep so an attached profiler sees the
         * OPTIMIZER phase, #432), then tier 4b: weights after. */
        optimizerStep(sgd);
        traceModelWeights(model, MODEL_SIZE, "w_after", npyDumpSink, &ctx);
        optimFns.zero(sgd);

        fprintf(stdout, "trace_c step %zu: effB=%zu mean_loss=%.6f -> %s\n", step, effB,
                sumLoss / (double)effB, dir);
    }

    return 0;
}
