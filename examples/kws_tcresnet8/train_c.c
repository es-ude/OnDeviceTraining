#define SOURCE_FILE "kws_tcresnet8_train_c"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "AdaptivePool1dApi.h"
#include "AddApi.h"
#include "BatchNorm1dApi.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "DropoutApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "LrScheduler.h"
#include "Model.h"
#include "ModelGraphApi.h"
#include "NPYLoaderApi.h"
#include "OptimizerApi.h"
#include "ParameterWalk.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

/* TC-ResNet8 (Choi et al. 2019) trained with the paper's recipe. The input is MFCC [40, T]
 * (coefficients as channels, frames as length); T and the class count come from the data. */

#define IN_CHANNELS 40
#define NUM_BLOCKS 3
#define BATCH 100
#define BASE_LR 0.1f
#define MOMENTUM 0.9f
#define WEIGHT_DECAY 1e-3f
#define LR_GAMMA 0.1f
#define PAPER_ITERS_PER_DROP 10000
#define DROPOUT_P 0.5f
#define BN_MOMENTUM 0.003f
#define BN_EPS 1e-3f

/* stem 3 + 10 per block (main 5, shortcut 3, add, relu) + head 5 */
#define MAX_NODES (3 + 10 * NUM_BLOCKS + 5)
/* one input per node, plus the second input of each Add */
#define MAX_EDGES (MAX_NODES + NUM_BLOCKS)

static const size_t FULL_STEM = 16;
static const size_t FULL_BLOCK_OUT[NUM_BLOCKS] = {24, 32, 48};

typedef struct runConfig {
    const char *dataDir;
    bool halfWidth;
    unsigned seed;
    size_t microBatch;
    size_t epochsOverride; /* 0 = the paper's iteration budget */
    const char *outDir;
} runConfig_t;

static dataset_t g_train;
static dataset_t g_val;
static dataset_t g_test;
static size_t g_numClasses;
static size_t g_inputLen;

static void failUsage(const char *msg, const char *name, const char *value) {
    fprintf(stderr, "ERROR: %s=%s: %s\n", name, value, msg);
    exit(1);
}

/* An unset knob takes dflt; a set one must be a whole number in [lo, hi]. */
static long envLong(const char *name, long dflt, long lo, long hi) {
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') {
        return dflt;
    }
    char *end = NULL;
    errno = 0;
    long x = strtol(v, &end, 10);
    if (errno != 0 || *end != '\0' || x < lo || x > hi) {
        char msg[96];
        snprintf(msg, sizeof(msg), "expected a whole number in [%ld, %ld]", lo, hi);
        failUsage(msg, name, v);
    }
    return x;
}

static runConfig_t readConfig(void) {
    runConfig_t cfg = {0};
    cfg.dataDir = getenv("KWS_DATA_DIR");
    if (cfg.dataDir == NULL || cfg.dataDir[0] == '\0') {
        fprintf(stderr, "ERROR: KWS_DATA_DIR is required (a directory with "
                        "{train,val,test}_{x,y}.npy)\n");
        exit(1);
    }
    const char *width = getenv("TCR_WIDTH");
    if (width == NULL || width[0] == '\0' || strcmp(width, "full") == 0) {
        cfg.halfWidth = false;
    } else if (strcmp(width, "half") == 0) {
        cfg.halfWidth = true;
    } else {
        failUsage("expected full or half", "TCR_WIDTH", width);
    }
    cfg.seed = (unsigned)envLong("TCR_SEED", 42, 0, (long)UINT32_MAX);
    cfg.microBatch = (size_t)envLong("TCR_MICROBATCH", BATCH, 1, BATCH);
    if (BATCH % cfg.microBatch != 0) {
        failUsage("the micro-batch must divide the batch of 100", "TCR_MICROBATCH",
                  getenv("TCR_MICROBATCH"));
    }
    cfg.epochsOverride = (size_t)envLong("TCR_EPOCHS", 0, 1, 100000);
    const char *out = getenv("TCR_OUT");
    cfg.outDir = (out != NULL && out[0] != '\0') ? out : "examples/kws_tcresnet8/logs";
    return cfg;
}

static int32_t labelAt(const tensorArray_t *labels, size_t i) {
    return ((const int32_t *)labels->array[i]->data)[0];
}

static tensorArray_t *buildOneHotLabels(const tensorArray_t *intLabels, const char *split) {
    tensorArray_t *out = reserveMemory(sizeof(tensorArray_t));
    out->array = reserveMemory(intLabels->size * sizeof(tensor_t *));
    out->size = intLabels->size;
    for (size_t i = 0; i < intLabels->size; ++i) {
        const int32_t cls = labelAt(intLabels, i);
        if (cls < 0 || (size_t)cls >= g_numClasses) {
            fprintf(stderr, "ERROR: %s label %d at row %zu is outside [0, %zu)\n", split, cls, i,
                    g_numClasses);
            exit(1);
        }
        tensor_t *t =
            initTensor(buildOwnedShape((size_t[]){g_numClasses}, 1), quantizationInitFloat(), NULL);
        ((float *)t->data)[cls] = 1.0f;
        out->array[i] = t;
    }
    return out;
}

static tensorArray_t *loadArray(const char *dir, const char *file) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    return npyLoad(path);
}

/* Every split must hold [N, 40, T] items with one T and N labels. */
static void checkItems(const tensorArray_t *items, const tensorArray_t *labels, const char *split) {
    if (items->size == 0 || items->size != labels->size) {
        fprintf(stderr, "ERROR: %s has %zu items and %zu labels\n", split, items->size,
                labels->size);
        exit(1);
    }
    const shape_t *s = items->array[0]->shape;
    if (s->numberOfDimensions != 2 || s->dimensions[0] != IN_CHANNELS ||
        s->dimensions[1] != g_inputLen) {
        fprintf(stderr, "ERROR: %s items are not [%d, %zu]\n", split, IN_CHANNELS, g_inputLen);
        exit(1);
    }
}

/* Labels first, each split's raw labels freed before the next split loads, items last: the heap
 * peak right after loading then equals the live dataset bytes, so a later peak minus those bytes
 * is training memory and never a loading transient. */
static void loadDatasets(const char *dir) {
    dataset_t *splits[3] = {&g_train, &g_val, &g_test};
    const char *names[3] = {"train", "val", "test"};
    char file[32];
    for (size_t s = 0; s < 3; ++s) {
        snprintf(file, sizeof(file), "%s_y.npy", names[s]);
        tensorArray_t *y = loadArray(dir, file);
        if (s == 0) {
            int32_t maxLabel = 0;
            for (size_t i = 0; i < y->size; ++i) {
                maxLabel = labelAt(y, i) > maxLabel ? labelAt(y, i) : maxLabel;
            }
            g_numClasses = (size_t)maxLabel + 1;
        }
        splits[s]->labels = buildOneHotLabels(y, names[s]);
        freeTensorArray(y);
    }
    for (size_t s = 0; s < 3; ++s) {
        snprintf(file, sizeof(file), "%s_x.npy", names[s]);
        splits[s]->items = loadArray(dir, file);
        if (s == 0 && splits[s]->items->size > 0) {
            g_inputLen = splits[s]->items->array[0]->shape->dimensions[1];
        }
        checkItems(splits[s]->items, splits[s]->labels, names[s]);
    }
    if (g_train.items->size < BATCH) {
        fprintf(stderr, "ERROR: train holds %zu samples, fewer than one batch of %d\n",
                g_train.items->size, BATCH);
        exit(1);
    }
}

static sample_t *getTrainSample(size_t id) {
    return npyGetSample(&g_train, id);
}
static sample_t *getValSample(size_t id) {
    return npyGetSample(&g_val, id);
}
static sample_t *getTestSample(size_t id) {
    return npyGetSample(&g_test, id);
}
static size_t getTrainSize(void) {
    return g_train.items->size;
}
static size_t getValSize(void) {
    return g_val.items->size;
}
static size_t getTestSize(void) {
    return g_test.items->size;
}

/* The model points into these arrays, so they live for the whole run. */
static layer_t *g_layers[MAX_NODES];
static uint16_t g_inputStart[MAX_NODES + 1];
static uint16_t g_inputIds[MAX_EDGES];
static const char *g_nodeNames[MAX_NODES];

/* Node j draws its init values from trainingRngInitKey(rng, j). */
typedef struct tcrBuilder {
    graphBuilder_t g;
    layerQuant_t *lq;
    const trainingRng_t *rng;
    size_t next;
} tcrBuilder_t;

static uint16_t addNode(tcrBuilder_t *b, const char *name, layer_t *layer, uint16_t in) {
    g_nodeNames[b->next++] = name;
    return graphNode(&b->g, layer, in);
}

static uint16_t conv(tcrBuilder_t *b, const char *name, size_t in, size_t out, size_t k,
                     size_t stride, uint16_t input) {
    const rngKey_t key = trainingRngInitKey(b->rng, b->next);
    layer_t *l =
        conv1dLayerInit(&(conv1dInit_t){.inChannels = in,
                                        .outChannels = out,
                                        .kernelSize = k,
                                        .stride = stride,
                                        .padding = SAME,
                                        .bias = BIAS_FALSE,
                                        .weightInit = {.scheme = INIT_XAVIER_UNIFORM, .key = &key}},
                        b->lq);
    return addNode(b, name, l, input);
}

static uint16_t bn(tcrBuilder_t *b, const char *name, size_t channels, uint16_t input) {
    layer_t *l = batchNorm1dLayerInit(&(batchNorm1dInit_t){.numChannels = channels,
                                                           .eps = BN_EPS,
                                                           .momentumMode = BN_MOMENTUM_VALUE,
                                                           .momentum = BN_MOMENTUM},
                                      b->lq);
    return addNode(b, name, l, input);
}

static uint16_t relu(tcrBuilder_t *b, const char *name, uint16_t input) {
    return addNode(b, name, reluLayerInit(b->lq), input);
}

/* main: conv k9 s2 -> BN -> ReLU -> conv k9 s1 -> BN; shortcut: conv k1 s2 -> BN -> ReLU;
 * out = ReLU(main + shortcut). */
static uint16_t residualBlock(tcrBuilder_t *b, size_t in, size_t out, uint16_t input) {
    uint16_t m = conv(b, "block.conv_a", in, out, 9, 2, input);
    m = bn(b, "block.bn_a", out, m);
    m = relu(b, "block.relu_a", m);
    m = conv(b, "block.conv_b", out, out, 9, 1, m);
    m = bn(b, "block.bn_b", out, m);
    uint16_t s = conv(b, "block.conv_short", in, out, 1, 2, input);
    s = bn(b, "block.bn_short", out, s);
    s = relu(b, "block.relu_short", s);
    g_nodeNames[b->next++] = "block.add";
    const uint16_t sum = graphMerge(&b->g, addLayerInit(NULL), m, s);
    return relu(b, "block.relu_out", sum);
}

typedef struct tcrModel {
    model_t model;
    tensor_t *dropoutMask;
    size_t numParams;
    size_t stemNode;
    size_t blockNode[NUM_BLOCKS];
    size_t poolNode;
} tcrModel_t;

static void countParameter(const paramRef_t *ref, void *ctx) {
    *(size_t *)ctx += calcNumberOfElementsByParameter(ref->parameter);
}

static void countNormParameter(const paramRef_t *ref, void *ctx) {
    if (paramSelectNormOrBias(ref, NULL)) {
        ++*(size_t *)ctx;
    }
}

static void countAllParameterTensors(const paramRef_t *ref, void *ctx) {
    (void)ref;
    ++*(size_t *)ctx;
}

static size_t channelsOf(size_t full, bool halfWidth) {
    return halfWidth ? full / 2 : full;
}

/* The count the paper's layer list gives, written out separately from the walk over the built
 * model, so a wrong layer argument shows up as a mismatch. Running statistics are buffers, not
 * parameters. */
static size_t expectedParamCount(bool halfWidth, size_t classes) {
    const size_t stem = channelsOf(FULL_STEM, halfWidth);
    size_t n = IN_CHANNELS * stem * 3 + 2 * stem;
    size_t in = stem;
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        const size_t out = channelsOf(FULL_BLOCK_OUT[i], halfWidth);
        n += in * out * 9 + out * out * 9 + in * out + 3 * 2 * out;
        in = out;
    }
    return n + in * classes;
}

/* dropoutP is a parameter so a run that must match another implementation's masks can turn the
 * draws off. The Dropout mask holds microBatch rows: train with exactly that micro-batch. */
static tcrModel_t buildModel(bool halfWidth, size_t classes, float dropoutP, size_t microBatch,
                             layerQuant_t *lq, const trainingRng_t *rng) {
    tcrModel_t r = {0};
    tcrBuilder_t b = {.lq = lq, .rng = rng, .next = 0};
    graphBegin(&b.g, g_layers, g_inputStart, g_inputIds, MAX_NODES, MAX_EDGES);

    const size_t stem = channelsOf(FULL_STEM, halfWidth);
    uint16_t h = conv(&b, "stem.conv", IN_CHANNELS, stem, 3, 1, GRAPH_INPUT);
    h = bn(&b, "stem.bn", stem, h);
    h = relu(&b, "stem.relu", h);
    r.stemNode = b.next - 1;
    size_t in = stem;
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        const size_t out = channelsOf(FULL_BLOCK_OUT[i], halfWidth);
        h = residualBlock(&b, in, out, h);
        r.blockNode[i] = b.next - 1;
        in = out;
    }
    h = addNode(&b, "head.pool",
                adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, lq), h);
    r.poolNode = b.next - 1;
    h = addNode(&b, "head.flatten", flattenLayerInit(), h);
    r.dropoutMask =
        initTensor(buildOwnedShape((size_t[]){microBatch, in}, 2), quantizationInitBool(), NULL);
    h = addNode(&b, "head.dropout",
                dropoutLayerInit(dropoutP, r.dropoutMask, lq->outputQ, lq->propLossQ), h);
    const rngKey_t fcKey = trainingRngInitKey(rng, b.next);
    h = addNode(&b, "head.fc",
                linearLayerInit(
                    &(linearInit_t){.inFeatures = in,
                                    .outFeatures = classes,
                                    .bias = BIAS_FALSE,
                                    .weightInit = {.scheme = INIT_XAVIER_UNIFORM, .key = &fcKey}},
                    lq),
                h);
    (void)addNode(&b, "head.softmax", softmaxLayerInit(lq), h);
    r.model = graphFinish(&b.g);
    return r;
}

/* Prints every node's output shape for one sample and the parameter count; exits 1 unless the
 * checkpoints match the paper's architecture (each stride-2 block maps T to ceil(T / 2)). */
static void checkModel(const tcrModel_t *r, bool halfWidth, size_t inputLen, size_t classes) {
    const size_t n = r->model.numLayers;
    size_t dims[MAX_NODES + 1][3];
    size_t order[MAX_NODES + 1][3];
    shape_t shapes[MAX_NODES + 1];
    for (size_t j = 0; j <= n; ++j) {
        shapes[j] = (shape_t){.dimensions = dims[j], .orderOfDimensions = order[j]};
    }
    size_t inDims[3] = {1, IN_CHANNELS, inputLen};
    size_t inOrder[3] = {0, 1, 2};
    const shape_t input = {
        .dimensions = inDims, .orderOfDimensions = inOrder, .numberOfDimensions = 3};
    modelShapePass(&r->model, &input, shapes);

    for (size_t j = 0; j < n; ++j) {
        const shape_t *s = &shapes[j + 1];
        printf("node %2zu %-18s [", j, g_nodeNames[j]);
        for (size_t d = 1; d < s->numberOfDimensions; ++d) {
            printf(d > 1 ? ",%zu" : "%zu", s->dimensions[d]);
        }
        printf("]\n");
    }

    bool ok = true;
    const size_t stem = channelsOf(FULL_STEM, halfWidth);
    const struct {
        size_t node, channels, length;
        const char *what;
    } want[] = {
        {r->stemNode, stem, inputLen, "stem"},
        {r->blockNode[0], channelsOf(FULL_BLOCK_OUT[0], halfWidth), (inputLen + 1) / 2, "block1"},
        {r->blockNode[1], channelsOf(FULL_BLOCK_OUT[1], halfWidth), (inputLen + 3) / 4, "block2"},
        {r->blockNode[2], channelsOf(FULL_BLOCK_OUT[2], halfWidth), (inputLen + 7) / 8, "block3"},
        {r->poolNode, channelsOf(FULL_BLOCK_OUT[2], halfWidth), 1, "pool"},
    };
    for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); ++i) {
        const shape_t *s = &shapes[want[i].node + 1];
        if (s->numberOfDimensions != 3 || s->dimensions[1] != want[i].channels ||
            s->dimensions[2] != want[i].length) {
            fprintf(stderr, "ERROR: %s output is not [%zu,%zu]\n", want[i].what, want[i].channels,
                    want[i].length);
            ok = false;
        }
    }
    const shape_t *out = &shapes[n];
    if (out->numberOfDimensions != 2 || out->dimensions[1] != classes) {
        fprintf(stderr, "ERROR: the model output is not [%zu]\n", classes);
        ok = false;
    }

    const size_t expected = expectedParamCount(halfWidth, classes);
    printf("parameters: %zu (%s width, %zu classes)\n", r->numParams, halfWidth ? "half" : "full",
           classes);
    if (r->numParams != expected) {
        fprintf(stderr, "ERROR: %zu parameters, the layer list gives %zu\n", r->numParams,
                expected);
        ok = false;
    }
    if (!ok) {
        exit(1);
    }
}

static size_t walkCount(const model_t *m, paramVisitFn_t visit) {
    size_t n = 0;
    forEachTrainableParameter(m, visit, &n);
    return n;
}

/* The paper drops the LR at 10k and 20k iterations and stops at 30k; whole epochs that come
 * closest to that at a batch of 100 (the last partial batch is dropped). */
static size_t paperStepSizeEpochs(size_t trainSize) {
    const size_t itersPerEpoch = trainSize / BATCH;
    const size_t s = (PAPER_ITERS_PER_DROP + itersPerEpoch / 2) / itersPerEpoch;
    return s > 0 ? s : 1;
}

static FILE *g_log = NULL;
static bool g_firstEpoch = true;
static struct timespec g_epochT0;

static double secondsSince(const struct timespec *t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0->tv_sec) + (double)(t1.tv_nsec - t0->tv_nsec) * 1e-9;
}

static void epochCallback(epochInfo_t info, epochStats_t evalStats) {
    const double wallS = secondsSince(&g_epochT0);
    fprintf(g_log,
            "%s    {\"epoch\": %zu, \"step_losses\": [], \"train_loss\": %.6f, "
            "\"val_loss\": %.6f, \"val_acc\": %.6f, \"wall_s\": %.4f, \"lr\": %.9g, "
            "\"batch_size\": %zu, \"parameter_updates\": %zu}",
            g_firstEpoch ? "" : ",\n", info.epoch, (double)info.trainLoss, (double)evalStats.loss,
            (double)evalStats.accuracy, wallS, (double)info.learningRate, info.batchSize,
            info.parameterUpdates);
    fflush(g_log);
    g_firstEpoch = false;
    printf("epoch %zu: lr=%.6g train_loss=%.4f val_loss=%.4f val_acc=%.4f wall_s=%.2f\n",
           info.epoch, (double)info.learningRate, (double)info.trainLoss, (double)evalStats.loss,
           (double)evalStats.accuracy, wallS);
    fflush(stdout);
    clock_gettime(CLOCK_MONOTONIC, &g_epochT0);
}

/* mkdir -p: the sweep runner hands each run its own nested directory. */
static void ensureDir(const char *dir) {
    char p[512];
    snprintf(p, sizeof(p), "%s", dir);
    for (char *c = p + 1;; ++c) {
        const bool end = *c == '\0';
        if (*c == '/' || end) {
            *c = '\0';
            if (mkdir(p, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 && errno != EEXIST) {
                fprintf(stderr, "ERROR: cannot create %s: %s\n", p, strerror(errno));
                exit(1);
            }
            if (end) {
                return;
            }
            *c = '/';
        }
    }
}

/* trainingRun over the paper's recipe: SGD with momentum, weight decay on conv/fc weights only
 * (BN scale/shift in a group without decay), STEP_LR per epoch, validation after every epoch and
 * the test set once at the end. */
static int runPaperTraining(const tcrModel_t *r, const runConfig_t *cfg, trainingRng_t *rng,
                            size_t datasetBytes) {
    const size_t stepSize = paperStepSizeEpochs(getTrainSize());
    const size_t epochs = cfg->epochsOverride > 0 ? cfg->epochsOverride : 3 * stepSize;
    const lossConfig_t lossConfig = {
        .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL};

    dataLoader_t *trainLoader = dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                                               /*shuffle*/ true, /*dropLast*/ true);
    dataLoaderSetReshufflePerEpoch(trainLoader, true);
    dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                             /*shuffle*/ false, /*dropLast*/ true);
    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, /*dropLast*/ true);

    optimizer_t *sgd =
        sgdMCreateOptim(BASE_LR, MOMENTUM, WEIGHT_DECAY, &r->model, quantizationInitFloat(),
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    (void)sgdMAddGroup(sgd, &r->model, paramSelectNormOrBias, NULL, 1.0f, 0.0f);
    lrScheduler_t lrSched;
    stepLrInit(&lrSched, sgd, stepSize, LR_GAMMA);

    ensureDir(cfg->outDir);
    char logPath[600];
    snprintf(logPath, sizeof(logPath), "%s/c.json", cfg->outDir);
    g_log = fopen(logPath, "w");
    if (g_log == NULL) {
        fprintf(stderr, "ERROR: cannot open %s: %s\n", logPath, strerror(errno));
        return 1;
    }
    printf("epochs=%zu step_size=%zu micro_batch=%zu seed=%u log=%s\n", epochs, stepSize,
           cfg->microBatch, cfg->seed, logPath);
    fprintf(g_log,
            "{\n"
            "  \"impl\": \"c\",\n"
            "  \"example\": \"kws_tcresnet8\",\n"
            "  \"config\": {\"epochs\": %zu, \"batch\": %d, \"lr\": %.6f, \"momentum\": %.6f, "
            "\"seed\": %u, \"lr_schedule\": \"step\", \"gamma\": %.6f, \"step_size\": %zu, "
            "\"optimizer\": \"sgd\", \"weight_decay\": %.6f, \"reshuffle\": 1, "
            "\"width\": \"%s\", \"micro_batch\": %zu, \"num_params\": %zu, "
            "\"toolchain\": \"%s\"},\n"
            "  \"epochs\": [\n",
            epochs, BATCH, (double)BASE_LR, (double)MOMENTUM, cfg->seed, (double)LR_GAMMA, stepSize,
            (double)WEIGHT_DECAY, cfg->halfWidth ? "half" : "full", cfg->microBatch, r->numParams,
            __VERSION__);
    fflush(g_log);

    const trainingRunOptions_t options = {.lrScheduler = &lrSched,
                                          .callback = epochCallback,
                                          .stopOnNonFiniteLoss = true,
                                          .microBatchSize = cfg->microBatch,
                                          .rng = rng};
    struct timespec runT0;
    clock_gettime(CLOCK_MONOTONIC, &runT0);
    g_epochT0 = runT0;
    const trainingRunResult_t result =
        trainingRun(&r->model, lossConfig, trainLoader, valLoader, sgd, epochs,
                    calculateGradsSequential, inferenceWithLoss, &options);
    const size_t heapPeak = memProfilePeakBytes();
    const epochStats_t test = evaluationEpochWithMetrics(
        &r->model, CROSS_ENTROPY, testLoader, inferenceWithLoss, REDUCTION_MEAN, cfg->microBatch);
    const double wallS = secondsSince(&runT0);

    fprintf(g_log,
            "\n  ],\n"
            "  \"final\": {\"test_loss\": %.6f, \"test_acc\": %.6f, \"test_auc\": null, "
            "\"diverged\": %d, \"epochs_completed\": %zu}",
            (double)test.loss, (double)test.accuracy, result.stoppedOnNonFiniteLoss ? 1 : 0,
            result.epochsCompleted);
#ifdef ODT_MEM_PROFILE
    /* Training memory excludes the dataset: the peak minus what the loaded splits hold. */
    fprintf(g_log,
            ",\n  \"memory\": {\"dataset_b\": %zu, \"heap_peak_b\": %zu, "
            "\"train_heap_peak_b\": %zu}",
            datasetBytes, heapPeak, heapPeak - datasetBytes);
    printf("MEMORY dataset_b=%zu heap_peak_b=%zu train_heap_peak_b=%zu\n", datasetBytes, heapPeak,
           heapPeak - datasetBytes);
#else
    (void)datasetBytes;
    (void)heapPeak;
#endif
    fprintf(g_log, "\n}\n");
    fclose(g_log);
    printf("FINAL test_loss=%.4f test_acc=%.4f wall_s=%.1f\n", (double)test.loss,
           (double)test.accuracy, wallS);
    /* A diverged run is a recorded result (final.diverged), not a failed program. */
    return 0;
}

int main(void) {
    const runConfig_t cfg = readConfig();
    loadDatasets(cfg.dataDir);
    const size_t datasetBytes = memProfileMark();
    printf("data: train %zu, val %zu, test %zu, [%d, %zu], %zu classes\n", getTrainSize(),
           getValSize(), getTestSize(), IN_CHANNELS, g_inputLen, g_numClasses);
#ifdef ODT_MEM_PROFILE
    printf("MEMORY after load: live %zu, peak %zu\n", datasetBytes, memProfilePeakBytes());
#endif

    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    trainingRng_t rng;
    trainingRngInit(&rng, cfg.seed);
    tcrModel_t r = buildModel(cfg.halfWidth, g_numClasses, DROPOUT_P, cfg.microBatch, &lq, &rng);
    r.numParams = walkCount(&r.model, countParameter);
    checkModel(&r, cfg.halfWidth, g_inputLen, g_numClasses);
    printf("parameter tensors: %zu, of them BN scale/shift (no weight decay): %zu\n",
           walkCount(&r.model, countAllParameterTensors), walkCount(&r.model, countNormParameter));

    return runPaperTraining(&r, &cfg, &rng, datasetBytes);
}
