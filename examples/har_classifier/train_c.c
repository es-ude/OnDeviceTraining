#define SOURCE_FILE "har_classifier_train_c"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "BsScheduler.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "LrScheduler.h"
#include "NPYLoaderApi.h"
#include "Pool1dApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

#include "mem_instrument.h"
#include "npy_writer.h"

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

/* 3 x (Conv1d + ReLU + Pool) + Flatten + Linear + Softmax = 12 layers */
#define MODEL_SIZE 12

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

/* Runtime config (env-overridable). Defaults are develop's, except RESHUFFLE,
 * which is now ON by default (mirrored in train_pytorch.py). See README. */
static int g_epochs = 20;
static float g_lr = 0.01f;
static float g_momentum = 0.9f;
static unsigned g_seed = 42;
static unsigned g_shuffleSeed = 42;
static int g_batchSize = 64;              /* BATCH_SIZE: the train loader's INITIAL batch */
static const char *g_lrSchedule = "none"; /* LR_SCHEDULE: none | step | exp */
static const char *g_bsSchedule = "none"; /* BS_SCHEDULE: none | step | exp */
static int g_bsLrCompensation = 0;        /* BS_LR_COMPENSATION: 1 = the "BC" arm */
static float g_gamma = 1.0f;              /* GAMMA: shared by both schedules; 1.0 = no-op */
static int g_stepSize = 1;                /* STEP_SIZE: step schedules only */
static int g_maxBatchSize = 0;            /* MAX_BATCH_SIZE: resolved after the data load */
static int g_reshuffle = 1;               /* RESHUFFLE: per-epoch reshuffle of the train loader */

static float envFloat(const char *name, float dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? strtof(v, NULL) : dflt;
}
static int envInt(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? (int)strtol(v, NULL, 10) : dflt;
}
static const char *envStr(const char *name, const char *dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? v : dflt;
}

/* "none" | "step" | "exp" — anything else is a fail-fast config error. */
static int isKnownSchedule(const char *s) {
    return strcmp(s, "none") == 0 || strcmp(s, "step") == 0 || strcmp(s, "exp") == 0;
}

/* __VERSION__ never contains a quote or backslash in practice, but the log is
 * a JSON contract (examples/_shared/log_schema.py): escape defensively. */
static void fputsJsonEscaped(FILE *f, const char *s) {
    for (; *s != '\0'; s++) {
        if (*s == '"' || *s == '\\') {
            fputc('\\', f);
        }
        fputc(*s, f);
    }
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

static void buildModel(layer_t **model, layerQuant_t *lq) {
    /* Block 1: Conv1d(9->16, K=7, padding=SAME), ReLU, MaxPool(K=2, S=2). */
    model[0] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = IN_CHANNELS, .outChannels = C1_OUT, .kernelSize = C1_K, .padding = SAME},
        lq);
    model[1] = reluLayerInit(lq);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = C1_OUT, .inputLength = LEN_INPUT},
        lq);

    /* Block 2 */
    model[3] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = C1_OUT, .outChannels = C2_OUT, .kernelSize = C2_K, .padding = SAME},
        lq);
    model[4] = reluLayerInit(lq);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = C2_OUT, .inputLength = LEN_INPUT / 2},
        lq);

    /* Block 3 */
    model[6] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = C2_OUT, .outChannels = C3_OUT, .kernelSize = C3_K, .padding = SAME},
        lq);
    model[7] = reluLayerInit(lq);
    model[8] = avgPool1dLayerInit(
        &(avgPool1dInit_t){.kernelSize = LEN_INPUT / 4, .stride = LEN_INPUT / 4}, lq);

    /* Head */
    model[9] = flattenLayerInit();
    model[10] =
        linearLayerInit(&(linearInit_t){.inFeatures = C3_OUT, .outFeatures = NUM_CLASSES}, lq);
    model[11] = softmaxLayerInit(lq);
}

/* Load PyTorch state_dict from per-layer .npy files written by
 * examples/har_classifier/train_pytorch.py --save-weights.
 *
 * Returns 0 on success, non-zero on first missing file. */
static int loadStateDictFromDir(layer_t **model, const char *weightsDir) {
    /* Param layer order in model[]: model[0] conv1, model[3] conv2,
     * model[6] conv3, model[10] fc. 4 entries. */
    char wPath[256], bPath[256];
    const char *names[4] = {"conv1", "conv2", "conv3", "fc"};
    tensor_t *w[4] = {0};
    tensor_t *b[4] = {0};

    for (int i = 0; i < 4; i++) {
        snprintf(wPath, sizeof(wPath), "%s/%s.weight.npy", weightsDir, names[i]);
        snprintf(bPath, sizeof(bPath), "%s/%s.bias.npy", weightsDir, names[i]);
        /* npyLoadFlat (not npyLoad): a weight file is ONE tensor of shape
         * [out, in, k] (or [out, in] for fc). npyLoad() slices dim0 (the output
         * axis) into row tensors, so array[0] is only output channel 0; the
         * subsequent layerLoadWeights memcpy then runs past that short buffer
         * into heap garbage — the issue #177 collapse. */
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
        },
        4);

    /* modelLoadStateDict copied the data into the layers; release the loaders. */
    for (int i = 0; i < 4; i++) {
        freeTensor(w[i]);
        freeTensor(b[i]);
    }
    return 0;
}

static FILE *g_log_file = NULL;
static int g_first_epoch = 1;
static struct timespec g_epoch_t0;

static void epochCallback(epochInfo_t info, epochStats_t evalStats) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall_s =
        (double)(t1.tv_sec - g_epoch_t0.tv_sec) + (double)(t1.tv_nsec - g_epoch_t0.tv_nsec) * 1e-9;

    if (!g_first_epoch) {
        fprintf(g_log_file, ",\n");
    }
    fprintf(g_log_file,
            "    {\"epoch\": %zu, \"step_losses\": [], \"train_loss\": %.6f, "
            "\"val_loss\": %.6f, \"val_acc\": %.6f, \"wall_s\": %.4f, "
            "\"lr\": %.9g, \"batch_size\": %zu, \"parameter_updates\": %zu}",
            info.epoch, (double)info.trainLoss, (double)evalStats.loss, (double)evalStats.accuracy,
            wall_s, (double)info.learningRate, info.batchSize, info.parameterUpdates);
    fflush(g_log_file);
    g_first_epoch = 0;

    fprintf(stdout, "epoch %zu: train_loss=%.4f val_loss=%.4f val_acc=%.4f wall_s=%.2f\n",
            info.epoch, (double)info.trainLoss, (double)evalStats.loss, (double)evalStats.accuracy,
            wall_s);
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

int main(void) {
    if (ensureDir("examples/har_classifier/logs") != 0) {
        return 1;
    }
    if (ensureDir("examples/har_classifier/outputs") != 0) {
        return 1;
    }

    g_epochs = envInt("EPOCHS", g_epochs);
    g_lr = envFloat("LR", g_lr);
    g_momentum = envFloat("MOMENTUM", g_momentum);
    g_seed = (unsigned)envInt("SEED", (int)g_seed);
    g_shuffleSeed = (unsigned)envInt("SHUFFLE_SEED", (int)g_shuffleSeed);
    g_batchSize = envInt("BATCH_SIZE", g_batchSize);
    g_lrSchedule = envStr("LR_SCHEDULE", g_lrSchedule);
    g_bsSchedule = envStr("BS_SCHEDULE", g_bsSchedule);
    g_bsLrCompensation = envInt("BS_LR_COMPENSATION", g_bsLrCompensation);
    g_gamma = envFloat("GAMMA", g_gamma);
    g_stepSize = envInt("STEP_SIZE", g_stepSize);
    g_reshuffle = envInt("RESHUFFLE", g_reshuffle);
    if (!isKnownSchedule(g_lrSchedule)) {
        fprintf(stderr, "ERROR: LR_SCHEDULE=%s (expected none|step|exp)\n", g_lrSchedule);
        return 1;
    }
    if (!isKnownSchedule(g_bsSchedule)) {
        fprintf(stderr, "ERROR: BS_SCHEDULE=%s (expected none|step|exp)\n", g_bsSchedule);
        return 1;
    }
    if (g_batchSize < 1 || g_batchSize > UINT16_MAX) {
        fprintf(stderr, "ERROR: BATCH_SIZE=%d must be in [1, %u]\n", g_batchSize,
                (unsigned)UINT16_MAX);
        return 1;
    }
    /* The scheduler inits take size_t, so a negative STEP_SIZE would cast to
     * SIZE_MAX and slip past their `stepSize < 1` guard as a permanently-zero
     * exponent (a silently constant schedule). Reject at the env boundary. */
    if (g_stepSize < 1) {
        fprintf(stderr, "ERROR: STEP_SIZE=%d must be >= 1\n", g_stepSize);
        return 1;
    }
    /* stepLrInit/exponentialLrInit only reject a non-finite gamma, so GAMMA <= 0
     * would flip the LR's sign every epoch (or zero it). BsScheduler already
     * rejects it; this makes the README's promise true for both schedules. */
    if ((strcmp(g_lrSchedule, "none") != 0 || strcmp(g_bsSchedule, "none") != 0) &&
        (!isfinite(g_gamma) || g_gamma <= 0.0f)) {
        fprintf(stderr, "ERROR: GAMMA=%g must be finite and > 0 when a schedule is set\n",
                (double)g_gamma);
        return 1;
    }
    /* trainingRun's guard (d) rejects a COMPENSATING batch scheduler next to an
     * LR scheduler, but only after the log file is open -- which would leave a
     * truncated JSON at LOG_PATH. Reject the same combination here, before any
     * output exists; trainingRun's guard stays as the framework backstop. */
    if (g_bsLrCompensation != 0 && strcmp(g_lrSchedule, "none") != 0 &&
        strcmp(g_bsSchedule, "none") != 0) {
        fprintf(stderr,
                "ERROR: BS_LR_COMPENSATION=1 cannot be combined with LR_SCHEDULE=%s: both would "
                "write the LR every epoch\n",
                g_lrSchedule);
        return 1;
    }
    const char *logPath = getenv("LOG_PATH");

#ifdef ODT_MEM_PROFILE
    /* Reset the heap counter before the first reserveMemory so dataset_b starts
     * from a clean baseline. */
    memProfileReset();
#endif

    initDataSets();

    /* Design assumption 3: default cap = train-set size / 10 (661 for this
     * split), i.e. >= 10 optimizer updates per epoch even at the cap. Read
     * here, after the data load, so the default can be derived from it. */
    g_maxBatchSize = envInt("MAX_BATCH_SIZE", (int)(getTrainSize() / 10));
    if (strcmp(g_bsSchedule, "none") != 0 && g_maxBatchSize > (int)getTrainSize()) {
        fprintf(stderr, "ERROR: MAX_BATCH_SIZE=%d exceeds the train set (%zu samples)\n",
                g_maxBatchSize, getTrainSize());
        return 1;
    }

#ifdef ODT_MEM_PROFILE
    size_t markDataset = memProfileMark(); /* dataset_b */
#endif

    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, /*shuffleSeed*/ 0,
                                              /*dropLast*/ true);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    /* Seed the RNG so weight init is reproducible AND matches the seed recorded
     * in the run log (previously the config claimed seed 42 but the RNG was
     * never seeded — it ran from its default state=1). BIT_PARITY overwrites
     * these weights from the state_dict, so this does not perturb the CI gate. */
    rngSetSeed(g_seed);

    layer_t *model[MODEL_SIZE];
#ifdef ODT_MEM_PROFILE
    size_t markBeforeModel = memProfileMark();
#endif
    buildModel(model, &lq);
#ifdef ODT_MEM_PROFILE
    size_t markAfterModel = memProfileMark(); /* params_grads_b = delta */
    size_t markBeforeOpt = 0, markAfterOpt = 0;
#endif
    optimizer_t *sgd = NULL;

    const char *bitParity = getenv("BIT_PARITY");
    if (bitParity != NULL && bitParity[0] != '\0') {
        /* Bit-parity mode: load PyTorch state_dict, skip training, run inference. */
        const char *wDir = "examples/har_classifier/weights";
        if (loadStateDictFromDir(model, wDir) != 0) {
            fprintf(stderr, "BIT_PARITY: state_dict load failed\n");
            return 1;
        }
        fprintf(stdout, "BIT_PARITY: loaded state_dict from %s\n", wDir);
    } else {
        dataLoader_t *trainLoader =
            dataLoaderInit(getTrainSample, getTrainSize, (uint16_t)g_batchSize, NULL, NULL,
                           /*shuffle*/ true, /*shuffleSeed*/ g_shuffleSeed, /*dropLast*/ true);
        dataLoaderSetReshufflePerEpoch(trainLoader, g_reshuffle != 0);
        dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                                 /*shuffle*/ false, /*shuffleSeed*/ 0,
                                                 /*dropLast*/ true);

#ifdef ODT_MEM_PROFILE
        markBeforeOpt = memProfileMark();
#endif
        /* FLOAT32 momentum accumulator (the conventional default). The optimizer
         * clones this config per state via getQLike and does NOT take ownership. */
        quantization_t *momentumQ = quantizationInitFloat();
        sgd = sgdMCreateOptim(g_lr, g_momentum, /*weightDecay*/ 0.0f, model, MODEL_SIZE, momentumQ,
                              (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
#ifdef ODT_MEM_PROFILE
        markAfterOpt = memProfileMark(); /* optstate_b = delta */
#endif

        /* One trainingRun call; the knobs only populate the options struct.
         * BS_LR_COMPENSATION=1 hands the optimizer to the batch scheduler (the
         * "BC" arm); together with LR_SCHEDULE != none that combination is
         * already rejected up front in main() — trainingRun's guard (d) is the
         * framework backstop. */
        lrScheduler_t lrSched;
        bsScheduler_t bsSched;
        trainingRunOptions_t options = {.callback = epochCallback};
        if (strcmp(g_lrSchedule, "step") == 0) {
            stepLrInit(&lrSched, sgd, (size_t)g_stepSize, g_gamma);
            options.lrScheduler = &lrSched;
        } else if (strcmp(g_lrSchedule, "exp") == 0) {
            exponentialLrInit(&lrSched, sgd, g_gamma);
            options.lrScheduler = &lrSched;
        }
        optimizer_t *bsOptim = (g_bsLrCompensation != 0) ? sgd : NULL;
        if (strcmp(g_bsSchedule, "step") == 0) {
            stepBsInit(&bsSched, trainLoader, bsOptim, (size_t)g_stepSize, g_gamma,
                       (size_t)g_maxBatchSize);
            options.bsScheduler = &bsSched;
        } else if (strcmp(g_bsSchedule, "exp") == 0) {
            exponentialBsInit(&bsSched, trainLoader, bsOptim, g_gamma, (size_t)g_maxBatchSize);
            options.bsScheduler = &bsSched;
        }

        const char *outLog = (logPath != NULL && logPath[0] != '\0')
                                 ? logPath
                                 : "examples/har_classifier/logs/c.json";
        g_log_file = fopen(outLog, "w");
        if (!g_log_file) {
            fprintf(stderr, "ERROR: cannot open log file for writing\n");
            return 1;
        }
        fprintf(g_log_file,
                "{\n"
                "  \"impl\": \"c\",\n"
                "  \"example\": \"har_classifier\",\n"
                "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
                "\"momentum\": %.6f, \"seed\": %u, \"shuffle_seed\": %u, "
                "\"lr_schedule\": \"%s\", \"bs_schedule\": \"%s\", \"bs_lr_compensation\": %d, "
                "\"gamma\": %#.9g, \"step_size\": %d, \"max_batch_size\": %d, "
                "\"reshuffle\": %d, \"toolchain\": \"",
                g_epochs, g_batchSize, (double)g_lr, (double)g_momentum, g_seed, g_shuffleSeed,
                g_lrSchedule, g_bsSchedule,
                (options.bsScheduler != NULL && g_bsLrCompensation != 0), (double)g_gamma,
                g_stepSize, g_maxBatchSize, g_reshuffle != 0);
        fputsJsonEscaped(g_log_file, __VERSION__);
        fprintf(g_log_file, "\"},\n"
                            "  \"epochs\": [\n");
        fflush(g_log_file);

        clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);

        trainingRunResult_t result =
            trainingRun(model, MODEL_SIZE,
                        (lossConfig_t){.funcType = CROSS_ENTROPY,
                                       .backwardReduction = REDUCTION_MEAN,
                                       .classWeights = NULL},
                        trainLoader, valLoader, sgd, g_epochs, calculateGradsSequential,
                        inferenceWithLoss, &options);
        (void)result;

        epochStats_t testStats = evaluationEpochWithMetrics(
            model, MODEL_SIZE, CROSS_ENTROPY, testLoader, inferenceWithLoss, REDUCTION_MEAN);

        /* Leave the JSON object OPEN (no closing brace): the "memory" block, if
         * profiling is enabled, is appended after predictions are written. */
        fprintf(g_log_file,
                "\n  ],\n"
                "  \"final\": {\"test_loss\": %.6f, \"test_acc\": %.6f, \"test_auc\": null}",
                (double)testStats.loss, (double)testStats.accuracy);
        fflush(g_log_file);

        fprintf(stdout, "FINAL test_loss=%.4f test_acc=%.4f\n", (double)testStats.loss,
                (double)testStats.accuracy);
    }

    /* Predictions on test set (both modes). */
    size_t numTest = getTestSize();
    int32_t *predictions = malloc(numTest * sizeof(int32_t));
    if (!predictions) {
        fprintf(stderr, "OOM allocating predictions\n");
        return 1;
    }

    for (size_t i = 0; i < numTest; ++i) {
        sample_t *s = getTestSample(i);
        tensor_t *out = inference(model, MODEL_SIZE, s->item);
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
    int rc = npyWriteInt32("examples/har_classifier/outputs/c_predictions.npy", predictions,
                           outShape, 1);
    if (rc != 0) {
        fprintf(stderr, "ERROR: npyWriteInt32 failed (rc=%d)\n", rc);
        status = 1;
    }
    free(predictions);

    /* Training mode left the run-log JSON object open. Append the memory block
     * (if profiling is enabled) after predictions are written — the stack probe
     * runs one REAL step and mutates the model, so it must follow the inference
     * loop above — then close the object. */
    if (g_log_file != NULL) {
#ifdef ODT_MEM_PROFILE
        memReport_t report = {0};
        report.sym_bits = -1; /* float binary: no SYM width */
        report.storage_dtype = "float";
        report.dataset_b = markDataset;
        report.params_grads_b = markAfterModel - markBeforeModel;
        report.optstate_b = markAfterOpt - markBeforeOpt;
        report.params_b = memInstrumentParamBytes(sgd);
        report.grads_b = memInstrumentGradBytes(sgd);
        report.optstate_analytic_b = memInstrumentOptStateBytes(sgd);
        report.activations_b = memInstrumentHarActivationBytes(MICRO_BATCH, NULL);
        report.io_b = memInstrumentHarIoBytes(MICRO_BATCH);
        report.pool_backward_b = memInstrumentPoolBackwardBytes(model, MODEL_SIZE);
        report.dx_peak_b = memInstrumentHarDxPeakBytes(MICRO_BATCH, NULL);

        sample_t *stepSample = getTrainSample(0);
        memStepCtx_t stepCtx = {
            .model = model,
            .modelSize = MODEL_SIZE,
            .lossConfig = (lossConfig_t){.funcType = CROSS_ENTROPY,
                                         .backwardReduction = REDUCTION_MEAN,
                                         .classWeights = NULL},
            .input = stepSample->item,
            .label = stepSample->label,
            .optim = sgd,
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
    }

    return status;
}
