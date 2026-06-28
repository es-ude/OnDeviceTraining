#define SOURCE_FILE "kws_raw_train_c"

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

#include "npy_writer.h"

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

static FILE *g_log_file = NULL;
static int g_first_epoch = 1;
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
            "    {\"epoch\": %zu, \"step_losses\": [], \"train_loss\": %.6f, "
            "\"val_loss\": %.6f, \"val_acc\": %.6f, \"wall_s\": %.4f}",
            epoch, (double)trainLoss, (double)evalStats.loss, (double)evalStats.accuracy, wall_s);
    fflush(g_log_file);
    g_first_epoch = 0;

    fprintf(stdout, "epoch %zu: train_loss=%.4f val_loss=%.4f val_acc=%.4f wall_s=%.2f\n", epoch,
            (double)trainLoss, (double)evalStats.loss, (double)evalStats.accuracy, wall_s);
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
    g_numClasses = readNumClasses();

    char dataDir[256], weightsDir[256], logsDir[256], outputsDir[256];
    snprintf(dataDir, sizeof(dataDir), "examples/kws_raw/data/%zuclass", g_numClasses);
    snprintf(weightsDir, sizeof(weightsDir), "examples/kws_raw/weights/%zuclass", g_numClasses);
    snprintf(logsDir, sizeof(logsDir), "examples/kws_raw/logs/%zuclass", g_numClasses);
    snprintf(outputsDir, sizeof(outputsDir), "examples/kws_raw/outputs/%zuclass", g_numClasses);

    if (ensureDir("examples/kws_raw/logs") != 0 || ensureDir(logsDir) != 0) {
        return 1;
    }
    if (ensureDir("examples/kws_raw/outputs") != 0 || ensureDir(outputsDir) != 0) {
        return 1;
    }

    initDataSets(dataDir);

    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, /*shuffleSeed*/ 0,
                                              /*dropLast*/ true);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    layer_t *model[MODEL_SIZE];
    buildModel(model, &lq);

    const char *bitParity = getenv("BIT_PARITY");
    if (bitParity != NULL && bitParity[0] != '\0') {
        /* Bit-parity mode: load PyTorch state_dict, skip training, run inference. */
        if (loadStateDictFromDir(model, weightsDir) != 0) {
            fprintf(stderr, "BIT_PARITY: state_dict load failed\n");
            return 1;
        }
        fprintf(stdout, "BIT_PARITY: loaded state_dict from %s\n", weightsDir);
    } else {
        dataLoader_t *trainLoader = dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                                                   /*shuffle*/ true, /*shuffleSeed*/ SHUFFLE_SEED,
                                                   /*dropLast*/ true);
        dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                                 /*shuffle*/ false, /*shuffleSeed*/ 0,
                                                 /*dropLast*/ true);

        optimizer_t *sgd =
            sgdMCreateOptim(LR, MOMENTUM, /*weightDecay*/ 0.0f, model, MODEL_SIZE, FLOAT32);

        char logPath[300];
        snprintf(logPath, sizeof(logPath), "%s/c.json", logsDir);
        g_log_file = fopen(logPath, "w");
        if (!g_log_file) {
            fprintf(stderr, "ERROR: cannot open log file for writing\n");
            return 1;
        }
        fprintf(g_log_file,
                "{\n"
                "  \"impl\": \"c\",\n"
                "  \"example\": \"kws_raw\",\n"
                "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
                "\"momentum\": %.6f, \"seed\": %d, \"shuffle_seed\": %d},\n"
                "  \"epochs\": [\n",
                EPOCHS, BATCH, (double)LR, (double)MOMENTUM, SEED, SHUFFLE_SEED);
        fflush(g_log_file);

        clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);

        trainingRunResult_t result =
            trainingRun(model, MODEL_SIZE,
                        (lossConfig_t){.funcType = CROSS_ENTROPY,
                                       .backwardReduction = REDUCTION_MEAN,
                                       .classWeights = NULL},
                        trainLoader, valLoader, sgd, EPOCHS, calculateGradsSequential,
                        inferenceWithLoss, epochCallback);
        (void)result;

        epochStats_t testStats = evaluationEpochWithMetrics(
            model, MODEL_SIZE, CROSS_ENTROPY, testLoader, inferenceWithLoss, REDUCTION_MEAN);

        fprintf(g_log_file,
                "\n  ],\n"
                "  \"final\": {\"test_loss\": %.6f, \"test_acc\": %.6f, "
                "\"test_auc\": null}\n"
                "}\n",
                (double)testStats.loss, (double)testStats.accuracy);
        fclose(g_log_file);

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
        for (size_t c = 1; c < g_numClasses; ++c) {
            if (probs[c] > best) {
                best = probs[c];
                argmax = c;
            }
        }
        predictions[i] = (int32_t)argmax;
        freeTensor(out);
        freeSample(s);
    }

    char predPath[300];
    snprintf(predPath, sizeof(predPath), "%s/c_predictions.npy", outputsDir);
    size_t outShape[] = {numTest};
    int status = 0;
    int rc = npyWriteInt32(predPath, predictions, outShape, 1);
    if (rc != 0) {
        fprintf(stderr, "ERROR: npyWriteInt32 failed (rc=%d)\n", rc);
        status = 1;
    }
    free(predictions);

    return status;
}
