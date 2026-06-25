#define SOURCE_FILE "ecg_anomaly_ae_train_c"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "Conv1dTransposedApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "LossFunction.h"
#include "NPYLoaderApi.h"
#include "Pool1dApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

#include "npy_writer.h"

#define EPOCHS 200
#define BATCH 32
#define LR 0.005f
#define MOMENTUM 0.9f
#define SEED 42
#define SHUFFLE_SEED 42

#define IN_CHANNELS 1
#define LEN_INPUT 140

#define E1_OUT 8
#define E1_K 7
#define E1_S 2
/* enc1 is a stride-2 conv; PyTorch trained it with symmetric padding=3. C SAME
 * would pick the minimal/asymmetric pad {2,3} and diverge, so use EXPLICIT
 * padding=(K-1)/2=3 to match PyTorch bit-for-bit (issue #177). */
#define E1_PAD (E1_K / 2)
#define E2_OUT 16
#define E2_K 5

#define D1_OUT 8
#define D1_K 5
#define D1_S 5
#define D2_OUT 4
#define D2_K 2
#define D2_S 2
#define D3_OUT 1
#define D3_K 2
#define D3_S 2

#define MODEL_SIZE 11

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

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

static void initDataSets(void) {
    tensorArray_t *trainItems = npyLoad("examples/ecg_anomaly_ae/data/train_x.npy");
    tensorArray_t *trainLabels = npyLoad("examples/ecg_anomaly_ae/data/train_x.npy");
    reshapeItemsAddBatchDim(trainItems);
    reshapeItemsAddBatchDim(trainLabels);
    g_trainDataset.items = trainItems;
    g_trainDataset.labels = trainLabels;

    tensorArray_t *valItems = npyLoad("examples/ecg_anomaly_ae/data/val_x.npy");
    tensorArray_t *valLabels = npyLoad("examples/ecg_anomaly_ae/data/val_x.npy");
    reshapeItemsAddBatchDim(valItems);
    reshapeItemsAddBatchDim(valLabels);
    g_valDataset.items = valItems;
    g_valDataset.labels = valLabels;

    tensorArray_t *testItems = npyLoad("examples/ecg_anomaly_ae/data/test_x.npy");
    tensorArray_t *testLabels = npyLoad("examples/ecg_anomaly_ae/data/test_x.npy");
    reshapeItemsAddBatchDim(testItems);
    reshapeItemsAddBatchDim(testLabels);
    g_testDataset.items = testItems;
    g_testDataset.labels = testLabels;
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
    /* Encoder */
    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = IN_CHANNELS,
                                               .outChannels = E1_OUT,
                                               .kernelSize = E1_K,
                                               .stride = E1_S,
                                               .padding = EXPLICIT,
                                               .paddingAmount = E1_PAD},
                               lq);
    model[1] = reluLayerInit(lq);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = E1_OUT, .inputLength = LEN_INPUT / E1_S},
        lq);

    model[3] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = E1_OUT, .outChannels = E2_OUT, .kernelSize = E2_K, .padding = SAME},
        lq);
    model[4] = reluLayerInit(lq);
    model[5] = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 5, .stride = 5}, lq);

    /* Decoder */
    model[6] = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = E2_OUT, .outChannels = D1_OUT, .kernelSize = D1_K, .stride = D1_S},
        lq);
    model[7] = reluLayerInit(lq);

    model[8] = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = D1_OUT, .outChannels = D2_OUT, .kernelSize = D2_K, .stride = D2_S},
        lq);
    model[9] = reluLayerInit(lq);

    model[10] = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = D2_OUT, .outChannels = D3_OUT, .kernelSize = D3_K, .stride = D3_S},
        lq);
}

static int loadStateDictFromDir(layer_t **model, const char *weightsDir) {
    /* Param layer order in model[]: e1 (0), e2 (3), d1 (6), d2 (8), d3 (10). 5 entries. */
    char wPath[256], bPath[256];
    const char *names[5] = {"e1", "e2", "d1", "d2", "d3"};
    tensor_t *w[5] = {0};
    tensor_t *b[5] = {0};

    for (int i = 0; i < 5; i++) {
        snprintf(wPath, sizeof(wPath), "%s/%s.weight.npy", weightsDir, names[i]);
        snprintf(bPath, sizeof(bPath), "%s/%s.bias.npy", weightsDir, names[i]);
        /* npyLoadFlat (not npyLoad): a weight file is ONE tensor of shape
         * [out, in, k] (Conv1d) or [in, out, k] (ConvTranspose1d). npyLoad()
         * slices dim0 into row tensors, so array[0] is only the first channel;
         * the subsequent layerLoadWeights memcpy then runs past that short
         * buffer into heap garbage — the issue #177 collapse. */
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
        },
        5);

    /* modelLoadStateDict copied the data into the layers; release the loaders. */
    for (int i = 0; i < 5; i++) {
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
            "\"val_loss\": %.6f, \"val_acc\": null, \"wall_s\": %.4f}",
            epoch, (double)trainLoss, (double)evalStats.loss, wall_s);
    fflush(g_log_file);
    g_first_epoch = 0;

    fprintf(stdout, "epoch %zu: train_loss=%.6f val_loss=%.6f wall_s=%.2f\n", epoch,
            (double)trainLoss, (double)evalStats.loss, wall_s);
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
}

static int writeAllReconstructions(layer_t **model, size_t modelSize,
                                   sample_t *(*getSample)(size_t), size_t n, const char *outPath) {
    size_t totalElems = n * IN_CHANNELS * LEN_INPUT;
    float *buf = malloc(totalElems * sizeof(float));
    if (!buf) {
        fprintf(stderr, "OOM allocating reconstruction buffer (n=%zu)\n", n);
        return 1;
    }

    for (size_t i = 0; i < n; ++i) {
        sample_t *s = getSample(i);
        tensor_t *out = inference(model, modelSize, s->item);
        const float *recon = (const float *)out->data;
        memcpy(buf + i * IN_CHANNELS * LEN_INPUT, recon, IN_CHANNELS * LEN_INPUT * sizeof(float));
        freeTensor(out);
        freeSample(s);
    }

    size_t outShape[3] = {n, IN_CHANNELS, LEN_INPUT};
    int rc = npyWriteFloat32(outPath, buf, outShape, 3);
    free(buf);
    return rc;
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
    if (ensureDir("examples/ecg_anomaly_ae/logs") != 0) {
        return 1;
    }
    if (ensureDir("examples/ecg_anomaly_ae/outputs") != 0) {
        return 1;
    }

    initDataSets();

    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, /*shuffleSeed*/ 0,
                                              /*dropLast*/ true);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    layer_t *model[MODEL_SIZE];
    buildModel(model, &lq);

    const char *bitParity = getenv("BIT_PARITY");
    if (bitParity != NULL && bitParity[0] != '\0') {
        const char *wDir = "examples/ecg_anomaly_ae/weights";
        if (loadStateDictFromDir(model, wDir) != 0) {
            fprintf(stderr, "BIT_PARITY: state_dict load failed\n");
            return 1;
        }
        fprintf(stdout, "BIT_PARITY: loaded state_dict from %s\n", wDir);
    } else {
        dataLoader_t *trainLoader = dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                                                   /*shuffle*/ true, /*shuffleSeed*/ SHUFFLE_SEED,
                                                   /*dropLast*/ true);
        dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                                 /*shuffle*/ false, /*shuffleSeed*/ 0,
                                                 /*dropLast*/ true);

        optimizer_t *sgd =
            sgdMCreateOptim(LR, MOMENTUM, /*weightDecay*/ 0.0f, model, MODEL_SIZE, FLOAT32);

        g_log_file = fopen("examples/ecg_anomaly_ae/logs/c.json", "w");
        if (!g_log_file) {
            fprintf(stderr, "ERROR: cannot open log file for writing\n");
            return 1;
        }
        fprintf(g_log_file,
                "{\n"
                "  \"impl\": \"c\",\n"
                "  \"example\": \"ecg_anomaly_ae\",\n"
                "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
                "\"momentum\": %.6f, \"seed\": %d, \"shuffle_seed\": %d},\n"
                "  \"epochs\": [\n",
                EPOCHS, BATCH, (double)LR, (double)MOMENTUM, SEED, SHUFFLE_SEED);
        fflush(g_log_file);

        clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);

        trainingRunResult_t result = trainingRun(
            model, MODEL_SIZE,
            (lossConfig_t){
                .funcType = MSE, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
            trainLoader, valLoader, sgd, EPOCHS, calculateGradsSequential, inferenceWithLoss,
            epochCallback);
        (void)result;

        float testLoss =
            evaluationEpoch(model, MODEL_SIZE, MSE, testLoader, inferenceWithLoss, REDUCTION_MEAN);

        fprintf(g_log_file,
                "\n  ],\n"
                "  \"final\": {\"test_loss\": %.6f, \"test_acc\": null, "
                "\"test_auc\": null}\n"
                "}\n",
                (double)testLoss);
        fclose(g_log_file);

        fprintf(stdout, "FINAL test_loss=%.6f\n", (double)testLoss);
    }

    int status = 0;
    int rc = writeAllReconstructions(model, MODEL_SIZE, getTestSample, getTestSize(),
                                     "examples/ecg_anomaly_ae/outputs/c_reconstructions.npy");
    if (rc != 0) {
        fprintf(stderr, "ERROR: c_reconstructions.npy write failed (rc=%d)\n", rc);
        status = 1;
    }

    return status;
}
