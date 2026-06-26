#define SOURCE_FILE "mnist_mlp_train_c"

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
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
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

#define EPOCHS 10
#define BATCH 64
#define LR 0.01f
#define MOMENTUM 0.9f
#define SEED 42
#define SHUFFLE_SEED 42
#define NUM_CLASSES 10

/* Flatten + Linear + ReLU + Linear + Softmax = 5 layers */
#define MODEL_SIZE 5

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

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

    tensorArray_t *valItems = npyLoad("examples/mnist_mlp/data/val_x.npy");
    tensorArray_t *valLabelsRaw = npyLoad("examples/mnist_mlp/data/val_y.npy");
    g_valDataset.items = valItems;
    g_valDataset.labels = buildOneHotLabels(valLabelsRaw);

    tensorArray_t *testItems = npyLoad("examples/mnist_mlp/data/test_x.npy");
    tensorArray_t *testLabelsRaw = npyLoad("examples/mnist_mlp/data/test_y.npy");
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
    /* Flatten [1,28,28] -> [1,784] (the channel-1 acts as batch). */
    model[0] = flattenLayerInit();
    model[1] = linearLayerInit(&(linearInit_t){.inFeatures = 28 * 28, .outFeatures = 64}, lq);
    model[2] = reluLayerInit(lq);
    model[3] = linearLayerInit(&(linearInit_t){.inFeatures = 64, .outFeatures = NUM_CLASSES}, lq);
    model[4] = softmaxLayerInit(lq);
}

/* Load PyTorch state_dict from per-layer .npy files written by
 * examples/mnist_mlp/train_pytorch.py --save-weights.
 *
 * Returns 0 on success, non-zero on first missing file. */
static int loadStateDictFromDir(layer_t **model, const char *weightsDir) {
    char wPath[256], bPath[256];
    const char *names[2] = {"fc1", "fc2"};
    tensor_t *w[2] = {0};
    tensor_t *b[2] = {0};

    for (int i = 0; i < 2; i++) {
        snprintf(wPath, sizeof(wPath), "%s/%s.weight.npy", weightsDir, names[i]);
        snprintf(bPath, sizeof(bPath), "%s/%s.bias.npy", weightsDir, names[i]);
        /* npyLoadFlat (not npyLoad): a weight file is ONE [out,in] tensor; npyLoad
         * would slice dim0 into rows and corrupt the memcpy (issue #177). */
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
        },
        2);

    for (int i = 0; i < 2; i++) {
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
    if (ensureDir("examples/mnist_mlp/logs") != 0) {
        return 1;
    }
    if (ensureDir("examples/mnist_mlp/outputs") != 0) {
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
        /* Bit-parity mode: load PyTorch state_dict, skip training, run inference. */
        const char *wDir = "examples/mnist_mlp/weights";
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

        g_log_file = fopen("examples/mnist_mlp/logs/c.json", "w");
        if (!g_log_file) {
            fprintf(stderr, "ERROR: cannot open log file for writing\n");
            return 1;
        }
        fprintf(g_log_file,
                "{\n"
                "  \"impl\": \"c\",\n"
                "  \"example\": \"mnist_mlp\",\n"
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
    int rc =
        npyWriteInt32("examples/mnist_mlp/outputs/c_predictions.npy", predictions, outShape, 1);
    if (rc != 0) {
        fprintf(stderr, "ERROR: npyWriteInt32 failed (rc=%d)\n", rc);
        status = 1;
    }
    free(predictions);

    return status;
}
