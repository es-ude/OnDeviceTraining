#define SOURCE_FILE "har_classifier_train_c"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "AvgPool1d.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "Distributions.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "MaxPool1d.h"
#include "NPYLoaderApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

#include "npy_writer.h"

#define EPOCHS 20
#define BATCH 64
#define LR 0.01f
#define MOMENTUM 0.9f
#define SEED 42
#define SHUFFLE_SEED 42
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

/* ------------------------------------------------------------------------- */
/* Datasets and dataloader thunks (mirrors example/MnistExperiment.c).       */
/* ------------------------------------------------------------------------- */

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

/* Per-sample shape after npyLoad strips the leading N dim is [9, 128] (rank-2)
 * for items and rank-0 (single int32 value) for labels. The C model expects
 * rank-3 inputs [B=1, 9, 128] for Conv1d and rank-1 one-hot float labels [6]
 * for CrossEntropy. We rebuild both at load time. */

static void reshapeItemsAddBatchDim(tensorArray_t *items) {
    /* items->array[i] currently has shape [9, 128] rank-2. Replace with
     * [1, 9, 128] rank-3. Data layout is row-major and unchanged, so we only
     * need to swap the shape header (allocate new dims/order arrays of length 3,
     * free the old ones). */
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
    /* intLabels->array[i] is a rank-0 int32 tensor (single class index 0..5).
     * We allocate a brand-new tensorArray_t whose entries are rank-1 float32
     * one-hot tensors of shape [NUM_CLASSES]. The original int32 array is
     * left intact (caller still owns it). */
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

/* ------------------------------------------------------------------------- */
/* Model parameters (file-static — must outlive buildModel).                 */
/* ------------------------------------------------------------------------- */

/* Conv1d weights: [Cout, Cin, K]. Bias: [Cout] rank-1 (matches Conv1d.c). */
static float c1_w_data[C1_OUT * IN_CHANNELS * C1_K];
static size_t c1_w_dims[3] = {C1_OUT, IN_CHANNELS, C1_K};
static float c1_b_data[C1_OUT];
static size_t c1_b_dims[1] = {C1_OUT};

static float c2_w_data[C2_OUT * C1_OUT * C2_K];
static size_t c2_w_dims[3] = {C2_OUT, C1_OUT, C2_K};
static float c2_b_data[C2_OUT];
static size_t c2_b_dims[1] = {C2_OUT};

static float c3_w_data[C3_OUT * C2_OUT * C3_K];
static size_t c3_w_dims[3] = {C3_OUT, C2_OUT, C3_K};
static float c3_b_data[C3_OUT];
static size_t c3_b_dims[1] = {C3_OUT};

/* Linear weights: [outFeat, inFeat]. Bias: [1, outFeat]. */
static float fc_w_data[NUM_CLASSES * C3_OUT];
static size_t fc_w_dims[2] = {NUM_CLASSES, C3_OUT};
static float fc_b_data[NUM_CLASSES];
static size_t fc_b_dims[2] = {1, NUM_CLASSES};

static parameter_t *buildParam(distributionType_t dist, float *data, size_t *dims, size_t ndim,
                               size_t fanIn, size_t fanOut) {
    quantization_t *q = quantizationInitFloat();
    tensor_t *p = tensorInitWithDistribution(dist, data, dims, ndim, q, NULL, fanIn, fanOut);
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}

/* MaxPool1d/AvgPool1d have no userApi; we mirror UnitTestMaxPool1d.c, but use
 * reserveMemory for backing storage (since these helpers may run more than once
 * and need addresses that survive across calls). */

static layer_t *buildMaxPool1dLayer(size_t kSize, size_t stride, size_t outC, size_t outLen) {
    quantization_t *q = quantizationInitFloat();

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    initKernel(kernel, kSize, VALID, /*dilation*/ 1, stride);

    /* Argmax buffer is sized for B=1 (training_batch iterates microbatch-by-
     * microbatch), shape [1, outC, outLen]. */
    size_t numArgmax = 1 * outC * outLen;
    int32_t *argmaxBuf = reserveMemory(numArgmax * sizeof(int32_t));
    size_t *argmaxDims = reserveMemory(3 * sizeof(size_t));
    argmaxDims[0] = 1;
    argmaxDims[1] = outC;
    argmaxDims[2] = outLen;
    tensor_t *argmax = tensorInitInt32(argmaxBuf, argmaxDims, 3, NULL);

    maxPool1dConfig_t *cfg = reserveMemory(sizeof(maxPool1dConfig_t));
    initMaxPool1dConfig(cfg, kernel, argmax, q, q);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    layer->type = MAXPOOL1D;
    lc->maxPool1d = cfg;
    layer->config = lc;
    return layer;
}

static layer_t *buildAvgPool1dLayer(size_t kSize, size_t stride) {
    quantization_t *q = quantizationInitFloat();

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    initKernel(kernel, kSize, VALID, /*dilation*/ 1, stride);

    avgPool1dConfig_t *cfg = reserveMemory(sizeof(avgPool1dConfig_t));
    initAvgPool1dConfig(cfg, kernel, q, q);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    layer->type = AVGPOOL1D;
    lc->avgPool1d = cfg;
    layer->config = lc;
    return layer;
}

static void buildModel(layer_t **model) {
    quantization_t *q1 = quantizationInitFloat();
    quantization_t *q2 = quantizationInitFloat();
    quantization_t *q3 = quantizationInitFloat();
    quantization_t *q4 = quantizationInitFloat();

    /* Block 1: Conv1d(9->16, K=7, padding=SAME), ReLU, MaxPool(K=2,S=2). */
    kernel_t *k1 = reserveMemory(sizeof(kernel_t));
    initKernel(k1, C1_K, SAME, 1, 1);
    parameter_t *c1_w =
        buildParam(XAVIER_UNIFORM, c1_w_data, c1_w_dims, 3, IN_CHANNELS * C1_K, C1_OUT * C1_K);
    parameter_t *c1_b = buildParam(ZEROS, c1_b_data, c1_b_dims, 1, 1, C1_OUT);
    model[0] = conv1dLayerInit(c1_w, c1_b, k1, q1, q2, q3, q4);
    model[1] = reluLayerInit(quantizationInitFloat(), quantizationInitFloat());
    model[2] = buildMaxPool1dLayer(2, 2, C1_OUT, LEN_INPUT / 2);

    /* Block 2: Conv1d(16->32, K=5, padding=SAME), ReLU, MaxPool(K=2,S=2). */
    kernel_t *k2 = reserveMemory(sizeof(kernel_t));
    initKernel(k2, C2_K, SAME, 1, 1);
    parameter_t *c2_w =
        buildParam(XAVIER_UNIFORM, c2_w_data, c2_w_dims, 3, C1_OUT * C2_K, C2_OUT * C2_K);
    parameter_t *c2_b = buildParam(ZEROS, c2_b_data, c2_b_dims, 1, 1, C2_OUT);
    model[3] = conv1dLayerInit(c2_w, c2_b, k2, quantizationInitFloat(), quantizationInitFloat(),
                               quantizationInitFloat(), quantizationInitFloat());
    model[4] = reluLayerInit(quantizationInitFloat(), quantizationInitFloat());
    model[5] = buildMaxPool1dLayer(2, 2, C2_OUT, LEN_INPUT / 4);

    /* Block 3: Conv1d(32->64, K=3, padding=SAME), ReLU, AvgPool(K=32,S=32). */
    kernel_t *k3 = reserveMemory(sizeof(kernel_t));
    initKernel(k3, C3_K, SAME, 1, 1);
    parameter_t *c3_w =
        buildParam(XAVIER_UNIFORM, c3_w_data, c3_w_dims, 3, C2_OUT * C3_K, C3_OUT * C3_K);
    parameter_t *c3_b = buildParam(ZEROS, c3_b_data, c3_b_dims, 1, 1, C3_OUT);
    model[6] = conv1dLayerInit(c3_w, c3_b, k3, quantizationInitFloat(), quantizationInitFloat(),
                               quantizationInitFloat(), quantizationInitFloat());
    model[7] = reluLayerInit(quantizationInitFloat(), quantizationInitFloat());
    model[8] = buildAvgPool1dLayer(LEN_INPUT / 4, LEN_INPUT / 4);

    /* Head: Flatten, Linear(64 -> 6), Softmax. */
    model[9] = flattenLayerInit();
    parameter_t *fc_w = buildParam(XAVIER_UNIFORM, fc_w_data, fc_w_dims, 2, C3_OUT, NUM_CLASSES);
    parameter_t *fc_b = buildParam(ZEROS, fc_b_data, fc_b_dims, 2, 1, NUM_CLASSES);
    model[10] = linearLayerInit(fc_w, fc_b, quantizationInitFloat(), quantizationInitFloat(),
                                quantizationInitFloat(), quantizationInitFloat());
    model[11] = softmaxLayerInit(quantizationInitFloat(), quantizationInitFloat());
}

/* ------------------------------------------------------------------------- */
/* Per-epoch JSON log writer + epoch callback.                               */
/* ------------------------------------------------------------------------- */

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
    if (ensureDir("examples/har_classifier/logs") != 0) {
        return 1;
    }
    if (ensureDir("examples/har_classifier/outputs") != 0) {
        return 1;
    }

    initDataSets();

    dataLoader_t *trainLoader = dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                                               /*shuffle*/ true, /*shuffleSeed*/ SHUFFLE_SEED,
                                               /*dropLast*/ true);
    dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                             /*shuffle*/ false, /*shuffleSeed*/ 0,
                                             /*dropLast*/ true);
    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, /*shuffleSeed*/ 0,
                                              /*dropLast*/ true);

    layer_t *model[MODEL_SIZE];
    buildModel(model);

    optimizer_t *sgd =
        sgdMCreateOptim(LR, MOMENTUM, /*weightDecay*/ 0.0f, model, MODEL_SIZE, FLOAT32);

    g_log_file = fopen("examples/har_classifier/logs/c.json", "w");
    if (!g_log_file) {
        fprintf(stderr, "ERROR: cannot open log file for writing\n");
        return 1;
    }
    fprintf(g_log_file,
            "{\n"
            "  \"impl\": \"c\",\n"
            "  \"example\": \"har_classifier\",\n"
            "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
            "\"momentum\": %.6f, \"seed\": %d, \"shuffle_seed\": %d},\n"
            "  \"epochs\": [\n",
            EPOCHS, BATCH, (double)LR, (double)MOMENTUM, SEED, SHUFFLE_SEED);
    fflush(g_log_file);

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);

    trainingRunResult_t result = trainingRun(
        model, MODEL_SIZE,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        trainLoader, valLoader, sgd, EPOCHS, calculateGradsSequential, inferenceWithLoss,
        epochCallback);
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

    /* Predictions: run inference on every test sample, write argmax to .npy.
     * inference() returns a fresh tensor we own; freeing every iteration via
     * freeTensor would also free its data buffer, which is what we want. */
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

    return status;
}
