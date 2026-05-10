#define SOURCE_FILE "ecg_anomaly_ae_train_c"

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
#include "Conv1dTransposed.h" /* no userApi yet — manual build below */
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "Distributions.h"
#include "InferenceApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LossFunction.h"
#include "MaxPool1d.h"
#include "NPYLoaderApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
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

/* Encoder channel widths */
#define E1_OUT 8
#define E1_K 7
#define E1_S 2
#define E2_OUT 16
#define E2_K 5

/* Decoder channel widths and kernel/strides (K=2,S=2 substitution for K=4-pad=1 spec) */
#define D1_OUT 8
#define D1_K 5
#define D1_S 5
#define D2_OUT 4
#define D2_K 2
#define D2_S 2
#define D3_OUT 1
#define D3_K 2
#define D3_S 2

/* Encoder: 2× (Conv1d + ReLU + Pool) = 6 layers
 * Decoder: 3× ConvT1d + 2× ReLU = 5 layers
 * Total = 11 */
#define MODEL_SIZE 11

/* Forward declaration; defined in Task 6. */
static void buildModel(layer_t **model);

/* ------------------------------------------------------------------------- */
/* Model parameters (file-static — must outlive buildModel).                 */
/* ------------------------------------------------------------------------- */

/* Conv1d weights: [Cout, Cin, K]. Bias: [Cout] rank-1 (matches Conv1d.c). */
static float e1_w_data[E1_OUT * IN_CHANNELS * E1_K];
static size_t e1_w_dims[3] = {E1_OUT, IN_CHANNELS, E1_K};
static float e1_b_data[E1_OUT];
static size_t e1_b_dims[1] = {E1_OUT};

static float e2_w_data[E2_OUT * E1_OUT * E2_K];
static size_t e2_w_dims[3] = {E2_OUT, E1_OUT, E2_K};
static float e2_b_data[E2_OUT];
static size_t e2_b_dims[1] = {E2_OUT};

/* Conv1dTransposed weights: [Cin, Cout/groups, K]  (note the SWAP from Conv1d).
 * Per src/layer/include/Conv1dTransposed.h:14. Bias: [Cout] rank-1. */
static float d1_w_data[E2_OUT * D1_OUT * D1_K];
static size_t d1_w_dims[3] = {E2_OUT, D1_OUT, D1_K};
static float d1_b_data[D1_OUT];
static size_t d1_b_dims[1] = {D1_OUT};

static float d2_w_data[D1_OUT * D2_OUT * D2_K];
static size_t d2_w_dims[3] = {D1_OUT, D2_OUT, D2_K};
static float d2_b_data[D2_OUT];
static size_t d2_b_dims[1] = {D2_OUT};

static float d3_w_data[D2_OUT * D3_OUT * D3_K];
static size_t d3_w_dims[3] = {D2_OUT, D3_OUT, D3_K};
static float d3_b_data[D3_OUT];
static size_t d3_b_dims[1] = {D3_OUT};

static parameter_t *buildParam(distributionType_t dist, float *data, size_t *dims, size_t ndim,
                               size_t fanIn, size_t fanOut) {
    quantization_t *q = quantizationInitFloat();
    tensor_t *p = tensorInitWithDistribution(dist, data, dims, ndim, q, NULL, fanIn, fanOut);
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}

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

/* Conv1dTransposed has no userApi yet (Phase 1 contract: paddingType_t = VALID
 * mandatory; SAME is rejected with PRINT_ERROR + exit). We mirror the manual
 * idiom from test/unit/layer/UnitTestConv1dTransposed.c, but use reserveMemory
 * so the cfg/layer survive across multiple buildModel calls (which doesn't
 * happen here, but is consistent with the rest of the file). */
static layer_t *buildConv1dTransposedLayer(parameter_t *w, parameter_t *b, size_t kSize,
                                           size_t stride, size_t outputPadding, size_t groups) {
    quantization_t *q = quantizationInitFloat();

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    initKernel(kernel, kSize, VALID, /*dilation*/ 1, stride);

    conv1dTransposedConfig_t *cfg = reserveMemory(sizeof(conv1dTransposedConfig_t));
    initConv1dTransposedConfigWithWeightsAndBias(cfg, kernel, w, b, groups, outputPadding, q, q, q,
                                                 q);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    layer->type = CONV1D_TRANSPOSED;
    lc->conv1dTransposed = cfg;
    layer->config = lc;
    return layer;
}

/* ------------------------------------------------------------------------- */
/* Datasets and dataloader thunks.                                           */
/* ------------------------------------------------------------------------- */

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

/* npyLoad strips the leading N dim, leaving each item with shape [1, 140]
 * rank-2. The C model expects rank-3 inputs [B=1, 1, 140] for Conv1d. The MSE
 * loss expects the label to have the same shape as the model output. Both
 * items AND labels are reshaped to [1, 1, 140]. */
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

/* AE: label IS the input. We re-load the same .npy file as the label tensor.
 * Two npyLoad calls produce two independent copies (no aliasing); RAM cost is
 * trivial (≤ 200 KB doubled for ECG5000). */
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

static void buildModel(layer_t **model) {
    quantization_t *q = quantizationInitFloat();

    /* ---- Encoder ---- */

    /* Block E1: Conv1d(1→8, K=7, S=2, padding=SAME), ReLU.
     * SAME with stride=2 on len 140 → len 70. */
    kernel_t *e1k = reserveMemory(sizeof(kernel_t));
    initKernel(e1k, E1_K, SAME, /*dilation*/ 1, /*stride*/ E1_S);
    parameter_t *e1_w =
        buildParam(XAVIER_UNIFORM, e1_w_data, e1_w_dims, 3, IN_CHANNELS * E1_K, E1_OUT * E1_K);
    parameter_t *e1_b = buildParam(ZEROS, e1_b_data, e1_b_dims, 1, 1, E1_OUT);
    model[0] = conv1dLayerInit(e1_w, e1_b, e1k, q, q, q, q);
    model[1] = reluLayerInit(quantizationInitFloat(), quantizationInitFloat());

    /* Block P1: MaxPool1d(K=2, S=2). 70 → 35. */
    model[2] = buildMaxPool1dLayer(/*K*/ 2, /*S*/ 2, /*outC*/ E1_OUT, /*outLen*/ 35);

    /* Block E2: Conv1d(8→16, K=5, padding=SAME), ReLU. */
    kernel_t *e2k = reserveMemory(sizeof(kernel_t));
    initKernel(e2k, E2_K, SAME, 1, 1);
    parameter_t *e2_w =
        buildParam(XAVIER_UNIFORM, e2_w_data, e2_w_dims, 3, E1_OUT * E2_K, E2_OUT * E2_K);
    parameter_t *e2_b = buildParam(ZEROS, e2_b_data, e2_b_dims, 1, 1, E2_OUT);
    model[3] = conv1dLayerInit(e2_w, e2_b, e2k, quantizationInitFloat(), quantizationInitFloat(),
                               quantizationInitFloat(), quantizationInitFloat());
    model[4] = reluLayerInit(quantizationInitFloat(), quantizationInitFloat());

    /* Block P2: AvgPool1d(K=5, S=5). 35 → 7 (bottleneck). */
    model[5] = buildAvgPool1dLayer(/*K*/ 5, /*S*/ 5);

    /* ---- Decoder ---- */

    /* Block D1: Conv1dTransposed(16→8, K=5, S=5, op=0). 7 → 35. ReLU. */
    parameter_t *d1_w =
        buildParam(XAVIER_UNIFORM, d1_w_data, d1_w_dims, 3, E2_OUT * D1_K, D1_OUT * D1_K);
    parameter_t *d1_b = buildParam(ZEROS, d1_b_data, d1_b_dims, 1, 1, D1_OUT);
    model[6] = buildConv1dTransposedLayer(d1_w, d1_b, /*K*/ D1_K, /*S*/ D1_S,
                                          /*outputPadding*/ 0, /*groups*/ 1);
    model[7] = reluLayerInit(quantizationInitFloat(), quantizationInitFloat());

    /* Block D2: Conv1dTransposed(8→4, K=2, S=2, op=0). 35 → 70. ReLU. */
    parameter_t *d2_w =
        buildParam(XAVIER_UNIFORM, d2_w_data, d2_w_dims, 3, D1_OUT * D2_K, D2_OUT * D2_K);
    parameter_t *d2_b = buildParam(ZEROS, d2_b_data, d2_b_dims, 1, 1, D2_OUT);
    model[8] = buildConv1dTransposedLayer(d2_w, d2_b, /*K*/ D2_K, /*S*/ D2_S,
                                          /*outputPadding*/ 0, /*groups*/ 1);
    model[9] = reluLayerInit(quantizationInitFloat(), quantizationInitFloat());

    /* Block D3: Conv1dTransposed(4→1, K=2, S=2, op=0). 70 → 140. NO ReLU on final. */
    parameter_t *d3_w =
        buildParam(XAVIER_UNIFORM, d3_w_data, d3_w_dims, 3, D2_OUT * D3_K, D3_OUT * D3_K);
    parameter_t *d3_b = buildParam(ZEROS, d3_b_data, d3_b_dims, 1, 1, D3_OUT);
    model[10] = buildConv1dTransposedLayer(d3_w, d3_b, /*K*/ D3_K, /*S*/ D3_S,
                                           /*outputPadding*/ 0, /*groups*/ 1);
}

/* ------------------------------------------------------------------------- */
/* Per-epoch JSON log writer + epoch callback.                               */
/* ------------------------------------------------------------------------- */

static FILE *g_log_file = NULL;
static int g_first_epoch = 1;
static struct timespec g_epoch_t0;

static void epochCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
    /* trainingRun's eval pass derives numClasses from label_num_elements (140
     * for our AE), so evalStats.accuracy / .precision / .recall / .f1 contain
     * argmax-based 140-class noise. We drop them; only evalStats.loss is
     * meaningful (it's the MSE-mean-per-element, matching PyTorch). val_acc
     * is null in the JSON to match the PyTorch side. */
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

/* Run forward inference on every sample of the given dataset, allocate a
 * single contiguous [N, 1, 140] float buffer, fill it, and write it to
 * `outPath` via npyWriteFloat32. The buffer is malloc-owned and freed by
 * this function. */
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
        (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        trainLoader, valLoader, sgd, EPOCHS, calculateGradsSequential, inferenceWithLoss,
        epochCallback);
    (void)result;

    /* Final test-set eval. Use evaluationEpoch (loss-only) to skip the
     * argmax-based metric pass that would do 140-class accuracy on this AE. */
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

    int status = 0;
    int rc = writeAllReconstructions(model, MODEL_SIZE, getTestSample, getTestSize(),
                                     "examples/ecg_anomaly_ae/outputs/c_reconstructions.npy");
    if (rc != 0) {
        fprintf(stderr, "ERROR: c_reconstructions.npy write failed (rc=%d)\n", rc);
        status = 1;
    }

    rc = writeAllReconstructions(model, MODEL_SIZE, getTrainSample, getTrainSize(),
                                 "examples/ecg_anomaly_ae/outputs/c_train_recons.npy");
    if (rc != 0) {
        fprintf(stderr, "ERROR: c_train_recons.npy write failed (rc=%d)\n", rc);
        status = 1;
    }

    return status;
}
