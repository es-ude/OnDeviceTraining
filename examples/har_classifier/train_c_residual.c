#define SOURCE_FILE "har_classifier_train_c_residual"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "AddApi.h"
#include "BatchView.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Model.h"
#include "ModelGraphApi.h"
#include "NPYLoaderApi.h"
#include "Pool1dApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "RematScheduler.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

#include "npy_writer.h"
#include "remat_select.h"

#define NUM_CLASSES 6
#define IN_CHANNELS 9
#define LEN_INPUT 128
#define CHANNELS 16
#define C1_K 7
#define C23_K 5

/* SKIP=1: conv1, relu1, pool1, conv2, relu2, conv3, ADD(pool1, conv3), relu3, gap, flatten, fc,
 * softmax. SKIP=0 drops the ADD: the chain twin the memory report compares against. */
#define MAX_NODES 12
#define MAX_EDGES 13

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

static int g_epochs = 20;
static const float g_lr = 0.01f;
static const float g_momentum = 0.9f;
static unsigned g_seed = 42;
static const int g_batchSize = 64;

static int envInt(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? (int)strtol(v, NULL, 10) : dflt;
}

static bool envSet(const char *name) {
    const char *v = getenv(name);
    return v != NULL && v[0] != '\0';
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
    tensorArray_t *trainItems = npyLoad("examples/har_classifier/data/train_x.npy");
    tensorArray_t *trainLabelsRaw = npyLoad("examples/har_classifier/data/train_y.npy");
    g_trainDataset.items = trainItems;
    g_trainDataset.labels = buildOneHotLabels(trainLabelsRaw);

    tensorArray_t *valItems = npyLoad("examples/har_classifier/data/val_x.npy");
    tensorArray_t *valLabelsRaw = npyLoad("examples/har_classifier/data/val_y.npy");
    g_valDataset.items = valItems;
    g_valDataset.labels = buildOneHotLabels(valLabelsRaw);

    tensorArray_t *testItems = npyLoad("examples/har_classifier/data/test_x.npy");
    tensorArray_t *testLabelsRaw = npyLoad("examples/har_classifier/data/test_y.npy");
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

/* The model points into these arrays, so they live for the whole run. */
static layer_t *g_layers[MAX_NODES];
static uint16_t g_inputStart[MAX_NODES + 1];
static uint16_t g_inputIds[MAX_EDGES];

static layer_t *conv(size_t in, size_t k, const trainingRng_t *rng, size_t node, layerQuant_t *lq) {
    const rngKey_t key = trainingRngInitKey(rng, node);
    return conv1dLayerInit(&(conv1dInit_t){.inChannels = in,
                                           .outChannels = CHANNELS,
                                           .kernelSize = k,
                                           .padding = SAME,
                                           .weightInit.key = &key},
                           lq);
}

/* Each weight init draws the key of its node index, so the two topologies differ only in fc's key.
 */
static model_t buildModel(bool skip, layerQuant_t *lq, const trainingRng_t *rng) {
    graphBuilder_t g;
    graphBegin(&g, g_layers, g_inputStart, g_inputIds, MAX_NODES, MAX_EDGES);
    uint16_t h = graphNode(&g, conv(IN_CHANNELS, C1_K, rng, 0, lq), GRAPH_INPUT);
    h = graphNode(&g, reluLayerInit(lq), h);
    const uint16_t pooled = graphNode(
        &g,
        maxPool1dLayerInit(
            &(maxPool1dInit_t){
                .kernelSize = 2, .stride = 2, .inputChannels = CHANNELS, .inputLength = LEN_INPUT},
            lq),
        h);
    h = graphNode(&g, conv(CHANNELS, C23_K, rng, 3, lq), pooled);
    h = graphNode(&g, reluLayerInit(lq), h);
    h = graphNode(&g, conv(CHANNELS, C23_K, rng, 5, lq), h);
    if (skip) {
        /* pool1 first, conv3 second: the order of the twin's x + y. */
        h = graphMerge(&g, addLayerInit(NULL), pooled, h);
    }
    h = graphNode(&g, reluLayerInit(lq), h);
    h = graphNode(&g,
                  avgPool1dLayerInit(
                      &(avgPool1dInit_t){.kernelSize = LEN_INPUT / 2, .stride = LEN_INPUT / 2}, lq),
                  h);
    h = graphNode(&g, flattenLayerInit(), h);
    const rngKey_t fcKey = trainingRngInitKey(rng, skip ? 10u : 9u);
    h = graphNode(&g,
                  linearLayerInit(&(linearInit_t){.inFeatures = CHANNELS,
                                                  .outFeatures = NUM_CLASSES,
                                                  .weightInit.key = &fcKey},
                                  lq),
                  h);
    (void)graphNode(&g, softmaxLayerInit(lq), h);
    return graphFinish(&g);
}

/* Loads the weights train_pytorch_residual.py exports, in the model's parameter-layer order.
 * Returns 0 on success, non-zero on the first missing file. */
static int loadStateDictFromDir(const model_t *model, const char *weightsDir) {
    char wPath[256], bPath[256];
    const char *names[4] = {"conv1", "conv2", "conv3", "fc"};
    tensor_t *w[4] = {0};
    tensor_t *b[4] = {0};

    for (int i = 0; i < 4; i++) {
        snprintf(wPath, sizeof(wPath), "%s/%s.weight.npy", weightsDir, names[i]);
        snprintf(bPath, sizeof(bPath), "%s/%s.bias.npy", weightsDir, names[i]);
        /* npyLoadFlat, not npyLoad: npyLoad slices dim 0 into row tensors, and the copy into the
         * layer would read past the first row. */
        w[i] = npyLoadFlat(wPath);
        b[i] = npyLoadFlat(bPath);
        if (w[i] == NULL || b[i] == NULL) {
            fprintf(stderr, "loadStateDictFromDir: missing %s or %s\n", wPath, bPath);
            return 1;
        }
    }

    modelLoadStateDict(
        model,
        (stateDictEntry_t[]){
            {.name = names[0], .weightData = (float *)w[0]->data, .biasData = (float *)b[0]->data},
            {.name = names[1], .weightData = (float *)w[1]->data, .biasData = (float *)b[1]->data},
            {.name = names[2], .weightData = (float *)w[2]->data, .biasData = (float *)b[2]->data},
            {.name = names[3], .weightData = (float *)w[3]->data, .biasData = (float *)b[3]->data},
        },
        4);

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
    g_seed = (unsigned)envInt("SEED", (int)g_seed);
    const int skipEnv = envInt("SKIP", 1);
    if (skipEnv != 0 && skipEnv != 1) {
        fprintf(stderr, "ERROR: SKIP=%d (expected 0 or 1)\n", skipEnv);
        return 1;
    }
    const bool skip = skipEnv == 1;
    const bool bitParity = envSet("BIT_PARITY");
    if (bitParity && !skip) {
        fprintf(stderr, "BIT_PARITY needs the residual model (SKIP=1)\n");
        return 1;
    }
    const size_t numNodes = skip ? MAX_NODES : MAX_NODES - 1;

    rematSelection_t rematSel;
    if (!rematSelectFromEnv(&rematSel, numNodes)) {
        return 1;
    }
    /* A BIT_PARITY run trains nothing, so a chosen memory scheme cannot apply. */
    if (rematSel.chosen && bitParity) {
        fprintf(stderr, "ERROR: REMAT_STORAGE, REMAT_PLAN, DRY_PLAN, DUMP_PROBLEM and "
                        "REMAT_BUDGET_B do not apply to a BIT_PARITY run (it trains nothing)\n");
        return 1;
    }
    const char *example = skip ? "har_classifier_residual" : "har_classifier_residual_chain";
    const lossConfig_t lossConfig = {
        .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL};
    /* Zeroed, so the final deinit is a no-op when no scheduler was built. */
    rematScheduler_t remat = {0};

    initDataSets();

    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false,
                                              /*dropLast*/ true);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    /* BIT_PARITY overwrites the seeded init weights from the state_dict. */
    trainingRng_t rng;
    trainingRngInit(&rng, g_seed);
    const model_t model = buildModel(skip, &lq, &rng);

    if (bitParity) {
        const char *wDir = "examples/har_classifier/weights_residual";
        if (loadStateDictFromDir(&model, wDir) != 0) {
            fprintf(stderr, "BIT_PARITY: state_dict load failed\n");
            return 1;
        }
        fprintf(stdout, "BIT_PARITY: loaded state_dict from %s\n", wDir);
    } else {
        if (rematSel.chosen) {
            /* Keyed to one training sample's [1, ...] view: every training call runs at
             * micro-batch 1. */
            sample_t *keySample = getTrainSample(0);
            batchView_t keyView;
            const tensor_t *inputLike = batchViewOf(&keyView, keySample->item);
            if (rematSel.dumpProblem) {
                bool dumped = rematSelectDumpProblem(stdout, stderr, &model, lossConfig, inputLike);
                freeSample(keySample);
                return dumped ? 0 : 1;
            }
            bool built = rematSelectInit(&remat, &rematSel, example, &model, lossConfig, inputLike);
            freeSample(keySample);
            if (!built) {
                return 1;
            }
            if (rematSel.dryPlan) {
                rematSelectPrintPlan(stdout, example, &rematSel, &remat);
                rematSchedulerDeinit(&remat);
                return 0;
            }
        }
        dataLoader_t *trainLoader =
            dataLoaderInit(getTrainSample, getTrainSize, (uint16_t)g_batchSize, NULL, NULL,
                           /*shuffle*/ true, /*dropLast*/ true);
        dataLoaderSetReshufflePerEpoch(trainLoader, true);
        dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                                 /*shuffle*/ false,
                                                 /*dropLast*/ true);

        quantization_t *momentumQ = quantizationInitFloat();
        optimizer_t *sgd =
            sgdMCreateOptim(g_lr, g_momentum, /*weightDecay*/ 0.0f, &model, momentumQ,
                            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
        freeQuantization(momentumQ);

        trainingRunOptions_t options = {.callback = epochCallback,
                                        .remat = rematSel.chosen ? &remat : NULL,
                                        .rng = &rng,
                                        .stopOnNonFiniteLoss = true};

        const char *logPath = getenv("LOG_PATH");
        const char *outLog = (logPath != NULL && logPath[0] != '\0') ? logPath
                             : skip ? "examples/har_classifier/logs/c_residual.json"
                                    : "examples/har_classifier/logs/c_residual_chain.json";
        g_log_file = fopen(outLog, "w");
        if (!g_log_file) {
            fprintf(stderr, "ERROR: cannot open log file for writing\n");
            return 1;
        }
        fprintf(g_log_file,
                "{\n"
                "  \"impl\": \"c\",\n"
                "  \"example\": \"%s\",\n"
                "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
                "\"momentum\": %.6f, \"seed\": %u, \"lr_schedule\": \"none\", "
                "\"optimizer\": \"sgd\", \"reshuffle\": 1",
                example, g_epochs, g_batchSize, (double)g_lr, (double)g_momentum, g_seed);
        rematSelectPrintConfigKeys(g_log_file, &rematSel);
        fprintf(g_log_file,
                ", \"toolchain\": \"%s\"},\n"
                "  \"epochs\": [\n",
                __VERSION__);
        fflush(g_log_file);

        clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
        trainingRunResult_t result =
            trainingRun(&model, lossConfig, trainLoader, valLoader, sgd, (size_t)g_epochs,
                        calculateGradsSequential, inferenceWithLoss, &options);
        epochStats_t testStats = evaluationEpochWithMetrics(&model, CROSS_ENTROPY, testLoader,
                                                            inferenceWithLoss, REDUCTION_MEAN, 0);

        fprintf(g_log_file,
                "\n  ],\n"
                "  \"final\": {\"test_loss\": %.6f, \"test_acc\": %.6f, \"test_auc\": null, "
                "\"diverged\": %d, \"epochs_completed\": %zu}\n}\n",
                (double)testStats.loss, (double)testStats.accuracy,
                result.stoppedOnNonFiniteLoss ? 1 : 0, result.epochsCompleted);
        fclose(g_log_file);
        fprintf(stdout, "FINAL test_loss=%.4f test_acc=%.4f\n", (double)testStats.loss,
                (double)testStats.accuracy);
    }

    size_t numTest = getTestSize();
    int32_t *predictions = malloc(numTest * sizeof(int32_t));
    if (!predictions) {
        fprintf(stderr, "OOM allocating predictions\n");
        return 1;
    }
    for (size_t i = 0; i < numTest; ++i) {
        sample_t *s = getTestSample(i);
        batchView_t itemView;
        tensor_t *out = inference(&model, batchViewOf(&itemView, s->item));
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
    const char *predPath = skip
                               ? "examples/har_classifier/outputs/c_residual_predictions.npy"
                               : "examples/har_classifier/outputs/c_residual_chain_predictions.npy";
    int rc = npyWriteInt32(predPath, predictions, outShape, 1);
    if (rc != 0) {
        fprintf(stderr, "ERROR: npyWriteInt32 failed (rc=%d)\n", rc);
        status = 1;
    }
    free(predictions);
    rematSchedulerDeinit(&remat);
    return status;
}
