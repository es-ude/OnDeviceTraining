#define SOURCE_FILE "TRAIN-C-HAR-ADAMW"

/* Same model/data as train_c.c; optimizer = AdamW at PyTorch defaults,
 * CONSTANT LR (the scheduler demo lives in train_c_sym.c). */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "AdamWApi.h"
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
#include "RNG.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
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

/* 3 x (Conv1d + ReLU + Pool) + Flatten + Linear + Softmax = 12 layers */
#define MODEL_SIZE 12

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

/* Runtime config (env-overridable); defaults match the historical #defines. */
static int g_epochs = 20;
static float g_lr =
    0.001f; /* torch.optim.AdamW default; SGD's 0.01 is deliberately not carried over */
static unsigned g_seed = 42;
static unsigned g_shuffleSeed = 42;

/* torch.optim.AdamW defaults; doubles on purpose (AdamW numerics contract).
 * Deliberately NOT env-tunable: env floats would round wd through float32
 * and break twin parity of the decay scalar. */
static const double kBeta1 = 0.9, kBeta2 = 0.999, kEps = 1e-8, kWeightDecay = 0.01;

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

    g_epochs = envInt("EPOCHS", g_epochs);
    g_lr = envFloat("LR", g_lr);
    g_seed = (unsigned)envInt("SEED", (int)g_seed);
    g_shuffleSeed = (unsigned)envInt("SHUFFLE_SEED", (int)g_shuffleSeed);
    const char *logPath = getenv("LOG_PATH");

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
    optimizer_t *optim = NULL;

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
            dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                           /*shuffle*/ true, /*shuffleSeed*/ g_shuffleSeed, /*dropLast*/ true);
        dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                                 /*shuffle*/ false, /*shuffleSeed*/ 0,
                                                 /*dropLast*/ true);

#ifdef ODT_MEM_PROFILE
        markBeforeOpt = memProfileMark();
#endif
        quantization_t *momentQ = quantizationInitFloat();
        optim =
            adamWCreateOptim(g_lr, kBeta1, kBeta2, kEps, kWeightDecay, model, MODEL_SIZE, momentQ,
                             (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
#ifdef ODT_MEM_PROFILE
        markAfterOpt = memProfileMark(); /* optstate_b = delta */
#endif

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
                "  \"impl\": \"c-adamw\",\n"
                "  \"example\": \"har_classifier\",\n"
                "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
                "\"optimizer\": \"adamw\", \"weight_decay\": %.6f, \"seed\": %u, "
                "\"shuffle_seed\": %u},\n"
                "  \"epochs\": [\n",
                g_epochs, BATCH, (double)g_lr, kWeightDecay, g_seed, g_shuffleSeed);
        fflush(g_log_file);

        clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);

        trainingRunResult_t result =
            trainingRun(model, MODEL_SIZE,
                        (lossConfig_t){.funcType = CROSS_ENTROPY,
                                       .backwardReduction = REDUCTION_MEAN,
                                       .classWeights = NULL},
                        trainLoader, valLoader, optim, NULL, g_epochs, calculateGradsSequential,
                        inferenceWithLoss, epochCallback);
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
        report.dataset_b = markDataset;
        report.params_grads_b = markAfterModel - markBeforeModel;
        report.optstate_b = markAfterOpt - markBeforeOpt;
        report.params_b = memInstrumentParamBytes(optim);
        report.grads_b = memInstrumentGradBytes(optim);
        report.optstate_analytic_b = memInstrumentOptStateBytes(optim);
        report.activations_b = memInstrumentHarActivationBytes(MICRO_BATCH);
        report.io_b = memInstrumentHarIoBytes(MICRO_BATCH);
        report.pool_backward_b = memInstrumentPoolBackwardBytes(model, MODEL_SIZE);
        report.dx_peak_b = memInstrumentHarDxPeakBytes(MICRO_BATCH);

        sample_t *stepSample = getTrainSample(0);
        memStepCtx_t stepCtx = {
            .model = model,
            .modelSize = MODEL_SIZE,
            .lossConfig = (lossConfig_t){.funcType = CROSS_ENTROPY,
                                         .backwardReduction = REDUCTION_MEAN,
                                         .classWeights = NULL},
            .input = stepSample->item,
            .label = stepSample->label,
            .optim = optim,
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
