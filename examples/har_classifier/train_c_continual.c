#define SOURCE_FILE "har_classifier_train_c_continual"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "ArithmeticType.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "ExemplarBuffer.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "NPYLoaderApi.h"
#include "Pool1dApi.h"
#include "PpcaReplay.h"
#include "PpcaReplayApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingEpochDefault.h"
#include "TrainingLoopApi.h"

#define BATCH 64 /* macro-batch: loader groups 64 samples per optimizer step */
#define NUM_CLASSES 6

#define IN_CHANNELS 9
#define LEN_INPUT 128
#define D_FEATURES (IN_CHANNELS * LEN_INPUT) /* 1152 = flat per-sample dim */

#define C1_OUT 16
#define C1_K 7
#define C2_OUT 32
#define C2_K 5
#define C3_OUT 64
#define C3_K 3

/* 3 x (Conv1d + ReLU + Pool) + Flatten + Linear + Softmax = 12 layers */
#define MODEL_SIZE 12

/* The fixed getSample/getSize signatures carry no context, so the active
 * domain's train/eval dataset is selected through these module globals: point
 * them at the desired dataset_t before each dataLoaderInit, keep them valid
 * for that loader's whole lifetime. */
static dataset_t *g_curTrain = NULL;
static dataset_t *g_curEval = NULL;

/* Runtime config (env-overridable). */
static int g_domains = 5;
static int g_pretrainEpochs = 10;
static int g_epochsPerDomain = 5;
static float g_lr = 0.01f;
static float g_momentum = 0.9f;
static unsigned g_seed = 42;
static int g_replay = 1;
static replayMode_t g_replayMode = REPLAY_MODE_PPCA_SAMPLE;
static int g_rPerClass = 2;
static int g_rank = 8;
static int g_minCount = 16;
static int g_maxSessionSamples = 512;
/* REPLAY_MODE=exemplar buffer capacity; default 9 = iso-memory with the
 * rank-8 PPCA generator (9 * 4608 B raw windows = 41,472 B vs 41,592 B). */
static int g_exemplarsPerClass = 9;

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

/* Load domain{t}_{split}_{x,y}.npy into ds (same helpers/dtype as train_c.c:
 * items get a batch dim, int32 labels become FLOAT32 one-hot). */
static void loadDomainDataset(int t, const char *split, dataset_t *ds) {
    char xPath[256], yPath[256];
    snprintf(xPath, sizeof(xPath), "examples/har_classifier/data/domains/domain%d_%s_x.npy", t,
             split);
    snprintf(yPath, sizeof(yPath), "examples/har_classifier/data/domains/domain%d_%s_y.npy", t,
             split);
    tensorArray_t *items = npyLoad(xPath);
    tensorArray_t *labelsRaw = npyLoad(yPath);
    if (items == NULL || labelsRaw == NULL) {
        fprintf(stderr, "ERROR: cannot load %s / %s\n", xPath, yPath);
        exit(1);
    }
    reshapeItemsAddBatchDim(items);
    ds->items = items;
    ds->labels = buildOneHotLabels(labelsRaw);
    freeTensorArray(labelsRaw); /* one-hots copied the class; raw ints no longer needed */
}

static sample_t *getTrainSample(size_t id) {
    return npyGetSample(g_curTrain, id);
}
static size_t getTrainSize(void) {
    return g_curTrain->items->size;
}
static sample_t *getEvalSample(size_t id) {
    return npyGetSample(g_curEval, id);
}
static size_t getEvalSize(void) {
    return g_curEval->items->size;
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

/* Borrow `stage` as a FLOAT32 [m,dim] tensor; release frees ONLY the
 * wrappers, never the staged data. */
static tensor_t *wrapStageAsTensor(float *stage, size_t m, size_t dim) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    size_t *order = reserveMemory(2 * sizeof(size_t));
    dims[0] = m;
    dims[1] = dim;
    order[0] = 0;
    order[1] = 1;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 2;
    tensor_t *t = reserveMemory(sizeof(tensor_t));
    t->data = (uint8_t *)stage;
    t->shape = shape;
    t->quantization = quantizationInitFloat();
    t->sparsity = NULL;
    return t;
}

static void releaseStageTensor(tensor_t *t) {
    freeQuantization(t->quantization);
    freeReservedMemory(t->shape->dimensions);
    freeReservedMemory(t->shape->orderOfDimensions);
    freeReservedMemory(t->shape);
    freeReservedMemory(t);
}

/* Group a domain's train items per class and merge each class's samples
 * into its generator, in chunks of <= maxSessionSamples. Items are [1,9,128]
 * dataset tensors; the flat view [chunk, 1152] borrows their data. */
static void absorbDomain(ppcaReplaySet_t *set, dataset_t *domainTrain) {
    size_t n = domainTrain->items->size;
    size_t dim = set->generators[0]->dim;
    size_t maxM = set->workspace->maxSessionSamples;
    float *stage = reserveMemory(maxM * dim * sizeof(float));
    for (size_t c = 0; c < set->numClasses; c++) {
        size_t filled = 0;
        for (size_t i = 0; i < n; i++) {
            float *label = (float *)domainTrain->labels->array[i]->data;
            size_t cls = 0;
            for (size_t j = 1; j < set->numClasses; j++) {
                if (label[j] > label[cls]) {
                    cls = j;
                }
            }
            if (cls != c) {
                continue;
            }
            float *item = (float *)domainTrain->items->array[i]->data;
            for (size_t j = 0; j < dim; j++) {
                stage[filled * dim + j] = item[j];
            }
            filled++;
            if (filled == maxM) {
                tensor_t *chunk = wrapStageAsTensor(stage, filled, dim);
                ppcaReplayUpdate(set->generators[c], chunk, set->workspace);
                releaseStageTensor(chunk);
                filled = 0;
            }
        }
        if (filled > 0) {
            tensor_t *chunk = wrapStageAsTensor(stage, filled, dim);
            ppcaReplayUpdate(set->generators[c], chunk, set->workspace);
            releaseStageTensor(chunk);
        }
    }
    freeReservedMemory(stage);
}

/* Fine-tune the model for `epochs` epochs on the given (possibly replay-wrapped)
 * loader; prints one line per epoch. */
static void trainEpochs(layer_t **model, optimizer_t *sgd, dataLoader_t *loader, int epochs,
                        int domain, const char *phase) {
    for (int e = 0; e < epochs; e++) {
        float loss = trainingEpochDefault(model, MODEL_SIZE,
                                          (lossConfig_t){.funcType = CROSS_ENTROPY,
                                                         .backwardReduction = REDUCTION_MEAN,
                                                         .classWeights = NULL},
                                          loader, sgd, calculateGradsSequential, REDUCTION_MEAN);
        fprintf(stdout, "[domain %d %s] epoch %d/%d train_loss=%.4f\n", domain, phase, e + 1,
                epochs, (double)loss);
        fflush(stdout);
    }
}

/* Evaluate the model on domain `j`'s eval set; returns top-1 accuracy. */
static float evalDomain(layer_t **model, dataset_t *evalSet) {
    g_curEval = evalSet;
    dataLoader_t *loader = dataLoaderInit(getEvalSample, getEvalSize, 1, NULL, NULL,
                                          /*shuffle*/ false, /*shuffleSeed*/ 0, /*dropLast*/ true);
    epochStats_t stats = evaluationEpochWithMetrics(model, MODEL_SIZE, CROSS_ENTROPY, loader,
                                                    inferenceWithLoss, REDUCTION_MEAN);
    freeDataLoader(loader);
    return stats.accuracy;
}

/* First-K semantics run over the GLOBAL sample stream: called after every
 * domain (mirroring absorbDomain's sequencing), but the buffer drops adds
 * once a class is full — in practice it freezes to the first
 * EXEMPLARS_PER_CLASS samples per class of domain 0. */
static void exemplarAbsorbDomain(exemplarBuffer_t *buf, dataset_t *domainTrain) {
    size_t n = domainTrain->items->size;
    for (size_t i = 0; i < n; i++) {
        float *label = (float *)domainTrain->labels->array[i]->data;
        size_t cls = 0;
        for (size_t j = 1; j < buf->numClasses; j++) {
            if (label[j] > label[cls]) {
                cls = j;
            }
        }
        exemplarBufferAdd(buf, domainTrain->items->array[i], cls);
    }
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

    g_seed = (unsigned)envInt("SEED", (int)g_seed);
    g_domains = envInt("DOMAINS", g_domains);
    g_pretrainEpochs = envInt("PRETRAIN_EPOCHS", g_pretrainEpochs);
    g_epochsPerDomain = envInt("EPOCHS_PER_DOMAIN", g_epochsPerDomain);
    g_lr = envFloat("LR", g_lr);
    g_momentum = envFloat("MOMENTUM", g_momentum);
    g_replay = envInt("REPLAY", g_replay);
    const char *modeStr = getenv("REPLAY_MODE");
    if (modeStr != NULL && modeStr[0] != '\0') {
        if (strcmp(modeStr, "mean") == 0) {
            g_replayMode = REPLAY_MODE_CLASS_MEAN;
        } else if (strcmp(modeStr, "exemplar") == 0) {
            g_replayMode = REPLAY_MODE_EXEMPLAR;
        } else if (strcmp(modeStr, "ppca") != 0) {
            fprintf(stderr, "ERROR: REPLAY_MODE must be 'ppca', 'mean' or 'exemplar' (got '%s')\n",
                    modeStr);
            return 1;
        }
    }
    g_exemplarsPerClass = envInt("EXEMPLARS_PER_CLASS", g_exemplarsPerClass);
    g_rPerClass = envInt("R_PER_CLASS", g_rPerClass);
    g_rank = envInt("RANK", g_rank);
    g_minCount = envInt("MIN_COUNT", g_minCount);
    g_maxSessionSamples = envInt("MAX_SESSION_SAMPLES", g_maxSessionSamples);
    const char *logPath = getenv("LOG_PATH");
    const char *outLog = (logPath != NULL && logPath[0] != '\0')
                             ? logPath
                             : "examples/har_classifier/logs/continual.json";

    if (g_domains < 1) {
        fprintf(stderr, "ERROR: DOMAINS must be >= 1 (got %d)\n", g_domains);
        return 1;
    }
    size_t T = (size_t)g_domains;

    /* Keep all T domains' datasets resident (host demo). */
    dataset_t *trainSets = reserveMemory(T * sizeof(dataset_t));
    dataset_t *evalSets = reserveMemory(T * sizeof(dataset_t));
    for (size_t t = 0; t < T; t++) {
        loadDomainDataset((int)t, "train", &trainSets[t]);
        loadDomainDataset((int)t, "eval", &evalSets[t]);
    }

    /* Model: identical topology + seed handling as train_c.c. */
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    rngSetSeed(g_seed);
    layer_t *model[MODEL_SIZE];
    buildModel(model, &lq);

    /* FLOAT32 momentum accumulator; optimizer clones the config per state via
     * getQLike and does NOT take ownership. One optimizer spans all domains so
     * momentum carries across the sequential protocol. */
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(g_lr, g_momentum, /*weightDecay*/ 0.0f, model, MODEL_SIZE, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* Replay source (only when REPLAY=1): PPCA set for ppca/mean modes, raw
     * exemplar buffer for exemplar mode (no generators, no workspace). */
    ppcaReplaySet_t *set = NULL;
    exemplarBuffer_t *exemplars = NULL;
    rng32_t replayStream = {.state = g_seed * 2654435761u | 1u};
    if (g_replay && g_replayMode == REPLAY_MODE_EXEMPLAR) {
        exemplars = exemplarBufferCreate(NUM_CLASSES, (size_t)g_exemplarsPerClass);
    } else if (g_replay) {
        quantization_t *meanQ = quantizationInitFloat();
        quantization_t *basisQ = quantizationInitFloat();
        quantization_t *eigvalsQ = quantizationInitFloat();
        ppcaReplayConfig_t cfg = {
            .dim = D_FEATURES,
            .rank = (size_t)g_rank,
            .maxSessionSamples = (size_t)g_maxSessionSamples,
            .mergeMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .streamMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .sampleMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .meanQ = meanQ,
            .basisQ = basisQ,
            .eigvalsQ = eigvalsQ,
            .sigma2Floor = 1e-6f,
            .shrinkageGamma = 0.0f,
        };
        set = ppcaReplaySetCreate(NUM_CLASSES, &cfg);
        /* create cloned the storage configs via getQLike; release ours. */
        freeQuantization(meanQ);
        freeQuantization(basisQ);
        freeQuantization(eigvalsQ);
    }

    /* Accuracy matrix R[t][j] = accuracy on domain j after training through
     * domain t (row t has t+1 valid entries). */
    float *R = reserveMemory(T * T * sizeof(float));

    size_t bytesAfterDomain0 = 0;
    size_t bytesFinal = 0;

    /* Protocol (0-indexed, spec section 9). Domain 0: pretrain. */
    g_curTrain = &trainSets[0];
    dataLoader_t *loader0 =
        dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                       /*shuffle*/ true, /*shuffleSeed*/ g_seed, /*dropLast*/ true);
    trainEpochs(model, sgd, loader0, g_pretrainEpochs, 0, "pretrain");
    freeDataLoader(loader0);

    if (g_replay && exemplars != NULL) {
        exemplarAbsorbDomain(exemplars, &trainSets[0]);
    } else if (g_replay) {
        absorbDomain(set, &trainSets[0]);
        bytesAfterDomain0 = ppcaReplayBytes(set->generators[0]);
        bytesFinal = bytesAfterDomain0; /* holds if T == 1 */
    }

    R[0] = evalDomain(model, &evalSets[0]);
    fprintf(stdout, "R[0][0] = %.4f\n", (double)R[0]);

    /* Domains 1..T-1: fine-tune (optionally replay-wrapped), eval, absorb. */
    for (size_t t = 1; t < T; t++) {
        g_curTrain = &trainSets[t];
        dataLoader_t *base =
            dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                           /*shuffle*/ true, /*shuffleSeed*/ g_seed, /*dropLast*/ true);
        dataLoader_t *loader = base;
        if (g_replay) {
            loader = replayDataLoaderWrap(
                base, &(replayLoaderConfig_t){.set = set,
                                              .exemplars = exemplars,
                                              .samplesPerClass = (size_t)g_rPerClass,
                                              .minCount = (uint32_t)g_minCount,
                                              .stream = &replayStream,
                                              .mode = g_replayMode});
        }
        trainEpochs(model, sgd, loader, g_epochsPerDomain, (int)t, "finetune");
        /* Free the WRAPPER with freeReplayDataLoader and the base with
         * freeDataLoader — never freeDataLoader on the wrapper. */
        if (g_replay) {
            freeReplayDataLoader(loader);
        }
        freeDataLoader(base);

        for (size_t j = 0; j <= t; j++) {
            R[t * T + j] = evalDomain(model, &evalSets[j]);
            fprintf(stdout, "R[%zu][%zu] = %.4f\n", t, j, (double)R[t * T + j]);
        }

        if (g_replay && exemplars != NULL) {
            exemplarAbsorbDomain(exemplars, &trainSets[t]); /* no-op once full */
        } else if (g_replay) {
            absorbDomain(set, &trainSets[t]);
            bytesFinal = ppcaReplayBytes(set->generators[0]);
        }
    }

    /* Constant-footprint machinery check: per-class generator bytes after the
     * final domain must equal those after domain 0 (exemplar buffer is frozen
     * at capacity by construction — nothing to check). */
    if (g_replay && set != NULL && bytesFinal != bytesAfterDomain0) {
        fprintf(stderr,
                "ERROR: replay footprint not constant: %zu bytes/class after domain 0 vs "
                "%zu after final domain\n",
                bytesAfterDomain0, bytesFinal);
        return 1;
    }

    /* Emit the run log. */
    FILE *log = fopen(outLog, "w");
    if (!log) {
        fprintf(stderr, "ERROR: cannot open log file %s for writing\n", outLog);
        return 1;
    }
    fprintf(log,
            "{\n"
            "  \"impl\": \"c\",\n"
            "  \"example\": \"har_continual\",\n"
            "  \"config\": {\"seed\": %u, \"domains\": %d, \"pretrain_epochs\": %d, "
            "\"epochs_per_domain\": %d, \"lr\": %.6f, \"momentum\": %.6f, \"replay\": %d, "
            "\"r_per_class\": %d, \"rank\": %d, \"min_count\": %d, \"max_session_samples\": %d},\n",
            g_seed, g_domains, g_pretrainEpochs, g_epochsPerDomain, (double)g_lr,
            (double)g_momentum, g_replay, g_rPerClass, g_rank, g_minCount, g_maxSessionSamples);

    fprintf(log, "  \"accuracy_matrix\": [\n");
    for (size_t t = 0; t < T; t++) {
        fprintf(log, "    [");
        for (size_t j = 0; j <= t; j++) {
            fprintf(log, "%.6f%s", (double)R[t * T + j], (j < t) ? ", " : "");
        }
        fprintf(log, "]%s\n", (t + 1 < T) ? "," : "");
    }
    fprintf(log, "  ],\n");

    fprintf(log, "  \"replay\": {");
    if (g_replay && exemplars != NULL) {
        fprintf(log,
                "\"enabled\": 1, \"mode\": \"exemplar\", \"exemplars_per_class\": %d, "
                "\"exemplar_state_bytes_per_class\": %zu, \"footprint_constant\": true",
                g_exemplarsPerClass,
                (size_t)g_exemplarsPerClass * D_FEATURES * sizeof(float) + sizeof(uint32_t));
    } else if (g_replay) {
        size_t wsBytes =
            ppcaWorkspaceBytes(D_FEATURES, (size_t)g_rank, (size_t)g_maxSessionSamples);
        size_t isoExemplars =
            ppcaReplayIsoExemplarCount(set->generators[0], (size_t)D_FEATURES * sizeof(float));
        fprintf(log,
                "\"enabled\": 1, \"mode\": \"%s\", \"bytes_per_class\": %zu, "
                "\"workspace_bytes\": %zu, \"iso_exemplar_count\": %zu, "
                "\"footprint_constant\": true",
                g_replayMode == REPLAY_MODE_CLASS_MEAN ? "mean" : "ppca", bytesFinal, wsBytes,
                isoExemplars);
        if (g_replayMode == REPLAY_MODE_CLASS_MEAN) {
            /* What a purpose-built centroid buffer would store (the demo
             * reuses the PPCA set, so bytes_per_class above overstates it). */
            fprintf(log, ", \"mean_state_bytes_per_class\": %zu",
                    (size_t)D_FEATURES * sizeof(float) + sizeof(uint32_t));
        }
    } else {
        fprintf(log, "\"enabled\": 0");
    }
    fprintf(log, "}\n}\n");
    fclose(log);

    fprintf(stdout, "wrote %s\n", outLog);

    /* Cleanup. */
    freeExemplarBuffer(exemplars);
    freePpcaReplaySet(set);
    freeReservedMemory(R);
    freeReservedMemory(trainSets);
    freeReservedMemory(evalSets);

    return 0;
}
