/*! Important: This experiment expects the MNIST dataset. You can load the dataset using the python
 * script, located in test/unit/data_loader/MNISTLoader.py
 *
 * You might have to change the defined paths below, if locations differ.
 *
 */

#define SOURCE_FILE "MNIST_EXPERIMENT"

#define USE_LOCAL_PATHS 1

#if USE_LOCAL_PATHS
#define MNIST_TEST_X "../../../test/unit/data_loader/mnist_test_x.npy"
#define MNIST_TEST_Y "../../../test/unit/data_loader/mnist_test_y.npy"
#define MNIST_TRAIN_X "../../../test/unit/data_loader/mnist_train_x.npy"
#define MNIST_TRAIN_Y "../../../test/unit/data_loader/mnist_train_y.npy"
#define LOG "../../../example/MnistExperimentLog.csv"

// used for running experiment on remote workstation
#else
#define MNIST_TEST_X "mnist_test_x.npy"
#define MNIST_TEST_Y "mnist_test_y.npy"
#define MNIST_TRAIN_X "mnist_train_x.npy"
#define MNIST_TRAIN_Y "mnist_train_y.npy"
#define LOG "MnistExperimentLog.csv"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "CSVHelper.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LinearApi.h"
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

static dataset_t trainDataset;
static dataset_t testDataset;

static size_t batchSize = 32;

static void initDataSets() {
    tensorArray_t *trainItems = npyLoad(MNIST_TRAIN_X);
    tensorArray_t *trainLabels = npyLoad(MNIST_TRAIN_Y);
    trainDataset.items = trainItems;
    trainDataset.labels = trainLabels;

    tensorArray_t *testItems = npyLoad(MNIST_TEST_X);
    tensorArray_t *testLabels = npyLoad(MNIST_TEST_Y);
    testDataset.items = testItems;
    testDataset.labels = testLabels;
}

static sample_t *getTrainSample(size_t id) {
    sample_t *sample = npyGetSample(&trainDataset, id);
    return sample;
}

static sample_t *getTestSample(size_t id) {
    sample_t *sample = npyGetSample(&testDataset, id);
    return sample;
}

static size_t getTrainDatasetSize() {
    return trainDataset.items->size;
}

static size_t getTestDatasetSize() {
    return testDataset.items->size;
}

static void epochCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
    char row[256] = {0};
    sprintf(row, "%lu, %f, %f, %f, %f, %f, %f\n", epoch, trainLoss, evalStats.loss,
            evalStats.accuracy, evalStats.precision, evalStats.recall, evalStats.f1);
    PRINT_DEBUG("%s\n", row);

    char *rows[] = {row};
    size_t entriesInRow[] = {7};
    csvData_t csvData;
    setCSVData(&csvData, rows, 1, entriesInRow);
    csvWriteRowsByBufferSize(LOG, &csvData, "a");
}

static void writeCsvHeader(char *filePath) {
    char *header =
        "epoch, train_loss, eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1\n";
    char *row[] = {header};
    size_t entriesInRow[] = {7};
    csvData_t csvData;
    setCSVData(&csvData, row, 1, entriesInRow);
    csvWriteRowsByBufferSize(filePath, &csvData, "w");
}

#define MODEL_SIZE 5

static void buildModel(layer_t **model) {
    quantization_t *q = quantizationInitFloat();

    // Flatten [1, 28, 28] -> [1, 784]
    model[0] = flattenLayerInit();

    // Linear 784→20
    static float weight0Data[20 * 28 * 28] = {0};
    static size_t weight0Dims[] = {20, 28 * 28};
    tensor_t *weight0Param = tensorInitWithDistribution(XAVIER_UNIFORM, weight0Data, weight0Dims, 2,
                                                        q, NULL, 28 * 28, 20);
    tensor_t *weight0Grad = gradInitFloat(weight0Param, NULL);
    parameter_t *weight0 = parameterInit(weight0Param, weight0Grad);

    static float bias0Data[20] = {0};
    static size_t bias0Dims[] = {1, 20};
    tensor_t *bias0Param =
        tensorInitWithDistribution(ZEROS, bias0Data, bias0Dims, 2, q, NULL, 1, 20);
    tensor_t *bias0Grad = gradInitFloat(bias0Param, NULL);
    parameter_t *bias0 = parameterInit(bias0Param, bias0Grad);

    model[1] = linearLayerInit(weight0, bias0, q, q, q, q);

    // ReLU
    model[2] = reluLayerInit(q, q);

    // Linear 20→10
    static float weight1Data[10 * 20] = {0};
    static size_t weight1Dims[] = {10, 20};
    tensor_t *weight1Param =
        tensorInitWithDistribution(XAVIER_UNIFORM, weight1Data, weight1Dims, 2, q, NULL, 20, 10);
    tensor_t *weight1Grad = gradInitFloat(weight1Param, NULL);
    parameter_t *weight1 = parameterInit(weight1Param, weight1Grad);

    static float bias1Data[10] = {0};
    static size_t bias1Dims[] = {1, 10};
    tensor_t *bias1Param =
        tensorInitWithDistribution(ZEROS, bias1Data, bias1Dims, 2, q, NULL, 1, 10);
    tensor_t *bias1Grad = gradInitFloat(bias1Param, NULL);
    parameter_t *bias1 = parameterInit(bias1Param, bias1Grad);

    model[3] = linearLayerInit(weight1, bias1, q, q, q, q);

    // Softmax
    model[4] = softmaxLayerInit(q, q);
}

int main(void) {
    writeCsvHeader(LOG);

    size_t numberOfEpochs = 10;
    initDataSets();

    dataLoader_t *trainDataloader =
        dataLoaderInit(getTrainSample, getTrainDatasetSize, batchSize, NULL, NULL, false, 0, true);

    dataLoader_t *testDataloader =
        dataLoaderInit(getTestSample, getTestDatasetSize, 1, NULL, NULL, false, 0, true);

    layer_t *model[MODEL_SIZE];
    buildModel(model);

    optimizer_t *sgd = sgdMCreateOptim(0.001f, 0.9f, 0.f, model, MODEL_SIZE, FLOAT32);

    clock_t start = clock();

    trainingRunResult_t result = trainingRun(
        model, MODEL_SIZE, (lossConfig_t){.funcType = CROSS_ENTROPY, .reduction = REDUCTION_MEAN},
        trainDataloader, testDataloader, sgd, numberOfEpochs, calculateGradsSequential,
        inferenceWithLoss, epochCallback);

    clock_t end = clock();

    double duration_sec = (double)(end - start) / CLOCKS_PER_SEC;
    PRINT_INFO("Training finished in %f seconds\n", duration_sec);
    PRINT_INFO("Final train loss: %f, eval loss: %f\n", result.finalTrainLoss,
               result.finalEvalStats.loss);
    PRINT_INFO("Final accuracy: %.2f%%\n", result.finalEvalStats.accuracy * 100.0f);
}
