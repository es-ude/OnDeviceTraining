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
#define LOG "../../../experiments/MnistExperimentLog.csv"

// used for running experiment on remote workstation
#else
#define MNIST_TEST_X "mnist_test_x.npy"
#define MNIST_TEST_Y "mnist_test_y.npy"
#define MNIST_TRAIN_X "mnist_train_x.npy"
#define MNIST_TRAIN_Y "mnist_train_y.npy"
#define LOG "MnistExperimentLog.csv"
#endif


#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "NPYLoaderApi.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "TensorApi.h"
#include "Tensor.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "SoftmaxApi.h"
#include "InferenceApi.h"
#include "CSVHelper.h"
#include "Common.h"
#include "TrainingLoopApi.h"
#include "CalculateGradsSequential.h"


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

// raw tensor dims look like this: [1, 28, 28]
// we want dims to look like this: [1, 28*28] (linear layer only accepts 2D tensors)
static void flattenItemDims() {

    size_t trainNumberOfTensors = getTrainDatasetSize();
    size_t testNumberOfTensors = getTestDatasetSize();

    size_t numberOfDims = 2;

    shape_t *shape;

    for (size_t i = 0; i < trainNumberOfTensors; i++) {
        shape = trainDataset.items->array[i]->shape;
        size_t *newDims = *reserveMemory(numberOfDims * sizeof(size_t));
        size_t newDimsData[] = {shape->dimensions[0],
                                shape->dimensions[2] * shape->dimensions[2]};
        size_t *newOrder = *reserveMemory(numberOfDims * sizeof(size_t));

        for (size_t j = 0; j < numberOfDims; j++) {
            newDims[j] = newDimsData[j];
            newOrder[j] = j;
        }

        freeReservedMemory(shape->dimensions);
        freeReservedMemory(shape->orderOfDimensions);

        shape->numberOfDimensions = numberOfDims;
        shape->dimensions = newDims;
        shape->orderOfDimensions = newOrder;
    }

    for (size_t i = 0; i < testNumberOfTensors; i++) {
        shape = testDataset.items->array[i]->shape;
        size_t *newDims = *reserveMemory(numberOfDims * sizeof(size_t));
        size_t newDimsData[] = {shape->dimensions[0],
                                shape->dimensions[2] * shape->dimensions[2]};
        size_t *newOrder = *reserveMemory(numberOfDims * sizeof(size_t));
        for (size_t j = 0; j < numberOfDims; j++) {
            newDims[j] = newDimsData[j];
            newOrder[j] = j;
        }

        freeReservedMemory(shape->dimensions);
        freeReservedMemory(shape->orderOfDimensions);

        shape->numberOfDimensions = numberOfDims;
        shape->dimensions = newDims;
        shape->orderOfDimensions = newOrder;

    }
}

static void writeCsvRow(char *filePath, size_t epochIndex, size_t batchIndex, float loss,
                 float validationLoss) {
    char epochChars[32] = {0};
    sprintf(epochChars, "%lu", epochIndex);

    char batchChars[32] = {0};
    sprintf(batchChars, "%lu", batchIndex);

    char lossChars[32] = {0};
    sprintf(lossChars, "%f", loss);

    char validationLossChars[32] = {0};
    sprintf(validationLossChars, "%f", validationLoss);

    char string[4 * 32] = {0};
    strcat(string, epochChars);
    strcat(string, ", ");
    strcat(string, batchChars);
    strcat(string, ", ");
    strcat(string, lossChars);
    strcat(string, ", ");
    strcat(string, validationLossChars);
    strcat(string, "\n");
    PRINT_DEBUG("%s\n", string);

    char *row[] = {string};
    size_t entriesInRow[] = {4};

    csvData_t csvData;
    setCSVData(&csvData, row, 1, entriesInRow);

    csvWriteRowsByBufferSize(filePath, &csvData, "a");
}

static void epochCallback(size_t epoch, float trainLoss, float evalLoss) {
    writeCsvRow(LOG, epoch, 0, trainLoss, evalLoss);
}

static void writeCsvHeader(char *filePath) {
    char *header = "epoch, batch, train_loss, eval_loss\n";
    char *row[] = {header};
    size_t entriesInRow[] = {4};
    csvData_t csvData;
    setCSVData(&csvData, row, 1, entriesInRow);
    csvWriteRowsByBufferSize(filePath, &csvData, "w");
}

#define MODEL_SIZE 4

static void buildModel(layer_t **model) {
    quantization_t *q = quantizationInitFloat();

    // Linear 784→20
    static float weight0Data[20 * 28 * 28] = {0};
    size_t weight0Dims[] = {20, 28 * 28};
    tensor_t *weight0Param = tensorInitWithDistribution(XAVIER_UNIFORM, weight0Data, weight0Dims, 2, q, NULL, 28*28, 20);
    tensor_t *weight0Grad = gradInitFloat(weight0Param, NULL);
    parameter_t *weight0 = parameterInit(weight0Param, weight0Grad);

    static float bias0Data[20] = {0};
    size_t bias0Dims[] = {1, 20};
    tensor_t *bias0Param = tensorInitWithDistribution(ZEROS, bias0Data, bias0Dims, 2, q, NULL, 1, 20);
    tensor_t *bias0Grad = gradInitFloat(bias0Param, NULL);
    parameter_t *bias0 = parameterInit(bias0Param, bias0Grad);

    model[0] = linearLayerInit(weight0, bias0, q, q, q, q);

    // ReLU
    model[1] = reluLayerInit(q, q);

    // Linear 20→10
    static float weight1Data[10 * 20] = {0};
    size_t weight1Dims[] = {10, 20};
    tensor_t *weight1Param = tensorInitWithDistribution(XAVIER_UNIFORM, weight1Data, weight1Dims, 2, q, NULL, 20, 10);
    tensor_t *weight1Grad = gradInitFloat(weight1Param, NULL);
    parameter_t *weight1 = parameterInit(weight1Param, weight1Grad);

    static float bias1Data[10] = {0};
    size_t bias1Dims[] = {1, 10};
    tensor_t *bias1Param = tensorInitWithDistribution(ZEROS, bias1Data, bias1Dims, 2, q, NULL, 1, 10);
    tensor_t *bias1Grad = gradInitFloat(bias1Param, NULL);
    parameter_t *bias1 = parameterInit(bias1Param, bias1Grad);

    model[2] = linearLayerInit(weight1, bias1, q, q, q, q);

    // Softmax
    model[3] = softmaxLayerInit(q, q);
}


int main(void) {
    writeCsvHeader(LOG);

    size_t numberOfEpochs = 10;
    initDataSets();
    flattenItemDims();

    dataLoader_t *trainDataloader = dataLoaderInit(getTrainSample,
                                                   getTrainDatasetSize,
                                                   batchSize,
                                                   NULL,
                                                   NULL,
                                                   false,
                                                   0,
                                                   true);

    dataLoader_t *testDataloader = dataLoaderInit(getTestSample,
                                                  getTestDatasetSize,
                                                  1,
                                                  NULL,
                                                  NULL,
                                                  false,
                                                  0,
                                                  true);

    layer_t *model[MODEL_SIZE];
    buildModel(model);

    optimizer_t *sgd = sgdMCreateOptim(0.001f, 0.f, 0.f, model, MODEL_SIZE, FLOAT32);

    clock_t start = clock();

    trainingRunResult_t result = trainingRun(model, MODEL_SIZE, CROSS_ENTROPY,
                                             trainDataloader, testDataloader, sgd,
                                             numberOfEpochs, calculateGradsSequential,
                                             inferenceWithLoss, epochCallback);

    clock_t end = clock();

    double duration_sec = (double)(end - start) / CLOCKS_PER_SEC;
    PRINT_INFO("Training finished in %f seconds\n", duration_sec);
    PRINT_INFO("Final train loss: %f, eval loss: %f\n", result.finalTrainLoss, result.finalEvalLoss);

    float accuracy = evaluationEpochAccuracy(model, MODEL_SIZE, testDataloader, 10, inference);

    PRINT_INFO("Integration test accuracy: %.2f%%\n", accuracy * 100.0f);
}
