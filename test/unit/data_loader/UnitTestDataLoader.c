// IMPORTANT: Before running this test, please run the MNISTLoader.py script

#define MNIST_TEST_X "../../../../../test/unit/data_loader/mnist_test_x.npy"
#define MNIST_TEST_Y "../../../../../test/unit/data_loader/mnist_test_y.npy"
#define MNIST_TRAIN_X "../../../../../test/unit/data_loader/mnist_train_x.npy"
#define MNIST_TRAIN_Y "../../../../../test/unit/data_loader/mnist_train_y.npy"

#include <stdlib.h>

#include "Tensor.h"
#include "DataLoaderApi.h"
#include "unity.h"
#include "NPYLoaderApi.h"
#include "RNG.h"
#include "DataLoader.h"

static dataset_t trainDataset;
static dataset_t testDataset;

void setUp() {
    tensorArray_t *trainItems = npyLoad(MNIST_TRAIN_X);
    tensorArray_t *trainLabels = npyLoad(MNIST_TRAIN_Y);
    trainDataset.items = trainItems;
    trainDataset.labels = trainLabels;

    tensorArray_t *testItems = npyLoad(MNIST_TEST_X);
    tensorArray_t *testLabels = npyLoad(MNIST_TEST_Y);
    testDataset.items = testItems;
    testDataset.labels = testLabels;
}

sample_t *getTrainSample(size_t id) {
    sample_t *sample = npyGetSample(&trainDataset, id);
    return sample;
}

size_t getTrainDatasetSize() {
    return trainDataset.items->size;
}

size_t getTestDatasetSize() {
    return testDataset.items->size;
}

sample_t *getTestSample(size_t id) {
    sample_t *sample = npyGetSample(&testDataset, id);
    return sample;
}

void tearDown() {}

void testLoadTrainAndTestDatasets() {
    TEST_ASSERT_EQUAL(60000, trainDataset.items->size);
    TEST_ASSERT_EQUAL(60000, trainDataset.labels->size);

    TEST_ASSERT_EQUAL(10000, testDataset.items->size);
    TEST_ASSERT_EQUAL(10000, testDataset.labels->size);
}

void testNPYGetSample() {
    size_t numberOfValues = 28 * 28;

    dataLoader_t *testDataLoader = dataLoaderInit(getTestSample, getTestDatasetSize, 1000, NULL,
                                                  NULL, false, 0, true);

    sample_t *testSample = testDataLoader->getSample(0);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(testDataset.items->array[0]->data, testSample->item->data,
                                  numberOfValues);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(testDataset.labels->array[0]->data, testSample->label->data,
                                  numberOfValues);
}

void testGetBatch() {
    size_t numberOfValues = 28 * 28;
    size_t batchSize = 5;

    dataLoader_t *testDataLoader = dataLoaderInit(getTestSample, getTestDatasetSize, batchSize,
                                                  NULL, NULL, false, 0, true);

    batch_t *batch = testDataLoader->getBatch(testDataLoader, 0);

    for (size_t i = 0; i < batchSize; i++) {
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(testDataset.items->array[i]->data,
                                      batch->samples[i]->item->data, numberOfValues);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(testDataset.labels->array[i]->data,
                                      batch->samples[i]->label->data, numberOfValues);
    }
}

void testShuffle() {
    size_t numberOfIndices = 10;
    size_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    rngSetSeed(1);
    rngShuffleIndices(indices, 10);

    /*for (size_t i = 0; i < numberOfIndices; i++) {
        printf("%lu\n", indices[i]);
    }*/

    size_t expected[] = {3, 1, 0, 2, 6, 7, 8, 5, 4, 9};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected, indices, numberOfIndices);
}


int main() {
    UNITY_BEGIN();
    RUN_TEST(testLoadTrainAndTestDatasets);
    RUN_TEST(testNPYGetSample);
    RUN_TEST(testGetBatch);
    RUN_TEST(testShuffle);
    return UNITY_END();
}
