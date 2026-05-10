#define SOURCE_FILE "ecg_anomaly_ae_train_c"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int main(void) {
    return 0;
}
