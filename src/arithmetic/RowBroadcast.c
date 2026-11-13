#define SOURCE_FILE "ROW_BROADCAST"

#include "RowBroadcast.h"
#include "Common.h"
#include "Tensor.h"

void scaleRowsFloat32(tensor_t *mat, tensor_t *rowScales, tensor_t *out) {
    size_t r = mat->shape->dimensions[0];
    size_t c = mat->shape->dimensions[1];
    float *m = (float *)mat->data;
    float *s = (float *)rowScales->data;
    float *o = (float *)out->data;
    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            o[i * c + j] = s[i] * m[i * c + j];
        }
    }
}

void subRowBroadcastFloat32(tensor_t *mat, tensor_t *row, tensor_t *out) {
    size_t r = mat->shape->dimensions[0];
    size_t c = mat->shape->dimensions[1];
    float *m = (float *)mat->data;
    float *v = (float *)row->data;
    float *o = (float *)out->data;
    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            o[i * c + j] = m[i * c + j] - v[j];
        }
    }
}
