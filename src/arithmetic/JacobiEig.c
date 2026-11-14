#define SOURCE_FILE "JACOBI_EIG"

#include "JacobiEig.h"
#include "Common.h"
#include "Reduce.h"
#include "Tensor.h"

static float offDiagNorm(const float *m, size_t p) {
    float acc = 0.0f;
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            if (i != j) {
                acc += m[i * p + j] * m[i * p + j];
            }
        }
    }
    return sqrtFloat32(acc);
}

void jacobiEigSymFloat32(tensor_t *a, tensor_t *eigvalsOut, tensor_t *eigvecsOut, size_t maxSweeps,
                         float tol) {
    size_t p = a->shape->dimensions[0];

    float *m = (float *)a->data;
    float *v = (float *)eigvecsOut->data;

    /* V starts as identity; we accumulate rotations as ROWS (V row i ends up
     * as the eigenvector of the i-th diagonal entry). */
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            v[i * p + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (size_t sweep = 0; sweep < maxSweeps; sweep++) {
        if (offDiagNorm(m, p) < tol) {
            break;
        }
        for (size_t i = 0; i + 1 < p; i++) {
            for (size_t j = i + 1; j < p; j++) {
                float apq = m[i * p + j];
                if (apq == 0.0f) {
                    continue;
                }
                float app = m[i * p + i];
                float aqq = m[j * p + j];
                float theta = (aqq - app) / (2.0f * apq);
                /* t = sgn(theta) / (|theta| + sqrt(theta^2 + 1)) — the
                 * smaller-angle root (Golub & Van Loan 8.5.2). */
                float t;
                if (theta >= 0.0f) {
                    t = 1.0f / (theta + sqrtFloat32(theta * theta + 1.0f));
                } else {
                    t = -1.0f / (-theta + sqrtFloat32(theta * theta + 1.0f));
                }
                float c = 1.0f / sqrtFloat32(t * t + 1.0f);
                float s = t * c;

                /* Update A = J^T A J on rows/cols i and j. */
                for (size_t k = 0; k < p; k++) {
                    float aki = m[k * p + i];
                    float akj = m[k * p + j];
                    m[k * p + i] = c * aki - s * akj;
                    m[k * p + j] = s * aki + c * akj;
                }
                for (size_t k = 0; k < p; k++) {
                    float aik = m[i * p + k];
                    float ajk = m[j * p + k];
                    m[i * p + k] = c * aik - s * ajk;
                    m[j * p + k] = s * aik + c * ajk;
                }
                /* Accumulate rotation into V rows i and j. */
                for (size_t k = 0; k < p; k++) {
                    float vik = v[i * p + k];
                    float vjk = v[j * p + k];
                    v[i * p + k] = c * vik - s * vjk;
                    v[j * p + k] = s * vik + c * vjk;
                }
            }
        }
    }

    /* Extract diagonal, sort DESCENDING (selection sort — p is small), and
     * permute V's rows in lockstep. */
    float *lam = (float *)eigvalsOut->data;
    for (size_t i = 0; i < p; i++) {
        lam[i] = m[i * p + i];
    }
    for (size_t i = 0; i < p; i++) {
        size_t best = i;
        for (size_t j = i + 1; j < p; j++) {
            if (lam[j] > lam[best]) {
                best = j;
            }
        }
        if (best != i) {
            float tmp = lam[i];
            lam[i] = lam[best];
            lam[best] = tmp;
            for (size_t k = 0; k < p; k++) {
                float tv = v[i * p + k];
                v[i * p + k] = v[best * p + k];
                v[best * p + k] = tv;
            }
        }
    }
}
