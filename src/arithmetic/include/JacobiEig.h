#ifndef ODT_JACOBI_EIG_H
#define ODT_JACOBI_EIG_H

#include <stddef.h>

#include "Tensor.h"

/* Symmetric eigendecomposition by cyclic Jacobi rotations (Golub & Van Loan,
 * Matrix Computations, 8.5). All tensors FLOAT32, identity order.
 *
 * a          symmetric [p,p]; destroyed (it is the workspace, so no hidden scratch).
 * eigvalsOut [p], sorted descending.
 * eigvecsOut [p,p], row i = unit eigenvector of eigvalsOut[i] (PPCA's row
 *            convention, so rotating back is a plain matmul).
 * maxSweeps  hard cap on sweeps (deterministic termination).
 * tol        stop when sqrt(sum of squared off-diagonals) < tol. */
void jacobiEigSymFloat32(tensor_t *a, tensor_t *eigvalsOut, tensor_t *eigvecsOut, size_t maxSweeps,
                         float tol);

#endif // ODT_JACOBI_EIG_H
