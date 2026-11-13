#ifndef ODT_ROW_BROADCAST_H
#define ODT_ROW_BROADCAST_H

#include "Tensor.h"

/* Row-broadcast primitives on a FLOAT32 matrix mat [r,c] and out [r,c].
 * All operands are read and written in flat storage order, so pass materialized
 * identity-order tensors, never transpose views. out may alias mat. */

/* out[i][j] = rowScales[i] * mat[i][j]; rowScales is [r]. */
void scaleRowsFloat32(tensor_t *mat, tensor_t *rowScales, tensor_t *out);

/* out[i][j] = mat[i][j] - row[j]; row is [c]. */
void subRowBroadcastFloat32(tensor_t *mat, tensor_t *row, tensor_t *out);

#endif // ODT_ROW_BROADCAST_H
