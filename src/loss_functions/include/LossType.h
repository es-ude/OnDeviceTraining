#ifndef LOSSTYPE_H
#define LOSSTYPE_H

/* Kept apart from LossFunction.h so code that only switches on the loss kind (e.g. Model)
 * does not pull in the loss implementations. */
typedef enum lossFuncType { MSE, CROSS_ENTROPY } lossFuncType_t;

#endif // LOSSTYPE_H
