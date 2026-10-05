#ifndef ODT_TEST_REMAT_TEST_DECORATORS_H
#define ODT_TEST_REMAT_TEST_DECORATORS_H

/* Pass-through slots for decorator rows (remat D24): a test points its own
 * instance's fns at a const table whose slots wrap these. Each delegates by
 * s->type, which a decorated instance keeps, so one decorator table serves
 * the ARENA and the HEAP row alike. The decorated table's name is what every
 * checker message prints. */

#include <stdbool.h>

#include "RematScheduler.h"

static inline void decoratedBegin(rematScheduler_t *s) {
    rematSchedulerFunctions[s->type].begin(s);
}

static inline bool decoratedNext(rematScheduler_t *s, rematStep_t *st) {
    return rematSchedulerFunctions[s->type].next(s, st);
}

static inline void decoratedDone(rematScheduler_t *s, const rematStep_t *st) {
    rematSchedulerFunctions[s->type].done(s, st);
}

static inline void decoratedEnd(rematScheduler_t *s) {
    rematSchedulerFunctions[s->type].end(s);
}

static inline void decoratedDeinit(rematScheduler_t *s) {
    rematSchedulerFunctions[s->type].deinit(s);
}

#endif // ODT_TEST_REMAT_TEST_DECORATORS_H
