#ifndef ODT_REMAT_ROWS_H
#define ODT_REMAT_ROWS_H

#include <stdbool.h>

#include "RematScheduler.h"

/* The rows' entry points. External linkage because the const
 * vtable in RematScheduler.c names them. Included only by RematScheduler.c
 * and the rows' own files; everything else reaches a row through the
 * dispatch or rematSchedulerFunctions[]. */

/* After the shared rematWireTableBind: requires the reserved arena, restarts
 * the walk. */
void rematArenaBegin(rematScheduler_t *s);
/* Binds every range that opens at the next step, then hands the step out;
 * false once the stream is complete. */
bool rematArenaNext(rematScheduler_t *s, rematStep_t *step);
/* Releases every range that closes at the step next() handed out. */
void rematArenaDone(rematScheduler_t *s, const rematStep_t *step);
/* Before the shared rematWireTableUnbind: every step done, every range closed. */
void rematArenaEnd(rematScheduler_t *s);

/* The row's own blocks only; safe after a failed init and repeatable. */
void rematArenaDeinit(rematScheduler_t *s);

/* After the shared rematWireTableBind: restarts the walk. */
void rematHeapBegin(rematScheduler_t *s);
/* Reserves and binds an exactly-sized block for every range that opens at the
 * next step, then hands the step out; false once the stream is complete. */
bool rematHeapNext(rematScheduler_t *s, rematStep_t *step);
/* Releases, then frees, every range that closes at the step next() handed out. */
void rematHeapDone(rematScheduler_t *s, const rematStep_t *step);
/* Before the shared rematWireTableUnbind: every step done, every block freed. */
void rematHeapEnd(rematScheduler_t *s);
/* HEAP keeps no private state: a block never outlives the call that opened
 * its range. */
void rematHeapDeinit(rematScheduler_t *s);

/* For rows that walk the static plan (ARENA, HEAP; PAGED later): every step
 * done and every range closed, else exits naming the row. EVICT ignores the
 * walk and checks its own end. */
void rematRequireWalkComplete(const rematScheduler_t *s, const char *row);

#endif // ODT_REMAT_ROWS_H
