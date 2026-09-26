#ifndef ODT_REMAT_ROWS_H
#define ODT_REMAT_ROWS_H

#include <stdbool.h>

#include "RematScheduler.h"

/* The rows' entry points (spec §2.1). External linkage because PR1c's const
 * vtable in RematScheduler.c names them. Included only by RematScheduler.c,
 * the row's own file, and the row's unit test, which drives the entry points
 * directly until the dispatch exists. */

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

#endif // ODT_REMAT_ROWS_H
