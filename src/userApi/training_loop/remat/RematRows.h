#ifndef ODT_REMAT_ROWS_H
#define ODT_REMAT_ROWS_H

#include <stdbool.h>

#include "RematScheduler.h"

/* The rows' entry points (spec §2.1). External linkage because PR1c's const
 * vtable in RematScheduler.c names them. Included only by RematScheduler.c,
 * the row's own file, and the row's unit test, which drives the entry points
 * directly until the dispatch exists. */

/* The row's own blocks only; safe after a failed init and repeatable. */
void rematArenaDeinit(rematScheduler_t *s);

#endif // ODT_REMAT_ROWS_H
