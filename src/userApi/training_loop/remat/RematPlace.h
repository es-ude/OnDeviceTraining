#ifndef ODT_REMAT_PLACE_H
#define ODT_REMAT_PLACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "RematPlan.h"

/* ARENA placement (#4, spec §5.5), private to the ARENA row; PAGED reuses it
 * later. Offsets never leave the row (R5): the shared table and plan stay
 * placement-free. */

/* Fixed, NOT -D-overridable: dumps, Python mirror, oracle and rig must agree.
 * A packed BFP wire can end at any byte; 8 keeps every offset legal for the
 * float and int32 kernels, at most 7 B per packed range. */
#define ODT_WIRE_ALIGN 8u
_Static_assert((ODT_WIRE_ALIGN & (ODT_WIRE_ALIGN - 1u)) == 0u, "power of two");
_Static_assert(ODT_WIRE_ALIGN >= _Alignof(float) && ODT_WIRE_ALIGN >= _Alignof(int32_t),
               "fits every wire dtype");
_Static_assert(ODT_WIRE_ALIGN <= _Alignof(max_align_t),
               "reserveMemory block starts are max-aligned: calloc (StorageApi.c:72-74), or the "
               "max_align_t header under ODT_MEM_PROFILE (StorageApi.c:15-18, :32-45)");

/* roundUp(bytes(w), ODT_WIRE_ALIGN), checked (D60): exits naming the wire. */
size_t arenaPlaced(const rematWireTable_t *t, uint16_t w);

/* First-fit-decreasing into offsets[range id]: placed size descending, then
 * begin, then wire id; each range takes the lowest offset that is 0 or the end
 * of a co-live (inclusive) placed range and overlaps no co-live placed range.
 * bytes = max(offset + placed); peakPlacedBytes = the peak of the placed sums
 * over the steps. Returns false only when its numRanges x uint16_t scratch
 * cannot be reserved. */
bool arenaPlaceFirstFitDecreasing(const rematWireTable_t *t, const rematProgram_t *p,
                                  size_t *offsets, size_t *bytes, size_t *peakPlacedBytes);

#endif // ODT_REMAT_PLACE_H
