#ifndef ODT_REMAT_PLACE_H
#define ODT_REMAT_PLACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "RematPlan.h"

/* ARENA placement (#4), private to the ARENA row; PAGED reuses it
 * later. Offsets never leave the row: the shared table and plan stay
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

/* Manual poisoning of the resident arena. The runtime entry points
 * are declared here rather than taken from <sanitizer/asan_interface.h>, which
 * the pinned devenv clang does not ship (the AsanDeath.h precedent). 8 is the
 * ASan shadow granule, so every range starts granule-aligned and an exact
 * unpoison leaves the pad poisoned. */
#if defined(__SANITIZE_ADDRESS__)
#define ODT_REMAT_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ODT_REMAT_ASAN 1
#endif
#endif

#ifdef ODT_REMAT_ASAN
_Static_assert(ODT_WIRE_ALIGN % 8u == 0u, "an exact unpoison keeps the pad poisoned only when "
                                          "every range starts on an 8-byte ASan granule");
void __asan_poison_memory_region(void const volatile *addr, size_t size);
void __asan_unpoison_memory_region(void const volatile *addr, size_t size);
#define ODT_ASAN_POISON(addr, size) __asan_poison_memory_region((addr), (size))
#define ODT_ASAN_UNPOISON(addr, size) __asan_unpoison_memory_region((addr), (size))
#else
#define ODT_ASAN_POISON(addr, size) ((void)(addr), (void)(size))
#define ODT_ASAN_UNPOISON(addr, size) ((void)(addr), (void)(size))
#endif

/* roundUp(bytes(w), ODT_WIRE_ALIGN), overflow-checked: exits naming the wire. */
size_t arenaPlaced(const rematWireTable_t *t, uint16_t w);

/* The peak of the placed (ODT_WIRE_ALIGN-rounded) live sums over the steps:
 * the placedPeak the members take and the portfolio reports. */
size_t arenaPlacedPeakBytes(const rematWireTable_t *t, const rematProgram_t *p);

/* First-fit-decreasing into offsets[range id]: placed size descending, then
 * begin, then wire id; each range takes the lowest offset that is 0 or the end
 * of a co-live (inclusive) placed range and overlaps no co-live placed range.
 * bytes = max(offset + placed). Returns false only when its numRanges x
 * uint16_t scratch cannot be reserved. placedPeak is unused. */
bool arenaPlaceFirstFitDecreasing(const rematWireTable_t *t, const rematProgram_t *p,
                                  size_t placedPeak, size_t *offsets, size_t *bytes);

/* The other portfolio members. Each places every range into
 * offsets[range id], sets bytes = max(offset + placed) and returns false only
 * when its scratch cannot be reserved. placedPeak is the peak of the placed
 * sums over the steps, which ping-pong uses as its height. */

/* First fit in the order: placed size descending, then end descending, then
 * begin, then range id. */
bool arenaPlaceSizeThenEnd(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                           size_t *offsets, size_t *bytes);

/* Ranges in begin order (then end descending, placed size descending, range
 * id). Over the gaps between its co-live placed ranges a range has a lowest
 * offset that fits and a highest one whose end stays at or below placedPeak;
 * it takes the highest when the co-live placed range that began last lies in
 * the lower half (2 x offset + placed < placedPeak), else the lowest, so a
 * chain of buffers alternates between bottom and top. */
bool arenaPlacePingPong(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                        size_t *offsets, size_t *bytes);

/* Busiest step first: the steps by live placed bytes descending (the earlier
 * step on ties); at each step its ranges not yet ordered, by placed size x
 * lifetime in steps descending, then wire id; then first fit in that order. */
bool arenaPlaceBreadthArea(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                           size_t *offsets, size_t *bytes);

/* The ranges alive across the first backward step (the earliest begin of a
 * GRAD range) first fit from the bottom in (begin, end descending, range id);
 * the others first fit on their own in (begin, end descending, placed size
 * descending, range id) and are mirrored from the top of the lowest height at
 * which no co-live pair of the two groups overlaps. A plan without a GRAD
 * range exits naming it (arenaPlanHasGradRange). */
bool arenaPlaceTwoEnded(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                        size_t *offsets, size_t *bytes);

/* False for a plan without a backward (no GRAD wire has a range). */
bool arenaPlanHasGradRange(const rematWireTable_t *t, const rematProgram_t *p);

/* ARENA's placement: runs first-fit-decreasing, breadth/area, ping-pong,
 * two-ended (only on a plan with a GRAD range) and size-then-end, in that
 * fixed order, and keeps the layout with the fewest bytes; on equal bytes the
 * earlier member's. So no layout is larger than first-fit-decreasing's.
 * peakPlacedBytes = arenaPlacedPeakBytes. Returns false only when
 * a scratch block cannot be reserved: the trial layouts' numRanges x size_t
 * block, or a member's own; every one is released before it returns, so the
 * init reserves no byte its report does not account for. */
bool arenaPlacePortfolio(const rematWireTable_t *t, const rematProgram_t *p, size_t *offsets,
                         size_t *bytes, size_t *peakPlacedBytes);

/* Always runs (firmware too) and trusts no layout, imported ones included:
 * every offset a multiple of ODT_WIRE_ALIGN, every range inside [0,
 * bytes), co-live (inclusive) ranges byte-disjoint. Exits naming the wire(s). */
void arenaVerifyPlacement(const rematWireTable_t *t, const rematProgram_t *p, const size_t *offsets,
                          size_t bytes);

#endif // ODT_REMAT_PLACE_H
