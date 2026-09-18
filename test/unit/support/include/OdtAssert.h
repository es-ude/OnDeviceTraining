#ifndef ODT_ASSERT_H
#define ODT_ASSERT_H

#include <stddef.h>
#include <stdio.h>

#include "unity.h"

/* Per-element size_t array comparison.
 *
 * Unity 2.5.2's TEST_ASSERT_EQUAL_size_t_ARRAY routes to the UINT array assert
 * (unity.h:305), which strides UNITY_INT_WIDTH/8 = 4 bytes per element; on an
 * LP64 host a size_t is 8 bytes, so only the first ceil(N/2) elements are
 * compared and the tail is never read. This macro compares every element
 * through the scalar path (64-bit under UNITY_SUPPORT_64) and names the
 * failing index. Use it for every size_t array in test/. */
#define ODT_ASSERT_EQUAL_size_t_ARRAY(expected, actual, numElements)                               \
    do {                                                                                           \
        for (size_t odtIdx_ = 0; odtIdx_ < (size_t)(numElements); odtIdx_++) {                     \
            char odtMsg_[48];                                                                      \
            snprintf(odtMsg_, sizeof odtMsg_, "element %zu", odtIdx_);                             \
            TEST_ASSERT_EQUAL_size_t_MESSAGE((expected)[odtIdx_], (actual)[odtIdx_], odtMsg_);     \
        }                                                                                          \
    } while (0)

#endif // ODT_ASSERT_H
