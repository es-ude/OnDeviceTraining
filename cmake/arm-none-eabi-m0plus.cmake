# Cortex-M0+ cross toolchain (Thumb-1, no FPU, no hardware divide): every float
# op and every integer division becomes a visible libgcc call, which is what
# tools/bfp-census/census.py counts. Compile-only, like arm-none-eabi.cmake.
#
# The -mcpu/-mthumb/-O2 flags live in the arm_cross_m0plus preset's
# CMAKE_C_FLAGS: a preset-provided CMAKE_C_FLAGS replaces CMAKE_C_FLAGS_INIT,
# so putting them here would be silently dropped.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)

# No OS, no linkable runtime: compile-only. Static-library try-compile keeps
# CMake's compiler check from demanding _exit/_sbrk stubs at configure time.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
