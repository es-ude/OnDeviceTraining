# Every enum switch must name all cases and every non-void function must return: the
# code removes defensive default: exits, so a new enum value must fail the build instead.
# Included by the root (whole ODT tree) and by src/ (a project that only
# add_subdirectory()s src/); the variable keeps src/ from adding them a second time.
if(NOT ODT_COMPILE_GUARDS_APPLIED)
    set(ODT_COMPILE_GUARDS_APPLIED TRUE)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Werror=switch;-Werror=return-type>")
endif()
