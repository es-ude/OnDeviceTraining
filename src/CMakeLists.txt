# Every enum switch must name all cases and every non-void function must return: the
# code removes defensive default: exits, so a new enum value must fail the build instead.
# Set here, not in the root, so a project that only add_subdirectory()s src/ gets it too.
add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Werror=switch;-Werror=return-type>")

add_subdirectory(arithmetic)
add_subdirectory(common)
add_subdirectory(continual_learning)
add_subdirectory(csv)
add_subdirectory(data_loader)
add_subdirectory(layer)
add_subdirectory(loss_functions)
add_subdirectory(serial)
add_subdirectory(optimizer)
add_subdirectory(rng)
add_subdirectory(tensor)
add_subdirectory(userApi)
