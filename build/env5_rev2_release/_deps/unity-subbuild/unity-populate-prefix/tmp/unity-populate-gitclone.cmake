# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/unity-populate-gitclone-lastrun.txt" AND EXISTS "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/unity-populate-gitinfo.txt" AND
  "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/unity-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/unity-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/unity-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/nix/store/4qhdhmi7pzgad0zfd7c5lsg235mbf9hv-git-2.51.2/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/ThrowTheSwitch/Unity.git" "unity-src"
    WORKING_DIRECTORY "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/ThrowTheSwitch/Unity.git'")
endif()

execute_process(
  COMMAND "/nix/store/4qhdhmi7pzgad0zfd7c5lsg235mbf9hv-git-2.51.2/bin/git"
          checkout "v2.5.2" --
  WORKING_DIRECTORY "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v2.5.2'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/nix/store/4qhdhmi7pzgad0zfd7c5lsg235mbf9hv-git-2.51.2/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-src"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/unity-populate-gitinfo.txt" "/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/unity-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/jan/Arbeit/OnDeviceTraining/build/env5_rev2_release/_deps/unity-subbuild/unity-populate-prefix/src/unity-populate-stamp/unity-populate-gitclone-lastrun.txt'")
endif()
