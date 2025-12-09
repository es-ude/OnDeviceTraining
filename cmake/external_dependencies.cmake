include(FetchContent)

macro(add_unity)
    FetchContent_Declare(
            unity
            GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
            GIT_TAG v2.5.2
            OVERRIDE_FIND_PACKAGE)
    FetchContent_MakeAvailable(unity)
    find_package(unity)
endmacro()

macro(add_ctest)
    include(CTest)
endmacro()