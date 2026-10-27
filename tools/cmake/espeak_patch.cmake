# Upstream 1.52 fetches Sonic even when USE_LIBSONIC=OFF. Respect that option
# so a phonemizer-only build has no unnecessary transitive network dependency.
set(path "${SOURCE_DIR}/cmake/deps.cmake")
file(READ "${path}" contents)
string(REPLACE "else()\n  FetchContent_Declare(sonic-git" "elseif(USE_LIBSONIC)\n  FetchContent_Declare(sonic-git" contents "${contents}")
file(WRITE "${path}" "${contents}")
# BEGIN UNSLOTH_WINDOWS_COMPAT
# The native data generator calls _setmode, declared in the Windows CRT's
# io.h. MSVC tolerated the implicit declaration; clang's C frontend refuses it.
if(WIN32)
    set(path "${SOURCE_DIR}/src/espeak-ng.c")
    file(READ "${path}" contents)
    if(NOT contents MATCHES "#include <io\\.h>")
        string(REPLACE "#include <fcntl.h>"
            "#include <fcntl.h>\n#ifdef _WIN32\n#include <io.h>\n#endif" contents "${contents}")
        file(WRITE "${path}" "${contents}")
    endif()
    # Its compatibility macro casts every pointer to char*, including argv.
    # Keep the argument vector's pointer depth for clang's C type checking.
    set(path "${SOURCE_DIR}/src/compat/getopt.c")
    file(READ "${path}" contents)
    string(REPLACE "__UNCONST(nargv)" "(char **)(nargv)" contents "${contents}")
    file(WRITE "${path}" "${contents}")
endif()
# END UNSLOTH_WINDOWS_COMPAT
set(path "${SOURCE_DIR}/src/CMakeLists.txt")
file(READ "${path}" contents)
if(NOT contents MATCHES "ARCHIVE_OUTPUT_NAME espeak-ng-tool")
    string(APPEND contents "\n# Avoid colliding with the static espeak-ng library on MSVC.\nset_target_properties(espeak-ng-bin PROPERTIES ARCHIVE_OUTPUT_NAME espeak-ng-tool)\n")
    file(WRITE "${path}" "${contents}")
endif()
