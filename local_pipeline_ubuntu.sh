#!/usr/bin/env bash
set -eou pipefail

CLEAN_BUILD=false
BUILD_DEBUG=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--clean-build)
        CLEAN_BUILD=true
        shift # past argument
        ;;
        -d|--debug)
        BUILD_DEBUG="-DCMAKE_BUILD_TYPE=Debug"
        shift # past argument
        ;;
        -h|--help)
        printf "Usage: $0 [-c] [-d] \n
This scripts configures, builds, and executes the dogm library, unit tests, and the demo.
You can use the following flags:\n
-c      Clean build (removes previous build files)
-d      Debug build\n"
        exit 0
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

find dogm/demo dogm/include dogm/src dogm/test -iname '*.h' -o -iname '*.cpp' -o -iname '*.cu' | xargs clang-format -i
mkdir -p build && cd build
if [ "$CLEAN_BUILD" = true ]; then rm -rf *; fi
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${BUILD_DEBUG}" ../dogm
make -j $(nproc)

# TODO: extend to check more/all .cpp (and .cu) files
cd ..
find dogm/demo/utils -iname '*.cpp' | xargs clang-tidy -p build

ctest . -j $(nproc)
./demo/demo
