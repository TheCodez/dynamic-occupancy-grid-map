#!/usr/bin/env bash
set -e
set -u
set -o pipefail

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
This scripts configures, builds, and executes the dogm library, unit tests, and the demo.\n
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


mkdir -p build
cd build
if [ "$CLEAN_BUILD" = true ]; then rm -rf *; fi
cmake ../dogm  "${BUILD_DEBUG}"
cmake --build build

# Run unit tests
ctest . -j $(nproc)

# Run demo application
./demo/demo