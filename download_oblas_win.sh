#! /usr/bin/bash

# TODO: (kelreeeeey) maybe use bat instead of bash for the sake of
# nativeness

# this script is meant for windows user
# it automates downloading and unpacking openblas-v0.3.30
# to TARGET_DIR based on the machine bit, currently
# it only support for 64x and 86x arch.

# TODO: (kelreeeeey) these are hardcoded
BLAS_64="OpenBLAS-0.3.30-x64.zip"
BLAS_32="OpenBLAS-0.3.30-x86.zip"
OBLAS_LINK="https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/"
TARGET_DIR="anvil/lib/windows/"
mkdir -p $TARGET_DIR"openblas"

OS_BIT=$(wmic os get osarchitecture | grep bit)

# download oblas based on the os bit
# TODO: (kelreeeeey) figure out for other archs.
if [ $OS_BIT == "64-bit" ]; then
    wget $OBLAS_LINK$BLAS_64 -P $TARGET_DIR
    DOWNLOADED=$TARGET_DIR$BLAS_64
else
    wget $OBLAS_LINK$BLAS_32 -P $TARGET_DIR
    DOWNLOADED=$TARGET_DIR$BLAS_32
fi

# unzip and remove the archive
yes | unzip $DOWNLOADED -d $TARGET_DIR"openblas"
rm $DOWNLOADED*
