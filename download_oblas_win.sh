#! /usr/bin/bash

BLAS_64="OpenBLAS-0.3.30-x64.zip"
BLAS_32="OpenBLAS-0.3.30-x86.zip"
OBLAS_LINK="https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/"
TARGET_DIR="anvil/lib/windows/"
mkdir -p $TARGET_DIR"openblas"


OS_BIT=$(wmic os get osarchitecture | grep bit)


# download oblas based on the os bit
if [ $OS_BIT == "64-bit" ]; then
    wget $OBLAS_LINK$BLAS_64 -P $TARGET_DIR
    DOWNLOADED=$TARGET_DIR$BLAS_64
else
    wget $OBLAS_LINK$BLAS_32 -P $TARGET_DIR
    DOWNLOADED=$TARGET_DIR$BLAS_32
fi

# unzip and remove the archive
unzip $DOWNLOADED -d $TARGET_DIR"openblas"
rm $DOWNLOADED
