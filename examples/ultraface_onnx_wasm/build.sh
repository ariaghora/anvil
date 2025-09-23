#!/bin/bash -eu

ODIN_PATH=$(odin root)
BASE_DIR="examples/ultraface_onnx_wasm"
OUT_DIR=$BASE_DIR"/build"

mkdir -p $OUT_DIR

FLAGS="-o:speed -no-bounds-check -target:js_wasm32"
cp $ODIN_PATH/core/sys/wasm/js/odin.js $OUT_DIR
cp $BASE_DIR/index.html $OUT_DIR

odin build $BASE_DIR $FLAGS -out:$OUT_DIR/out.wasm
