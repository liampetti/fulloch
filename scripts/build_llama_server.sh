#!/usr/bin/env bash
# Build the exact llama.cpp revision used by the MTP local-LLM backend.
set -euo pipefail

ref="e3546c7948e3af463d0b401e6421d5a4c2faf565"
root="$(git rev-parse --show-toplevel)"
source_dir="$root/.cache/llama-cpp/source"
build_dir="$source_dir/build"
output_dir="$root/.cache/llama-cpp"

rm -rf "$source_dir"
git init "$source_dir"
git -C "$source_dir" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$source_dir" fetch --depth 1 origin "$ref"
git -C "$source_dir" checkout --detach FETCH_HEAD
cmake -S "$source_dir" -B "$build_dir" -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build "$build_dir" --target llama-server -j "${MAX_JOBS:-4}"
cp "$build_dir/bin/llama-server" "$output_dir/llama-server"
