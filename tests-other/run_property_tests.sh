#!/usr/bin/env bash
# Copyright 2026 The SPIRV-Cross Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${PROPERTY_BUILD_ROOT:-/tmp/spirv-cross-property}"
ref_dir="${build_root}/reference"
asan_dir="${build_root}/asan"

cmake -S "${root}" -B "${ref_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPIRV_CROSS_ENABLE_TESTS=OFF \
    -DSPIRV_CROSS_SKIP_INSTALL=ON
cmake --build "${ref_dir}" --target spirv-cross -j "${JOBS:-4}"

cmake -S "${root}" -B "${asan_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPIRV_CROSS_ENABLE_TESTS=OFF \
    -DSPIRV_CROSS_SKIP_INSTALL=ON \
    -DSPIRV_CROSS_SANITIZE_ADDRESS=ON \
    -DSPIRV_CROSS_SANITIZE_UNDEFINED=ON \
    -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \
    -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer"
cmake --build "${asan_dir}" --target spirv-cross -j "${JOBS:-4}"

exec python3 "${root}/tests-other/property_test.py" \
    --candidate "${asan_dir}/spirv-cross" \
    --reference "${ref_dir}/spirv-cross" \
    --artifacts "${root}/artifacts/property" \
    --timeout "${PROPERTY_TIMEOUT:-30}" \
    "$@"
