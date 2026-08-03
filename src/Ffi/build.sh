#!/usr/bin/env bash
# Compiles kernel.c into a shared library used by FlashAttention\Kernel\FfiKernel.
# Safe to re-run; only needed once per machine/architecture.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/libflashkernel.so"

if [[ "$(uname)" == "Darwin" ]]; then
    OUT="$DIR/libflashkernel.dylib"
fi

gcc -O3 -march=native -fPIC -shared -o "$OUT" "$DIR/kernel.c"

echo "Built: $OUT"
