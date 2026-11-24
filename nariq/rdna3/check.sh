#!/usr/bin/env bash
# Local checks only: never downloads model weights or starts a server.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${NARIQ_RDNA3_OUT:-build-rdna3-shaders}
mkdir -p "$out"
compiler=${GLSLANG_VALIDATOR:-glslangValidator}
validator=${SPIRV_VAL:-spirv-val}
"$compiler" -V --target-env vulkan1.1 -S comp nariq/rdna3/shaders/product.comp -o "$out/product.spv"
"$compiler" -V --target-env vulkan1.1 -S comp -DDECODE_ONLY=1 nariq/rdna3/shaders/product.comp -o "$out/decode.spv"
"$validator" --target-env vulkan1.1 "$out/product.spv"
"$validator" --target-env vulkan1.1 "$out/decode.spv"
for variant in side gather; do
    extra=()
    if [[ "$variant" == gather ]]; then extra=(-DGATHER=1); fi
    "$compiler" -V --target-env vulkan1.1 -S comp "${extra[@]}" nariq/rdna3/shaders/side.comp -o "$out/$variant.spv"
    "$validator" --target-env vulkan1.1 "$out/$variant.spv"
done
python3 nariq/rdna3/tests/test_product.py "$out" | tee "$out/results.json"
