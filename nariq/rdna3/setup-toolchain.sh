#!/usr/bin/env bash
# Fetch a local Vulkan SDK (glslc + headers + SPIRV-Headers cmake config) and
# configure/build the ggml-vulkan backend of this fork.
#
# Why a local SDK: Ubuntu 22.04 (jammy) ships no glslc package and Vulkan
# headers 1.3.204; the ggml-vulkan build wants glslc and newer headers.
# No root required.
#
# Usage:
#   bash nariq/rdna3/setup-toolchain.sh            # fetch SDK + configure + build
#   bash nariq/rdna3/setup-toolchain.sh configure  # skip SDK download if present
#
# Low-memory hosts (<4GB RAM): the generated mul_mm.comp.cpp (~79MB of SPIR-V
# byte arrays) OOMs gcc at any -O. Build with -O1 (script default) and split
# that TU with nariq/rdna3/lowmem-build.py, then rerun cmake --build.
set -euo pipefail

SDKVER="${VULKAN_SDK_VERSION:-1.4.363.1}"
SDKROOT="${VULKAN_SDK_ROOT:-$HOME/.local/vulkansdk}"
SDK="$SDKROOT/$SDKVER/x86_64"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${BUILD_DIR:-$REPO/build-vk}"

if [ ! -x "$SDK/bin/glslc" ]; then
    mkdir -p "$SDKROOT"
    echo ">> downloading Vulkan SDK $SDKVER"
    curl -fSL -o "$SDKROOT/vulkansdk.tar.xz" \
        "https://sdk.lunarg.com/sdk/download/$SDKVER/linux/vulkansdk-linux-x86_64-$SDKVER.tar.xz"
    tar -xf "$SDKROOT/vulkansdk.tar.xz" -C "$SDKROOT"
fi

# The tarball ships headers/tools but no loader; link the system libvulkan in.
if [ ! -e "$SDK/lib/libvulkan.so" ]; then
    SYSVK="$(ldconfig -p | awk '/libvulkan\.so\.1/{print $NF; exit}')"
    [ -n "$SYSVK" ] || { echo "no system libvulkan.so.1 (install mesa-vulkan-drivers)"; exit 1; }
    ln -sf "$SYSVK" "$SDK/lib/libvulkan.so"
fi

export VULKAN_SDK="$SDK"
export PATH="$SDK/bin:$PATH"

cmake -B "$BUILD" -S "$REPO" \
    -DGGML_VULKAN=ON -DLLAMA_CURL=OFF \
    -DCMAKE_PREFIX_PATH="$SDK" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS='-g0' -DCMAKE_C_FLAGS='-g0' \
    -DCMAKE_CXX_FLAGS_RELEASE='-O1' -DCMAKE_C_FLAGS_RELEASE='-O1' \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "$BUILD" --target ggml-vulkan -j"$(nproc)" || {
    echo ">> if cc1plus was OOM-killed on mul_mm.comp.cpp, run:"
    echo "   python3 $REPO/nariq/rdna3/lowmem-build.py && cmake --build $BUILD --target ggml-vulkan"
    exit 1
}
echo ">> OK: $BUILD/bin/libggml-vulkan.so"
