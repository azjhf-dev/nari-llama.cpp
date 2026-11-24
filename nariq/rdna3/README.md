# NariQ on RDNA3: active port

Status on 2026-11-24: **not yet a usable AMD model runtime**.
Branch: `rdna3-vulkan`, based on Nari commit `8e0e016e0cd13f153380bde2bec01add08eae85e`.

## Backend decision

Vulkan first, in a dedicated Nari fork rather than Avalon. This preserves the
existing loader, tensor IDs, metadata validation, CPU reference and regression
suite without a second merge across different llama.cpp bases.

Nari's CUDA GEMV/GEMM use inline PTX `mma.sync.m16n8k16`, NVIDIA atomics and
warp-specific layouts. HIP is not a mechanical source translation. Its Metal
implementation separates input gather/scaling, Hadamard transforms, compressed
product and output transforms/gather. That decomposition can be expressed in
portable Vulkan compute. The first shaders use 32-bit arithmetic, packed half
scale loads and no fixed subgroup size. RDNA3 wave32/wave64 and WMMA tuning come
only after numerical parity. HIP remains a fallback if Vulkan dispatch overhead
or integration limits are material on hardware, not a promised second backend.

## What is verified

- CPU Release build: `test-trellis`, `test-trellis-metadata`, and
  `test-trellis-model` compile successfully with GCC/CMake on Linux x86-64.
- Two registered CPU trellis CTests pass. They include TRL1-6, transforms,
  permutations, embedding bands, batches and CPU scheduler behavior.
- Python suite: 25 tests, 23 pass, 2 skipped because torch/transformers are absent.
  Native dense decoder fixture enabled via `NARIQ_TEST_LIBRARY`.
- New isolated Vulkan decode/product shader compiles with glslang 11.8, passes
  `spirv-val` for Vulkan 1.1 and **runs on Mesa lavapipe software Vulkan**.
  18 cases cover TRL1-6, random/zero/all-one circular code streams, unrotated
  products, three tokens and an odd packed-half scale count. Decoder output is
  exact against an independent bit-string oracle; FP32 products pass
  `rtol=3e-5, atol=2e-4` against FP64 accumulation. Recorded results are in
  `tests/lavapipe-baseline.json`.

## What is NOT verified or implemented

- No AMD GPU, NVIDIA GPU or real RDNA3 driver run. Software Vulkan is not a GPU
  benchmark and establishes neither RDNA3 compatibility nor useful speed.
- The new shader is **not registered with ggml-vulkan**. No AMD full-model run.
- Input/output transforms, permutations, ggml scheduling, scratch lifetimes and
  backend allocation plumbing are still to be ported.
- No model weights downloaded or evaluated yet. No full-model KL, perplexity,
  generation quality or speed result. CPU synthetic checks do not establish
  correctness against the released model.
- Prototype product is deliberately scalar and slow; no performance claim.

## RX 7900 XTX test box

Ash confirmed the target is a 7900 XTX on Nov 24. OS and driver are not yet
confirmed. Current quick-start below targets Linux; do not assume a Windows
shell has the same dependencies or commands. No full-model memory fit is claimed.

## One-command isolated test on the RDNA3 box

Install Vulkan drivers/loader, `glslangValidator`, `spirv-val`, Python 3, and the
Python packages `numpy` and `vulkan` in your own environment. On Ubuntu the
shader tools are provided by `glslang-tools` and `spirv-tools`.

```sh
NARIQ_VULKAN_DEVICE="7900" bash nariq/rdna3/check.sh
```

Return `build-rdna3-shaders/results.json` and the card, OS and driver versions.
The example requires a device name containing 7900 and fails if none matches.
Without `NARIQ_VULKAN_DEVICE`, the runner chooses the first visible device. Set
`VK_ICD_FILENAMES` to the desired AMD driver when multiple ICDs are present.
The JSON captures OS, device, vendor/device IDs, raw Vulkan API/driver versions
and ICD override. Also include `vulkaninfo --summary` output if available, since
vendor driver version encodings differ. Check the device name before interpreting a pass. No weights, network,
root access, server exposure or paid service is needed by this script.

## Four-week plan, target 2026-12-24

1. Nov 24-30: map all operations, CPU/reference fixtures, portable shader
   primitives, establish model fixture and box details.
2. Dec 1-7: Vulkan transforms and compressed products; integrate ggml backend,
   scheduler and support guards. Keep embeddings on CPU initially, as Nari does.
3. Dec 8-14: real RDNA3 operator tests and teacher-forced full-vocabulary KL
   versus CPU for each released bit budget; deterministic prompt/token fixtures,
   batch/tail coverage and perplexity checks. No speed work before parity.
4. Dec 15-24: measure prefill/decode and memory on actual hardware, then tune
   LDS, workgroups and optional matrix instructions. Re-run parity after tuning.

Dates are targets, not evidence of tests. GPU stages require a test box and
returned logs. The existing `test-trellis-model` plus `nariq/compare_logits.py`
provide bounded teacher-forced captures and full-distribution comparison; use
max KL <= 1e-3 and top-1 agreement 1.0 as the initial gate, not a claim that this
threshold has been met. Full-model CPU evaluation can be slow and is separate
from the isolated one-command shader test.

## License and attribution

Preserve the repository MIT license, `nariq/NOTICE`, exllamav3 notice and vendored
libtrellis Apache-2.0 license. The new GLSL prototype follows the documented
NariQ/exllamav3 packing; it does not copy CUDA implementation code. No upstream
PR, issue, release or community post is part of this branch.
