# ggml-vulkan integration notes for NariQ trellis ops (2026-11-28)

References are to rdna3-vulkan @ 2903c82.

## The op contract (from the Metal path, our integration template)

- `GGML_OP_TRELLIS` = coded linear (`ggml/include/ggml.h:439`),
  `GGML_OP_TRELLIS_GET_ROWS` = coded embedding rows (line 440).
- Operand layout (`ggml_metal_op_trellis_supported`, ggml-metal-ops.cpp:6362):
  - src[0] `w`: weight, one of TRL1..6 (ggml.h types 44-49), 2-D matrix, contiguous
  - src[1] `x`: F32, contiguous rows, `x->ne[0] == w->ne[0]`
  - src[2..6] side tensors: F16 `ne0/16`, F32 `ne0`, F32 `ne1`, I32 `ne0`,
    I32 `ne1` (scales/gathers/signs; src[5..6] optional)
  - dst: F32 contiguous
  - `op_params` int32 descriptor `d[]`: `d[6]` = band part count 0..8,
    row0 lives at op_params slot 15 (`GGML_TRELLIS_BAND_ROW0`, ggml.h:1593);
    validate with `ggml_trellis_desc_supported(w->ne[0], w->ne[1], d, 7+d[6])`
  - scratch sizing contract (`ggml_metal_op_trellis_extra`):
    `2*batch*(ne0+ne1)*sizeof(float)`, batch = min(rows(x), TRELLIS_BATCH cap)
- Graph semantics: trellis weight leaves carry GGML_OP_NONE and only trellis
  kernels read them (ggml-metal-device.m:1571). Ops can self-report fusion
  length (`n_fuse` return), so a multi-node coded linear can be one dispatch.

## Vulkan integration points

1. Shaders: add `trellis.comp` under `ggml/src/ggml-vulkan/vulkan-shaders/`.
   The build globs `*.comp` automatically (CMakeLists.txt:221) and
   vulkan-shaders-gen embeds it into a per-file `.comp.cpp`. Port from the
   validated isolated shaders `nariq/rdna3/shaders/{product,side}.comp`;
   side.comp is the quadratic direct-sum correctness baseline, to be
   replaced by staged butterfly transforms for speed (separate milestone).
2. Pipeline registration: device init creates pipelines via the
   `ggml_vk_create_pipeline` lambda (ggml-vulkan.cpp:2211). Add
   `vk_pipeline pipeline_trellis_*` members to the device struct (pattern:
   `pipeline_dequant[GGML_TYPE_COUNT]`, ggml-vulkan-types.h:824).
   `vk_pipeline_struct` (types.h:228) holds module/layout/push-const size/
   wg_denoms/alignment; pipelines are lazily compiled with `compile_pending`.
3. Dispatch: `ggml_vk_compute_forward` (ggml-vulkan.cpp:13471) switches on
   op; add `case GGML_OP_TRELLIS` / `GGML_OP_TRELLIS_GET_ROWS` calling a new
   `ggml_vk_trellis(ctx, subctx, ...)` mirroring `ggml_metal_op_trellis`.
   Generic single-dispatch ops use `ggml_vk_op_f32` (line 10070) as template
   for buffer binding + workgroup math; trellis needs multi-pass dispatch
   (gather/scale -> column transform -> compressed product -> inverse row
   transform), so model on the multi-encoder Metal host code instead.
4. Scratch/sync: per-submit `vk_context` (`vk_context_struct`,
   types.h:1177) carries `sync_staging` plus in/out memcpy lists
   (types.h:1183-1185) and prealloc buffers (types.h:1296). Reuse the
   Metal `extra` sizing formula for the transform scratch and hang it off
   the subctx so sync stays inside the existing submit machinery.
5. Capability gate: `ggml_backend_vk_device_supports_op`
   (ggml-vulkan.cpp:16124) must accept GGML_OP_TRELLIS only when the device
   qualifies: subgroup size handling (RDNA3 wave32/wave64 - never assume 32),
   shared memory >= transform tile, per-tensor size vs `max_buffer_size` /
   `maxStorageBufferRange` (the 10.12GB model is per-tensor fine: largest
   single TRL tensor is far under 4GB), optional fast paths gated on
   `shader_int_dot` / coopmat like the matmul keys do.

## RDNA3 (gfx1100, 7900 XTX) notes

- Dual-issue wave32; expose both subgroup variants or query at pipeline
  creation (wg_denoms pattern already exists for this).
- RADV exposes VK_KHR_cooperative_matrix on RDNA3 - check at runtime, keep a
  non-coopmat fallback (the isolated baseline path).
- `GGML_VK_VISIBLE_DEVICES` is needed only on llvmpipe (CPU-type devices are
  skipped by default); a real 7900 XTX enumerates normally.

## Open items

- Whether test-backend-ops covers GGML_OP_TRELLIS; if not, extend it or run
  the composed TRL1-6 oracle cases from `nariq/rdna3/tests/` through the
  Vulkan backend once dispatch exists.
- GET_ROWS variant scoping (embedding rows) - second priority after linear.
- Fusion (n_fuse) support in ggml_vk_build_graph for trellis chains.
